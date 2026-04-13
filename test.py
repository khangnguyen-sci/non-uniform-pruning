import os, glob, copy, time
import torch
import torch.nn.functional as F  # Added for padding
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from model import Net


# --- HELPER FUNCTIONS FOR ROBUST INFERENCE ---
def pad_to_multiple(x, mult=64, mode="reflect"):
    """
    Pads the input (N,C,H,W) so H and W are divisible by 'mult'.
    Returns: padded_x, (original_height, original_width)
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    
    # Even if padding is 0, we return the info so loop logic stays consistent
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
        
    # F.pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (h, w)


def crop_back(x, orig_hw, scale=4):
    """
    Crops the SR output back to the correct size.
    CRITICAL: Must multiply original LR dimensions by 'scale' (usually 4).
    """
    h, w = orig_hw
    target_h = h * scale
    target_w = w * scale
    return x[:, :, :target_h, :target_w]
# ---------------------------------------------


def load_ckpt_flexible_sliced(model, ckpt_path: str):
    """
    Load a checkpoint into model:
      - handles module. prefix mismatch
      - slices ckpt tensors if ckpt tensor is larger than model tensor
      - skips keys not found
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_sd = model.state_dict()
    tgt = {k: v.clone() for k, v in model_sd.items()}

    ckpt_keys = list(ckpt.keys())
    ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)
    model_has_module = any(k.startswith("module.") for k in model_sd.keys())

    # Normalize ckpt keys to match model keys
    if ckpt_has_module and not model_has_module:
        ckpt = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    elif (not ckpt_has_module) and model_has_module:
        ckpt = {("module." + k) if not k.startswith("module.") else k: v for k, v in ckpt.items()}

    loaded, skipped = 0, 0
    for k in tgt.keys():
        if k not in ckpt:
            skipped += 1
            continue

        src = ckpt[k]
        dst = tgt[k]

        if src.shape == dst.shape:
            tgt[k].copy_(src)
        else:
            # slice src down to dst
            slices = tuple(slice(0, s) for s in dst.shape)
            try:
                tgt[k].copy_(src[slices])
            except Exception:
                # if slicing still fails, skip
                skipped += 1
                continue
        loaded += 1

    model.load_state_dict(tgt, strict=False)
    return loaded, skipped


def set_env_if_not_none(k, v):
    # IMPORTANT: never write "None" into env
    if v is not None:
        os.environ[k] = str(v)


parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--LR_dir", type=str, default="testset/RealSR/LR")
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")

# ratios (optional; if not given, use whatever is already in env, else default 0.75 in model.py)
parser.add_argument("--r0", type=float, default=None)
parser.add_argument("--r1", type=float, default=None)
parser.add_argument("--r2", type=float, default=None)
parser.add_argument("--r3", type=float, default=None)
parser.add_argument("--rmid", type=float, default=None)

parser.add_argument("--time_file", type=str, default=None)

args = parser.parse_args()

# If user passes ratios, set them. Otherwise, do NOT touch env (SLURM script controls it).
set_env_if_not_none("ADCSR_R0", args.r0)
set_env_if_not_none("ADCSR_R1", args.r1)
set_env_if_not_none("ADCSR_R2", args.r2)
set_env_if_not_none("ADCSR_R3", args.r3)

# RMID defaults to R3 only if user explicitly set R3 but not RMID
if args.rmid is not None:
    set_env_if_not_none("ADCSR_RMID", args.rmid)
elif args.r3 is not None:
    os.environ["ADCSR_RMID"] = str(args.r3)

print("Using ratios:",
      "R0=", os.environ.get("ADCSR_R0", "0.75"),
      "R1=", os.environ.get("ADCSR_R1", "0.75"),
      "R2=", os.environ.get("ADCSR_R2", "0.75"),
      "R3=", os.environ.get("ADCSR_R3", "0.75"),
      "RMID=", os.environ.get("ADCSR_RMID", os.environ.get("ADCSR_R3", "0.75")))

device = torch.device("cuda")

from diffusers import StableDiffusionPipeline
model_id = "Manojb/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

unet = pipe.unet

from diffusers.models.autoencoders.vae import Decoder
ckpt_halfdecoder = torch.load("./weight/pretrained/halfDecoder.ckpt", weights_only=False)

decoder = Decoder(
    in_channels=4,
    out_channels=3,
    up_block_types=["UpDecoderBlock2D" for _ in range(4)],
    block_out_channels=[64, 128, 256, 256],
    layers_per_block=2,
    norm_num_groups=32,
    act_fn="silu",
    norm_type="group",
    mid_block_add_attention=True,
).to(device)

decoder_ckpt = {}
for k, v in ckpt_halfdecoder["state_dict"].items():
    if "decoder" in k:
        decoder_ckpt[k.replace("decoder.", "")] = v
decoder.load_state_dict(decoder_ckpt, strict=True)

# Build student with current env ratios (must match the ckpt you load)
student = Net(unet, copy.deepcopy(decoder)).to(device)

ckpt_path = os.path.join(args.model_dir, f"net_params_{args.epoch}.pkl")
print("Loading ckpt:", ckpt_path)
loaded, skipped = load_ckpt_flexible_sliced(student, ckpt_path)
print(f"Loaded tensors: {loaded}, skipped: {skipped}")

model = torch.nn.Sequential(
    student,
    *decoder.up_blocks,
    decoder.conv_norm_out,
    decoder.conv_act,
    decoder.conv_out,
).to(device)

model.eval()

test_LR_paths = list(sorted(glob.glob(os.path.join(args.LR_dir, "*.png"))))
os.makedirs(args.SR_dir, exist_ok=True)

total_t = 0.0
n = 0

with torch.no_grad():
    for path in test_LR_paths:
        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        
        # --- ROBUST PADDING START ---
        # 1. Pad so dimensions are divisible by 64 (Safe for U-Net & PixelShuffle)
        LR_pad, orig_hw = pad_to_multiple(LR, mult=64, mode="reflect")
        # ----------------------------

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # 2. Run model on PADDED image
        SR_pad = model(LR_pad)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        # --- CROP BACK START ---
        # 3. Crop back to original size * scale (x4)
        SR = crop_back(SR_pad, orig_hw, scale=4)
        # -----------------------

        total_t += (t1 - t0)
        n += 1

        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(dim=[2, 3], keepdim=True) \
             * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)

        SR_img = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).cpu())
        SR_img.save(os.path.join(args.SR_dir, os.path.basename(path)))

avg_t = total_t / max(1, n)
print(f"Inference time: avg {avg_t:.6f} s/img over {n} images")

if args.time_file is not None:
    os.makedirs(os.path.dirname(args.time_file), exist_ok=True)
    with open(args.time_file, "w") as f:
        f.write(f"{avg_t}\n")
