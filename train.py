import torch, os, glob, random, copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from time import time
from tqdm import tqdm
from omegaconf import OmegaConf
from dataset import RealESRGANDataset, RealESRGANDegrader
from model import Net
from ram.models.ram_lora import ram
from torchvision import transforms
from utils import add_lora_to_unet
def init_distributed_if_needed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
    return rank, world_size

def load_checkpoint_sliced(model, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    tgt = model.state_dict()
    loaded, skipped = 0, 0

    def candidates(k):
        cands = [k]
        if k.startswith("module."):
            cands.append(k[len("module."):])
        else:
            cands.append("module." + k)
        return cands

    for k in tgt.keys():
        src_key = None
        for kk in candidates(k):
            if kk in ckpt:
                src_key = kk
                break
        if src_key is None:
            skipped += 1
            continue

        src = ckpt[src_key]
        dst = tgt[k]

        if src.shape == dst.shape:
            tgt[k].copy_(src)
        else:
            slices = tuple(slice(0, s) for s in dst.shape)
            tgt[k].copy_(src[slices])

        loaded += 1

    model.load_state_dict(tgt, strict=False)
    return loaded, skipped

# main
# -----------------------------
rank, world_size = init_distributed_if_needed()

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=3)

parser.add_argument("--dir_hr", type=str, required=True, help="DIV2K100/HR")
parser.add_argument("--dir_lr", type=str, default=None, help="(Unused) LR folder; LR is synthesized by degrader")

parser.add_argument("--pretrained_student", type=str, default="./weight/net_params_200.pkl")

args = parser.parse_args()

# fixed seed
seed = rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

config = OmegaConf.load("config.yml")

config.dataroot_gt = args.dir_hr
if rank == 0:
    print("Override config.dataroot_gt (HR) ->", args.dir_hr)
    if args.dir_lr:
        print("Note: dir_lr is not used by this dataset. LR is synthesized by RealESRGANDegrader.")

epoch = args.epoch
learning_rate = args.learning_rate
bsz = args.batch_size

device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

if rank == 0:
    print("WORLD_SIZE =", world_size)
    print("batch size per gpu =", bsz)

from diffusers import StableDiffusionPipeline
model_id = "Manojb/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

vae = pipe.vae
tokenizer = pipe.tokenizer
unet = pipe.unet
text_encoder = pipe.text_encoder

unet_D = copy.deepcopy(unet)
new_conv_in = torch.nn.Conv2d(256, 320, 3, padding=1).to(device)
new_conv_in.weight.data = unet_D.conv_in.weight.data.repeat(1, 64, 1, 1) / 64
new_conv_in.bias.data = unet_D.conv_in.bias.data
unet_D.conv_in = new_conv_in
unet_D = add_lora_to_unet(unet_D)
unet_D.set_adapters(["default_encoder", "default_decoder", "default_others"])

vae_teacher = copy.deepcopy(vae)
unet_teacher = copy.deepcopy(unet)

osediff = torch.load("./weight/pretrained/osediff.pkl", weights_only=False)
vae_teacher.load_state_dict(osediff["vae"])
unet_teacher.load_state_dict(osediff["unet"])

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
    mid_block_add_attention=True
).to(device)

decoder_ckpt = {}
for k, v in ckpt_halfdecoder["state_dict"].items():
    if "decoder" in k:
        decoder_ckpt[k.replace("decoder.", "")] = v
decoder.load_state_dict(decoder_ckpt, strict=True)

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DAPE = ram(
    pretrained="./weight/pretrained/ram_swin_large_14m.pth",
    pretrained_condition="./weight/pretrained/DAPE.pth",
    image_size=384,
    vit="swin_l"
).eval().to(device)

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
vae_teacher.requires_grad_(False)
unet_teacher.requires_grad_(False)
decoder.requires_grad_(False)
DAPE.requires_grad_(False)

student = Net(unet, copy.deepcopy(decoder)).to(device)

# DDP only if world_size > 1
if world_size > 1:
    model = DDP(student, device_ids=[rank])
else:
    model = student

# Load pretrained student (params200) with slicing
if rank == 0:
    print("Loading pretrained student:", args.pretrained_student)
loaded, skipped = load_checkpoint_sliced(model, args.pretrained_student)
if rank == 0:
    print(f"Loaded {loaded} tensors (skipped {skipped}).")

# Discriminator
disc = unet_D.to(device)
if world_size > 1:
    model_D = DDP(disc, device_ids=[rank])
else:
    model_D = disc

model.requires_grad_(True)
model_D.requires_grad_(False)

params_to_opt = []
for n, p in model_D.named_parameters():
    if "lora" in n or "conv_in" in n:
        p.requires_grad = True
        params_to_opt.append(p)

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt / 1e6, "M")

dataset = RealESRGANDataset(config, bsz)
degrader = RealESRGANDegrader(config, device)
dataloader = DataLoader(dataset, batch_size=bsz, num_workers=8)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(params_to_opt, lr=1e-6)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,], gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

model_dir = args.model_dir
log_path = os.path.join(args.log_dir, "log.txt")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

print("start training...")
timesteps = torch.tensor([999], device=device).long().expand(bsz,)
alpha = pipe.scheduler.alphas_cumprod[999]

for epoch_i in range(1, epoch + 1):
    start_time = time()
    loss_avg = 0.0
    loss_distil_avg = 0.0
    loss_adv_avg = 0.0
    loss_D_avg = 0.0
    iter_num = 0

    if world_size > 1:
        dist.barrier()

    for batch in tqdm(dataloader):
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                LR, HR = degrader.degrade(batch)
                text_input = tokenizer(
                    DAPE.generate_tag(ram_transforms(LR))[0],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                encoder_hidden_states = text_encoder(text_input.input_ids, return_dict=False)[0]
                LR, HR = LR * 2 - 1, HR * 2 - 1
                LR_ = F.interpolate(LR, scale_factor=4, mode="bicubic")
                LR_latents = vae_teacher.encode(LR_).latent_dist.mean * vae_teacher.config.scaling_factor
                HR_latents = vae.encode(HR).latent_dist.mean
                pred_teacher = unet_teacher(
                    LR_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]
                z0_teacher = (LR_latents - ((1 - alpha) ** 0.5) * pred_teacher) / (alpha ** 0.5)
                z0_teacher = vae_teacher.post_quant_conv(z0_teacher / vae_teacher.config.scaling_factor)
                z0_teacher = decoder.conv_in(z0_teacher)
                z0_teacher = decoder.mid_block(z0_teacher)

                z0_gt = vae.post_quant_conv(HR_latents)
                z0_gt = decoder.conv_in(z0_gt)
                z0_gt = decoder.mid_block(z0_gt)

            z0_student = model(LR)
            loss_distil = (z0_student - z0_teacher).abs().mean()
            loss_adv = F.softplus(-model_D(
                z0_student,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]).mean()
            loss = loss_distil + loss_adv

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.cuda.amp.autocast(enabled=True):
            pred_real = model_D(
                z0_gt.detach(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            pred_fake = model_D(
                z0_student.detach(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            loss_D = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()

        optimizer_D.zero_grad(set_to_none=True)
        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        scaler.update()

        loss_avg += loss.item()
        loss_distil_avg += loss_distil.item()
        loss_adv_avg += loss_adv.item()
        loss_D_avg += loss_D.item()
        iter_num += 1

    scheduler.step()

    loss_avg /= iter_num
    loss_distil_avg /= iter_num
    loss_adv_avg /= iter_num
    loss_D_avg /= iter_num

    log_data = "[%d/%d] Average loss: %f, distil loss: %f, adv loss: %f, D loss: %f, time cost: %.2fs, cur lr is %f." % (
        epoch_i, epoch, loss_avg, loss_distil_avg, loss_adv_avg, loss_D_avg,
        time() - start_time, scheduler.get_last_lr()[0]
    )

    if rank == 0:
        print(log_data)
        with open(log_path, "a") as log_file:
            log_file.write(log_data + "\n")
        if epoch_i % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"net_params_{epoch_i}.pkl"))

if world_size > 1:
    dist.destroy_process_group()
