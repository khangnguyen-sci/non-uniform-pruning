import json
import torch, os, glob, pyiqa, json
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

parser = ArgumentParser()
parser.add_argument("--HR_dir", type=str, default="testset/RealSR/HR")
parser.add_argument("--SR_dir", type=str, default="result/RealSR")

parser.add_argument("--out_json", type=str, default=None, help="Write metrics JSON to this path")
parser.add_argument("--out_txt", type=str, default=None, help="Write metrics text to this path")

args = parser.parse_args()

device = torch.device("cuda")

psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device=device)
ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device=device)
lpips = pyiqa.create_metric("lpips", device=device)
dists = pyiqa.create_metric("dists", device=device)
fid = pyiqa.create_metric("fid", device=device)
niqe = pyiqa.create_metric("niqe", device=device)
maniqa = pyiqa.create_metric("maniqa-pipal", device=device)
clipiqa = pyiqa.create_metric("clipiqa", device=device)
musiq = pyiqa.create_metric("musiq", device=device)

test_SR_paths = list(sorted(glob.glob(os.path.join(args.SR_dir, "*"))))
test_HR_paths = list(sorted(glob.glob(os.path.join(args.HR_dir, "*"))))

metrics = {
    "psnr": [],
    "ssim": [],
    "lpips": [],
    "dists": [],
    "niqe": [],
    "maniqa": [],
    "musiq": [],
    "clipiqa": [],
}

for _, (SR_path, HR_path) in tqdm(enumerate(zip(test_SR_paths, test_HR_paths))):
    SR = Image.open(SR_path).convert("RGB")
    SR = transforms.ToTensor()(SR).to(device).unsqueeze(0)
    HR = Image.open(HR_path).convert("RGB")
    HR = transforms.ToTensor()(HR).to(device).unsqueeze(0)

    metrics["psnr"].append(psnr(SR, HR).item())
    metrics["ssim"].append(ssim(SR, HR).item())
    metrics["lpips"].append(lpips(SR, HR).item())
    metrics["dists"].append(dists(SR, HR).item())
    metrics["niqe"].append(niqe(SR).item())
    metrics["maniqa"].append(maniqa(SR).item())
    metrics["clipiqa"].append(clipiqa(SR).item())
    metrics["musiq"].append(musiq(SR).item())

for k in metrics.keys():
    metrics[k] = float(np.mean(metrics[k]))

metrics["fid"] = float(fid(args.SR_dir, args.HR_dir))

for k, v in metrics.items():
    if k == "niqe":
        print(k, f"{v:.3g}")
    elif k == "fid":
        print(k, f"{v:.5g}")
    else:
        print(k, f"{v:.4g}")
payload = {
    "HR_dir": args.HR_dir,
    "SR_dir": args.SR_dir,
    "num_pairs": int(min(len(test_SR_paths), len(test_HR_paths))),
    "metrics": metrics,
}

if args.out_json is not None:
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

if args.out_txt is not None:
    out_dir = os.path.dirname(args.out_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
