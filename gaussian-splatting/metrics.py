#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
    LPIPS_IMPORT_ERROR = ""
except Exception as exc:
    lpips = None
    LPIPS_AVAILABLE = False
    LPIPS_IMPORT_ERROR = str(exc)
    print(f"[WARNING] lpipsPyTorch unavailable; LPIPS will be written as null. Reason: {LPIPS_IMPORT_ERROR}")

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not render_path.is_file() or not gt_path.is_file():
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpips_failed = False

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(float(ssim(renders[idx], gts[idx]).detach().mean().item()))
                    psnrs.append(float(psnr(renders[idx], gts[idx]).detach().mean().item()))
                    if LPIPS_AVAILABLE and not lpips_failed:
                        try:
                            lpips_value = lpips(renders[idx], gts[idx], net_type='vgg')
                            if torch.isfinite(lpips_value).all():
                                lpipss.append(float(lpips_value.detach().mean().item()))
                            else:
                                print("[WARNING] LPIPS returned a non-finite value; LPIPS will be written as null for this scene.")
                                lpipss = []
                                lpips_failed = True
                        except Exception as exc:
                            print(f"[WARNING] LPIPS compute failed; LPIPS will be written as null for this scene. Reason: {exc}")
                            lpipss = []
                            lpips_failed = True

                ssim_mean = (sum(ssims) / float(len(ssims))) if len(ssims) > 0 else 0.0
                psnr_mean = (sum(psnrs) / float(len(psnrs))) if len(psnrs) > 0 else 0.0
                print("  SSIM : {:>12.7f}".format(ssim_mean, ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_mean, ".5"))
                lpips_mean = (sum(lpipss) / float(len(lpipss))) if len(lpipss) > 0 else None
                if lpips_mean is None:
                    print("  LPIPS: unavailable")
                else:
                    print("  LPIPS: {:>12.7f}".format(lpips_mean, ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": ssim_mean,
                                                        "PSNR": psnr_mean,
                                                        "LPIPS": lpips_mean,
                                                        "LPIPS_status": "ok" if lpips_mean is not None else "unavailable"})
                per_view_dict[scene_dir][method].update({"SSIM": {name: value for value, name in zip(ssims, image_names)},
                                                            "PSNR": {name: value for value, name in zip(psnrs, image_names)},
                                                            "LPIPS": (
                                                                {name: lp for lp, name in zip(lpipss, image_names)}
                                                                if lpips_mean is not None
                                                                else {name: None for name in image_names}
                                                            )})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:
            print(f"Unable to compute metrics for model {scene_dir}: {exc}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
