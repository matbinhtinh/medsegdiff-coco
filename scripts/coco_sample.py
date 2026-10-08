"""
Sample colorization + segmentation from grayscale COCO-Stuff images.

    python scripts/coco_sample.py --model_path ./results/coco/emasavedmodel_0.9999_100000.pt --out_dir ./results/coco_samples

Writes, per image:
    pred_rgb/<name>.png     colorized image (fused ab)
    pred_label/<name>.png   uint8 label map (fused, 0..182)
    vis/<name>.jpg          [gray | GT color | pred color | GT seg | pred seg (diffusion) | pred seg (fused)]
Run scripts/coco_eval.py on out_dir afterwards for metrics.
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as th
from PIL import Image

from guided_diffusion import logger
from guided_diffusion.cocostuff_loader import CocoStuffDataset, lab_to_rgb, label_palette
from guided_diffusion.coco_util import coco_model_and_diffusion_defaults, fuse_predictions, sample_joint
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def create_argparser():
    defaults = dict(
        data_dir="C:/Users/Admin/Downloads/cocostuff-10k-v1.1",
        split="test",
        model_path="",
        out_dir="./results/coco_samples",
        num_samples=0,            # 0 = whole split
        batch_size=4,
        num_ensemble=1,           # MedSegDiff-style ensembling of several stochastic samples
        eta=0.0,                  # 0 = DDIM (deterministic), 1 = ancestral-like
        alpha_seg=0.5,            # weight of the diffusion label vote in the fused segmentation
        w_ab=0.5,                 # weight of the highway ab in the fused colors
        bf16=True,
        seed=0,
    )
    defaults.update(coco_model_and_diffusion_defaults())
    defaults["timestep_respacing"] = "ddim50"
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    th.manual_seed(args.seed)
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")
    for sub in ("pred_rgb", "pred_label", "vis"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)
    logger.configure(dir=args.out_dir, format_strs=["stdout", "log"])

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(dev).eval()

    ds = CocoStuffDataset(args.data_dir, args.split, args.image_size, augment=False,
                          max_items=args.num_samples or None)
    loader = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pal = label_palette()
    logger.log(f"sampling {len(ds)} images with {diffusion.num_timesteps} steps x {args.num_ensemble} ensemble")

    for cond, target, label, names in loader:
        cond = cond.to(dev)
        samples, cals = [], []
        with th.autocast(device_type=dev.type, dtype=th.bfloat16, enabled=args.bf16):
            for _ in range(args.num_ensemble):
                s, c = sample_joint(diffusion, model, cond, eta=args.eta)
                samples.append(s.float())
                cals.append(c.float())
        sample = th.stack(samples).mean(0)
        cal = th.stack(cals).mean(0)
        pred = fuse_predictions(sample, cal, alpha_seg=args.alpha_seg, w_ab=args.w_ab)

        rgb_pred = lab_to_rgb(cond, pred["ab"])
        rgb_gt = lab_to_rgb(cond, target[:, :2])
        gray = ((cond.cpu().numpy()[:, 0] + 1) / 2)[..., None].repeat(3, -1)
        for i, name in enumerate(names):
            lab_pred = pred["label"][i].cpu().numpy().astype(np.uint8)
            Image.fromarray((rgb_pred[i] * 255).round().astype(np.uint8)).save(
                os.path.join(args.out_dir, "pred_rgb", name + ".png"))
            Image.fromarray(lab_pred).save(os.path.join(args.out_dir, "pred_label", name + ".png"))
            tiles = [gray[i], rgb_gt[i], rgb_pred[i],
                     pal[label[i].numpy()] / 255.0,
                     pal[pred["label_diff"][i].cpu().numpy()] / 255.0,
                     pal[lab_pred] / 255.0]
            vis = (np.concatenate(tiles, axis=1) * 255).round().astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(args.out_dir, "vis", name + ".jpg"), quality=90)
        logger.log(f"done: {', '.join(names)}")


if __name__ == "__main__":
    main()
