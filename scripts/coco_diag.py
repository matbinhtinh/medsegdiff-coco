"""
Diagnose what each branch of a checkpoint has learned, on a few images:
highway head alone (cal) vs. diffusion sample vs. fused output, for segmentation and color.

    python scripts/coco_diag.py --model_path <ckpt> --split train --max_items 1000 --n 64
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as th

from guided_diffusion import coco_ckpt
from guided_diffusion.coco_util import fuse_predictions, sample_joint, split_cal
from guided_diffusion.cocostuff_loader import NUM_CLASSES, build_dataset, lab_to_rgb
from guided_diffusion.script_util import create_gaussian_diffusion, create_model_and_diffusion, \
    model_and_diffusion_defaults


def seg_scores(pred, gt):
    valid = gt > 0
    acc = float((pred[valid] == gt[valid]).mean()) if valid.any() else 0.0
    ious = []
    for c in np.unique(gt[valid]):
        p, g = pred == c, gt == c
        ious.append((p & g & valid).sum() / max(((p | g) & valid).sum(), 1))
    return acc, float(np.mean(ious)) if ious else 0.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_dir", default="/kaggle/input/datasets/dntai2/cocostuf-2017")
    p.add_argument("--split", default="train")
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--steps", default="ddim25")
    args = p.parse_args()
    dev = th.device("cuda")
    amp = th.bfloat16 if th.cuda.get_device_capability()[0] >= 8 else th.float16

    ckpt = coco_ckpt.load(args.model_path)
    margs = dict(model_and_diffusion_defaults())
    margs.update({k: v for k, v in ckpt["args"].items() if k in margs})
    margs["timestep_respacing"] = ""
    print(f"checkpoint step {ckpt.get('step')} epoch {ckpt.get('epoch')}")

    ds = build_dataset(args.data_dir, args.split, margs["image_size"], augment=False,
                       max_items=args.max_items or None)
    idx = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)
    batches = [idx[i:i + args.batch_size] for i in range(0, len(idx), args.batch_size)]
    diffusion = create_gaussian_diffusion(steps=margs["diffusion_steps"], noise_schedule=margs["noise_schedule"],
                                          timestep_respacing=args.steps, target_channels=margs["target_ch"])

    # majority-class baseline over the evaluated images
    gts = np.stack([ds[i][2].numpy() for i in idx])
    counts = np.bincount(gts[gts > 0], minlength=NUM_CLASSES)
    print(f"baseline (always class {counts.argmax()}): pixel_acc={counts.max() / max(counts.sum(), 1):.4f}")

    for tag, weights in (("raw", ckpt["model"]), ("ema", coco_ckpt.model_weights(ckpt))):
        model, _ = create_model_and_diffusion(**margs)
        model.load_state_dict(weights)
        model.to(dev).eval()
        res = {k: [] for k in ("cal", "diff", "fused")}
        col = {k: [] for k in ("cal", "diff", "fused")}
        for b in batches:
            items = [ds[i] for i in b]
            cond = th.stack([it[0] for it in items]).to(dev)
            target = th.stack([it[1] for it in items])
            gt = th.stack([it[2] for it in items]).numpy()
            with th.no_grad(), th.autocast("cuda", dtype=amp):
                sample, cal = sample_joint(diffusion, model, cond)
            pred = fuse_predictions(sample.float(), cal.float())
            rgb_gt = lab_to_rgb(cond, target[:, :2])
            for k, lab_key, ab_key in (("cal", "label_cal", "ab_cal"), ("diff", "label_diff", "ab_diff"),
                                       ("fused", "label", "ab")):
                for pr, g in zip(pred[lab_key].cpu().numpy(), gt):
                    res[k].append(seg_scores(pr, g))
                col[k].append(psnr(lab_to_rgb(cond, pred[ab_key]), rgb_gt))
        print(f"[{tag}]")
        for k in res:
            acc, miou = np.mean(res[k], axis=0)
            print(f"  {k:5s}  seg pixel_acc={acc:.4f}  mIoU(per-image)={miou:.4f}   color PSNR={np.mean(col[k]):.2f}")


if __name__ == "__main__":
    main()
