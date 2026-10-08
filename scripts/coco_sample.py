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

from guided_diffusion import coco_ckpt, logger
from guided_diffusion.cocostuff_loader import build_dataset, lab_to_rgb, label_palette
from guided_diffusion.coco_util import coco_model_and_diffusion_defaults, fuse_predictions, sample_joint
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def create_argparser():
    defaults = dict(
        data_dir="/kaggle/input/datasets/dntai2/cocostuf-2017",
        split="val",              # "test" for COCO-Stuff 10k
        model_path="",            # checkpoint (.pt), local path or URL / Google Drive link
        use_ema=True,             # EMA weights when the checkpoint has them
        out_dir="./results/coco_samples",
        num_samples=0,            # 0 = whole split
        batch_size=4,
        num_ensemble=1,           # MedSegDiff-style ensembling of several stochastic samples
        eta=0.0,                  # 0 = DDIM (deterministic), 1 = ancestral-like
        alpha_seg=0.5,            # weight of the diffusion label vote in the fused segmentation
        w_ab=0.5,                 # weight of the highway ab in the fused colors
        precision="auto",        # auto | bf16 | fp16 | fp32
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

    path = args.model_path
    if coco_ckpt.is_url(path):
        path = coco_ckpt.download(path, os.path.join(args.out_dir, "_download"))
    ckpt = coco_ckpt.load(path)
    for k, v in (ckpt.get("args") or {}).items():  # architecture from the checkpoint
        if k in model_and_diffusion_defaults() and k != "timestep_respacing":
            setattr(args, k, v)
    self_cond = (ckpt.get("args") or {}).get("self_cond", False)
    logger.log(f"checkpoint {path}: step {ckpt.get('step')} epoch {ckpt.get('epoch')} self_cond={self_cond}")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(coco_ckpt.model_weights(ckpt, use_ema=args.use_ema))
    model.to(dev).eval()
    ckpt = None
    amp = None
    if dev.type == "cuda" and args.precision != "fp32":
        amp = {"bf16": th.bfloat16, "fp16": th.float16}.get(args.precision) or (
            th.bfloat16 if th.cuda.get_device_capability()[0] >= 8 else th.float16)

    ds = build_dataset(args.data_dir, args.split, args.image_size, augment=False,
                       max_items=args.num_samples or None)
    loader = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pal = label_palette()
    logger.log(f"sampling {len(ds)} images with {diffusion.num_timesteps} steps x {args.num_ensemble} ensemble")

    for cond, target, label, names in loader:
        cond = cond.to(dev)
        samples, cals, segs = [], [], []
        with th.autocast(device_type=dev.type, dtype=amp or th.float32, enabled=amp is not None):
            for _ in range(args.num_ensemble):
                s, c, sg = sample_joint(diffusion, model, cond, eta=args.eta, self_cond=self_cond)
                samples.append(s.float())
                cals.append(c.float())
                if sg is not None:
                    segs.append(sg)
        sample = th.stack(samples).mean(0)
        cal = th.stack(cals).mean(0)
        seg = None if not segs else {k: th.stack([g[k] for g in segs]).mean(0) for k in ("first", "mean")}
        pred = fuse_predictions(sample, cal, alpha_seg=args.alpha_seg, w_ab=args.w_ab, seg=seg)

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
