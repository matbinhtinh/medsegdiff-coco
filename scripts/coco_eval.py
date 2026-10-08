"""
Evaluate outputs of coco_sample.py against COCO-Stuff ground truth.

    python scripts/coco_eval.py --pred_dir ./results/coco_samples

Segmentation: pixel accuracy, mean accuracy, mIoU (182 classes, unlabeled pixels ignored).
Colorization: PSNR / SSIM vs. the ground-truth RGB (same center crop), and colorfulness
(Hasler & Suesstrunk) of predictions vs. ground truth.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from guided_diffusion.cocostuff_loader import NUM_CLASSES, build_dataset, lab_to_rgb, load_class_names


def colorfulness(rgb):
    r, g, b = (rgb * 255.0).transpose(2, 0, 1)
    rg, yb = r - g, 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/kaggle/input/datasets/dntai2/cocostuf-2017")
    p.add_argument("--pred_dir", default="./results/coco_samples")
    p.add_argument("--split", default="val")
    p.add_argument("--image_size", type=int, default=256)
    args = p.parse_args()

    ds = build_dataset(args.data_dir, args.split, args.image_size, augment=False)
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    psnr, ssim, cf_pred, cf_gt = [], [], [], []
    n = 0
    for idx, name in enumerate(ds.names):
        lab_path = os.path.join(args.pred_dir, "pred_label", name + ".png")
        rgb_path = os.path.join(args.pred_dir, "pred_rgb", name + ".png")
        if not (os.path.exists(lab_path) and os.path.exists(rgb_path)):
            continue
        cond, target, label, _ = ds[idx]
        gt = label.numpy()
        pr = np.asarray(Image.open(lab_path), dtype=np.int64)
        valid = gt > 0
        conf += np.bincount(gt[valid] * NUM_CLASSES + pr[valid],
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

        rgb_gt = lab_to_rgb(cond[None], target[None, :2])[0]
        rgb_pr = np.asarray(Image.open(rgb_path), dtype=np.float64) / 255.0
        psnr.append(peak_signal_noise_ratio(rgb_gt, rgb_pr, data_range=1.0))
        ssim.append(structural_similarity(rgb_gt, rgb_pr, channel_axis=2, data_range=1.0))
        cf_pred.append(colorfulness(rgb_pr))
        cf_gt.append(colorfulness(rgb_gt))
        n += 1

    if n == 0:
        sys.exit(f"no predictions found in {args.pred_dir}")

    conf = conf[1:, 1:]  # drop 'unlabeled'
    tp = np.diag(conf).astype(np.float64)
    gt_count, pred_count = conf.sum(1), conf.sum(0)
    present = gt_count > 0
    iou = tp / np.maximum(gt_count + pred_count - tp, 1)
    acc_c = tp / np.maximum(gt_count, 1)
    names = load_class_names(args.data_dir)

    res = {
        "num_images": n,
        "pixel_acc": float(tp.sum() / max(conf.sum(), 1)),
        "mean_acc": float(acc_c[present].mean()),
        "mIoU": float(iou[present].mean()),
        "PSNR": float(np.mean(psnr)),
        "SSIM": float(np.mean(ssim)),
        "colorfulness_pred": float(np.mean(cf_pred)),
        "colorfulness_gt": float(np.mean(cf_gt)),
        "per_class_IoU": {names[i]: float(iou[i]) for i in np.where(present)[0]},
    }
    for k, v in res.items():
        if k != "per_class_IoU":
            print(f"{k:>18}: {v:.4f}" if isinstance(v, float) else f"{k:>18}: {v}")
    with open(os.path.join(args.pred_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("saved", os.path.join(args.pred_dir, "metrics.json"))


if __name__ == "__main__":
    main()
