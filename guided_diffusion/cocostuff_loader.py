"""
COCO-Stuff 10k (v1.1) loader for joint colorization + semantic segmentation.

Layout expected under `root`:
    images/<name>.jpg
    annotations/<name>.mat      (key 'S': uint8 HxW, 0 = unlabeled, 1..182 = classes)
    imageLists/{train,test,all}.txt

Each sample is returned as
    cond   [1, H, W]   L channel (Lab) scaled to [-1, 1]
    target [2 + n_bits, H, W]  ab / AB_SCALE (≈[-1, 1]) followed by analog label bits in {-1, 1}
    label  [H, W]      long class map (0 = unlabeled)
    name   str
"""
import os
import random

import numpy as np
import scipy.io
import torch
from PIL import Image
from skimage import color
from torch.utils.data import Dataset

from .utils import int2bits

NUM_CLASSES = 183   # 0 = unlabeled + 182 COCO-Stuff ids
N_BITS = 8          # ceil(log2(183))
AB_SCALE = 110.0


def load_class_names(root):
    """Return the 182 class names (index i -> label id i + 1)."""
    lst = os.path.join(root, "imageLists", "train.txt")
    with open(lst) as f:
        first = f.readline().strip()
    mat = scipy.io.loadmat(os.path.join(root, "annotations", first + ".mat"))
    return [str(n[0][0]) if n[0].size else "" for n in mat["names"]]


def lab_to_rgb(L, ab):
    """L: [B,1,H,W] in [-1,1]; ab: [B,2,H,W] in [-1,1] -> float numpy RGB [B,H,W,3] in [0,1]."""
    L = (L.detach().float().cpu().numpy() + 1.0) * 50.0
    ab = ab.detach().float().cpu().numpy() * AB_SCALE
    lab = np.concatenate([L, ab], axis=1).transpose(0, 2, 3, 1)
    return np.stack([np.clip(color.lab2rgb(x), 0, 1) for x in lab])


def label_palette(num_classes=NUM_CLASSES, seed=0):
    rng = np.random.RandomState(seed)
    pal = rng.randint(0, 256, size=(num_classes, 3)).astype(np.uint8)
    pal[0] = 0
    return pal


class CocoStuffDataset(Dataset):
    def __init__(self, root, split="train", image_size=256, augment=None, max_items=None):
        self.root = root
        self.image_size = image_size
        self.augment = (split == "train") if augment is None else augment
        with open(os.path.join(root, "imageLists", f"{split}.txt")) as f:
            self.names = [l.strip() for l in f if l.strip()]
        if max_items:
            self.names = self.names[:max_items]

    def __len__(self):
        return len(self.names)

    def _resize_crop(self, img, lab):
        size = self.image_size
        w, h = img.size
        if self.augment:
            short = random.randint(size, int(size * 1.5))
        else:
            short = size
        scale = short / min(w, h)
        nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
        lab = lab.resize((nw, nh), Image.Resampling.NEAREST)
        if self.augment:
            x0, y0 = random.randint(0, nw - size), random.randint(0, nh - size)
        else:
            x0, y0 = (nw - size) // 2, (nh - size) // 2
        box = (x0, y0, x0 + size, y0 + size)
        img, lab = img.crop(box), lab.crop(box)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img, lab

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.root, "images", name + ".jpg")).convert("RGB")
        S = scipy.io.loadmat(os.path.join(self.root, "annotations", name + ".mat"))["S"]
        lab = Image.fromarray(S.astype(np.uint8))
        img, lab = self._resize_crop(img, lab)

        rgb = np.asarray(img, dtype=np.float32) / 255.0
        lab_img = color.rgb2lab(rgb).astype(np.float32)          # H,W,3
        L = lab_img[..., 0] / 50.0 - 1.0
        ab = np.clip(lab_img[..., 1:] / AB_SCALE, -1.0, 1.0)

        label = torch.from_numpy(np.asarray(lab, dtype=np.int64).copy())
        label[label >= NUM_CLASSES] = 0
        cond = torch.from_numpy(L)[None]
        ab = torch.from_numpy(ab.transpose(2, 0, 1).copy())
        target = torch.cat([ab, int2bits(label, N_BITS)], dim=0)
        return cond, target, label, name
