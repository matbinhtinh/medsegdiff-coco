"""
COCO-Stuff loaders for joint colorization + semantic segmentation.

Two on-disk formats are supported (auto-detected by `build_dataset`):

* COCO-Stuff 10k (v1.1):
    images/<name>.jpg
    annotations/<name>.mat      (key 'S': uint8 HxW, 0 = unlabeled, 1..182 = classes)
    imageLists/{train,test,all}.txt
* COCO-Stuff 164k (2017):
    train2017/[train2017/]<id>.jpg, val2017/[val2017/]<id>.jpg
    stuffthingmaps_trainval2017/{train2017,val2017}/<id>.png   (0..181 = classes, 255 = unlabeled)
    cocostuff-labels.txt         ("0: unlabeled", "1: person", ...)

Both are mapped to the same label space: 0 = unlabeled, 1..182 = COCO-Stuff ids.
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


def detect_format(root):
    if os.path.isdir(os.path.join(root, "imageLists")):
        return "10k"
    if os.path.isdir(os.path.join(root, "stuffthingmaps_trainval2017")):
        return "164k"
    raise ValueError(f"unknown COCO-Stuff layout in {root}")


def load_class_names(root):
    """Return the 182 class names (index i -> label id i + 1)."""
    if detect_format(root) == "164k":
        names = {}
        with open(os.path.join(root, "cocostuff-labels.txt")) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    names[int(k)] = v.strip()
        return [names.get(i, str(i)) for i in range(1, NUM_CLASSES)]
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


class _CocoStuffBase(Dataset):
    """Shared crop / augmentation / Lab conversion. Subclasses implement _load(name)."""

    def __init__(self, image_size=256, augment=False):
        self.image_size = image_size
        self.augment = augment
        self.names = []

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        """-> (PIL RGB image, PIL 'L' label map with ids 0..182)"""
        raise NotImplementedError

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
        img, lab = self._load(name)
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


class CocoStuffDataset(_CocoStuffBase):
    """COCO-Stuff 10k v1.1 (splits: train / test / all)."""

    def __init__(self, root, split="train", image_size=256, augment=None, max_items=None):
        super().__init__(image_size, (split == "train") if augment is None else augment)
        self.root = root
        with open(os.path.join(root, "imageLists", f"{split}.txt")) as f:
            self.names = [l.strip() for l in f if l.strip()]
        if max_items:
            self.names = self.names[:max_items]

    def _load(self, name):
        img = Image.open(os.path.join(self.root, "images", name + ".jpg")).convert("RGB")
        S = scipy.io.loadmat(os.path.join(self.root, "annotations", name + ".mat"))["S"]
        return img, Image.fromarray(S.astype(np.uint8))


class CocoStuff164kDataset(_CocoStuffBase):
    """COCO-Stuff 164k / 2017 (splits: train / val)."""

    def __init__(self, root, split="train", image_size=256, augment=None, max_items=None):
        super().__init__(image_size, (split == "train") if augment is None else augment)
        split = {"test": "val"}.get(split, split)
        self.root = root
        img_dir = os.path.join(root, f"{split}2017")
        nested = os.path.join(img_dir, f"{split}2017")
        self.img_dir = nested if os.path.isdir(nested) else img_dir
        self.lab_dir = os.path.join(root, "stuffthingmaps_trainval2017", f"{split}2017")
        self.names = sorted(f[:-4] for f in os.listdir(self.lab_dir) if f.endswith(".png"))
        if max_items:
            self.names = self.names[:max_items]
        # PNG value v (0..181) -> id v + 1; 255 (unlabeled) -> 0
        lut = np.zeros(256, dtype=np.uint8)
        lut[:NUM_CLASSES - 1] = np.arange(1, NUM_CLASSES, dtype=np.uint8)
        self.lut = lut

    def _load(self, name):
        img = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        lab = np.asarray(Image.open(os.path.join(self.lab_dir, name + ".png")))
        return img, Image.fromarray(self.lut[lab])


def build_dataset(root, split="train", image_size=256, augment=None, max_items=None):
    cls = CocoStuff164kDataset if detect_format(root) == "164k" else CocoStuffDataset
    return cls(root, split, image_size, augment=augment, max_items=max_items)
