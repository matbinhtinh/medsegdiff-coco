"""
Self-contained training checkpoints for the COCO colorization + segmentation model.

A checkpoint is one .pt file:
    {
      "format": "medsegdiff-coco-v1",
      "model": state_dict,                 # raw weights
      "ema": {"0.9999": state_dict, ...},  # EMA weights per rate
      "optimizer", "scaler": state dicts   # absent in weights-only exports
      "step", "epoch", "samples_seen", "epoch_step": progress counters
      "args": dict of all training args,   # incl. model/diffusion config
      "metrics": last logged training metrics,
      "wandb_run_id": str | None,
      "created": ISO time, "elapsed_hours": float (total training time so far),
      "world_size": int, "torch": version,
    }
"""
import datetime
import glob
import os
import re
import shutil

import torch as th

FORMAT = "medsegdiff-coco-v1"


def is_url(path):
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def download(url, out_dir):
    """Download a checkpoint from Google Drive (file or folder link) or plain HTTP(S)."""
    os.makedirs(out_dir, exist_ok=True)
    if "drive.google.com" in url or "docs.google.com" in url:
        import gdown
        if "/folders/" in url:
            files = gdown.download_folder(url, output=out_dir, quiet=False)
            pts = sorted(f for f in (files or []) if f.endswith(".pt"))
            if not pts:
                raise FileNotFoundError(f"no .pt file in Google Drive folder {url}")
            return find_latest(os.path.dirname(pts[0])) or pts[-1]
        return gdown.download(url, output=os.path.join(out_dir, "resume.pt"), quiet=False, fuzzy=True)
    dst = os.path.join(out_dir, "resume.pt")
    th.hub.download_url_to_file(url, dst)
    return dst


def find_latest(ckpt_dir):
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    return files[-1] if files else None


def load(path, map_location="cpu"):
    ckpt = th.load(path, map_location=map_location, weights_only=False)
    if not (isinstance(ckpt, dict) and ckpt.get("format") == FORMAT):
        # plain state_dict (old savedmodel*.pt / emasavedmodel*.pt)
        ckpt = {"format": "state_dict", "model": ckpt, "ema": {}, "step": 0, "epoch": 0}
    return ckpt


def model_weights(ckpt, use_ema=True, rate=None):
    """Pick EMA weights (default, highest rate) or raw weights from a loaded checkpoint."""
    ema = ckpt.get("ema") or {}
    if use_ema and ema:
        key = str(rate) if rate is not None else max(ema, key=float)
        return ema[key]
    return ckpt["model"]


def save(path, keep_last=2, **payload):
    """Atomically write a checkpoint, refresh latest.pt and prune old ckpt_*.pt files."""
    payload.setdefault("format", FORMAT)
    payload.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("torch", th.__version__)
    tmp = path + ".tmp"
    th.save(payload, tmp)
    os.replace(tmp, path)
    ckpt_dir = os.path.dirname(path)
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.abspath(path) != os.path.abspath(latest):
        try:
            if os.path.lexists(latest):
                os.remove(latest)
            os.link(path, latest)          # hard link: no extra disk space
        except OSError:
            shutil.copyfile(path, latest)
        if keep_last > 0:
            files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")),
                           key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[0]))
            for f in files[:-keep_last]:
                os.remove(f)
    return path


def export_weights(src, dst, use_ema=True):
    """Small inference/fine-tuning file: weights + metadata, no optimizer state."""
    ckpt = load(src)
    out = {k: v for k, v in ckpt.items() if k not in ("optimizer", "scaler", "model", "ema")}
    out["model"] = model_weights(ckpt, use_ema=use_ema)
    out["ema"] = {}
    out["exported_from"] = os.path.basename(src)
    th.save(out, dst)
    return dst
