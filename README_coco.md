# MedSegDiff → Joint Colorization + Segmentation (COCO-Stuff)

This repository adapts the MedSegDiff architecture (conditional DDPM + "highway" condition encoder)
to: **grayscale image (L channel) → colors (ab channels, Lab space) + semantic segmentation map (182 COCO-Stuff classes)**.
Supports COCO-Stuff 164k (2017, PNG labels) and COCO-Stuff 10k (.mat); the format is auto-detected.

## Final design (default)

| Component | Original MedSegDiff | This version (default) |
|---|---|---|
| Condition | Medical image | L channel (1 ch, [-1,1]) |
| Diffusion | 1-channel binary mask, ε-prediction, linear schedule | **ab color (2 ch), x0-prediction, cosine schedule** |
| Highway (condition encoder) | nnUNet (32 features), anchors detached | **ResNet-50 pretrained on ImageNet** + FPN decoder, same `(anchors, cal)` interface |
| `cal` head | 1 channel sigmoid | 183 segmentation logits (CE) + 2 ab (L1) |
| Main-UNet segmentation | – | **183-logit head on the main UNet's final features** (CE at every noise level) |
| Segmentation output | Dice-based fusion | mean main-UNet probabilities over the sampling steps (+ ensemble with highway) |

Loss = MSE(x0_ab) + λ_ce·CE(highway) + λ_ab·L1(highway ab) + λ_seg·CE(main UNet).

Other options (kept for experiments): `--target_ch 10` (analog-bit diffusion of the labels, the original design),
`--highway nnunet` (original encoder), `--self_cond True` (self-conditioning), `--seg_ch 0` (disable the main-UNet head).

## Why this design: overfitting experiments (1000 train images, 256×256, T4)

Measured on 32–64 training images with `scripts/coco_diag.py` (pixel acc / mIoU averaged per image, PSNR on RGB).
Baseline "always predict the most frequent class": pixel acc ≈ 0.10–0.11.

| Config | Steps (batch) | Seg acc | mIoU | Color PSNR | Note |
|---|---|---|---|---|---|
| ε-pred, linear, label bits, nnUNet (MedSegDiff as-is) | 6953 (6) | bits 0.006 / highway 0.39 | 0.19 | diff 12.7 dB | Diffusion learns nothing at high noise |
| **A** x0-pred + cosine, label bits | 12441 (3) | bits 0.05 / highway 0.32 | 0.13 | 22.6 dB | Colors fixed; label bits still very slow |
| B A + self-conditioning | 1939 (3) | worse than A at equal steps | | | stopped |
| C A + main-UNet seg head | 2395 (3) | main 0.086 | | | **label leak**: the head reads the label bits in x_t |
| D color diffusion + main seg head (nnUNet) | 4867 (3) | 0.20 | 0.07 | 22.3 dB | no leak, but slow |
| **F** D + **pretrained ResNet-50 highway** | 2209 (3) | 0.32 | 0.13 | 22.6 dB | reaches A's 12441-step level after only 2209 steps |
| **F** (end, 2h) | 8782 (3) | **0.66** (EMA, main) | **0.37** | **23.4 dB** | still improving |

Conclusions: (1) ε-prediction does not work for this task, use x0-prediction; (2) diffusing labels as analog
bits learns too slowly and leaks the label to any head that reads x_t; (3) segmentation from grayscale
needs a pretrained encoder — ResNet-50 makes it ~5–6× faster.

## Training on Kaggle (2× T4)

Use the notebook [`kaggle/train_kaggle.ipynb`](kaggle/train_kaggle.ipynb): fill in `WANDB_API_KEY`, `RESUME` → *Save Version (Save & Run All)*.
Equivalent command:

```bash
torchrun --standalone --nproc_per_node=2 scripts/coco_train.py --data_dir /kaggle/input/datasets/dntai2/cocostuf-2017 --out_dir /kaggle/working/run --batch_size 3 --grad_accum 2 --use_checkpoint False --max_hours 11.3 --resume auto
```

- Throughput: ~7.7 img/s on 2× T4 (fp16 + GradScaler; T4 has no bf16). 1 epoch of 118k images ≈ 4.3 h.
- **Checkpoints** (`out_dir/checkpoints/ckpt_*.pt` + `latest.pt`, ~2.4 GB): model + EMA + optimizer + scaler + step/epoch/position within the epoch + all args + metrics + wandb run id + total training time. Saved every 30 min and on stop (`--max_hours`, SIGTERM, Ctrl+C).
- **Continue in a new session**: upload `latest.pt` to Google Drive → `--resume <drive link>` (or a path / URL). Continues exactly (same epoch position, same wandb run). Changing the number of GPUs is allowed (restarts the current epoch).
- **Fine-tuning**: `--init_from <checkpoint>` loads only the weights (new optimizer, step 0).
- **wandb**: `--wandb True` with the environment variable `WANDB_API_KEY`; logs losses, lr, speed, VRAM and every `--vis_interval` steps an image grid `gray | GT | color raw | color EMA | GT seg | seg raw | seg EMA` plus pixel-acc/PSNR metrics.
- Throughput/VRAM knobs: `--use_checkpoint True --batch_size 4` if out of memory; `--hw_lr_mult` (highway decoder lr), `--hw_enc_lr_mult` (pretrained encoder lr), `--lambda_seg`, `--lambda_ce`, `--lambda_ab`.

## Sampling, evaluation, diagnostics

```bash
python scripts/coco_sample.py --model_path /kaggle/working/run/checkpoints/latest.pt --split val --num_samples 500 --out_dir ./results/samples --timestep_respacing ddim25
```
```bash
python scripts/coco_eval.py --pred_dir ./results/samples --split val
```
```bash
python scripts/coco_diag.py --model_path /kaggle/working/run/checkpoints/latest.pt --split val --n 64
```

- `coco_sample.py` reads the architecture from the checkpoint; `--use_ema True` by default; `--model_path` also accepts a Google Drive link.
- `coco_eval.py`: pixel acc, mean acc, mIoU (182 classes), PSNR, SSIM, colorfulness → `metrics.json`.
- `coco_diag.py`: scores each branch separately (highway / diffusion / main head / ensemble), useful for debugging.

## Changes to the original code (backward compatible: original MedSegDiff scripts keep their defaults)

- `guided_diffusion/unet.py`: `cond_channels`, `cal_channels`, `seg_channels`, `highway`; no Dropout in multi-class mode; fixes a stray conv being created in every forward.
- `guided_diffusion/gaussian_diffusion.py`: `target_channels` replaces `C=1`; `training_losses_joint` (self-cond, main seg head); fixes the `progress=True` and `ddim_sample` bugs.
- `guided_diffusion/pretrained_highway.py`, `coco_ckpt.py`, `coco_util.py`, `cocostuff_loader.py`: new.
- `guided_diffusion/nn.py`: gradient checkpointing via `torch.utils.checkpoint` (works with autocast).
