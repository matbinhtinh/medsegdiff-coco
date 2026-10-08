# MedSegDiff → Joint Colorization + Segmentation (COCO-Stuff 10k)

This variant reuses the MedSegDiff architecture (conditional DDPM + highway/condition encoder) for:
**grayscale image (L channel) → color (ab) + semantic segmentation map (182 classes)**.

## Design

| Component | Original MedSegDiff | COCO variant |
|---|---|---|
| Condition | MRI/ISIC image (3–4 channels) | L channel (Lab), 1 channel |
| Diffused target | 1-channel binary mask | 10 channels: `ab/110` (2) + 8 *analog bits* of the label (0..182) |
| Main output | eps (out_channels=2) | eps for 10 channels |
| Highway head (`cal`) | 1 channel sigmoid, MSE (not added to the loss) | 185 channels: 183 seg logits (CE, ignore 0) + 2 deterministic ab (L1) |
| Loss | MSE + 10·MSE(cal) | MSE(eps) + λ_ce·CE + λ_ab·L1 |
| Sampling | DDPM 1000 steps / DPM-Solver, Dice-based fusion | DDIM (default 50 steps), fusion `softmax(cal) + α·onehot(bits)`, `ab = mix(diffusion, cal)` |

Changes to the original code (backward compatible, defaults unchanged):
- `guided_diffusion/unet.py`: `UNetModel_newpreview` gains `cond_channels`, `cal_channels`, `cal_nonlin`; optional `batchgenerators` import.
- `guided_diffusion/gaussian_diffusion.py`: `target_channels` replaces the hard-coded `C=1`; new `training_losses_joint`; fixes the `progress=True` bug and `ddim_sample` slicing.
- `guided_diffusion/script_util.py`: new args `target_ch`, `cal_ch`, `cond_ch`.
- `guided_diffusion/nn.py`: gradient checkpointing uses `torch.utils.checkpoint` (works with bf16 autocast).
- New files: `guided_diffusion/cocostuff_loader.py`, `guided_diffusion/coco_util.py`, `scripts/coco_train.py`, `scripts/coco_sample.py`, `scripts/coco_eval.py`.

## Installation

```bash
pip install scikit-image blobfile scipy tqdm
```

## Training (256×256, ~4.4 GB VRAM with batch 2)

```bash
python scripts/coco_train.py --data_dir C:/Users/Admin/Downloads/cocostuff-10k-v1.1 --out_dir ./results/coco --batch_size 2 --grad_accum 4 --save_interval 5000
```

- Effective batch = `batch_size × grad_accum`. 8 GB VRAM still has room — try `--batch_size 4 --grad_accum 2`.
- Resume: `--resume_checkpoint ./results/coco/savedmodel050000.pt` (automatically loads the EMA and optimizer state with the same step).
- Ctrl+C saves a checkpoint before exiting. Logs go to `out_dir/log.txt` and `progress.csv`.
- Important flags: `--lambda_ce`, `--lambda_ab`, `--lr`, `--hw_lr_mult` (highway head lr = lr × 5 by default; that head learns CE very slowly at a small lr), `--ema_rate`, `--grad_clip` (default off), `--max_items` (train on a subset), `--max_steps`.
- `--num_channels` must stay 128 (the highway anchors are added with a fixed 32+32+64 channels).

## Sampling and evaluation

```bash
python scripts/coco_sample.py --model_path ./results/coco/emasavedmodel_0.9999_100000.pt --out_dir ./results/coco_samples --timestep_respacing ddim50
```

```bash
python scripts/coco_eval.py --pred_dir ./results/coco_samples
```

- `vis/*.jpg`: grayscale | GT color | predicted color | GT seg | seg (diffusion) | seg (fused).
- `--num_ensemble 5` averages several samples (as in MedSegDiff), `--eta 1` for stochastic sampling, `--alpha_seg`/`--w_ab` adjust the fusion weights.
- `coco_eval.py` writes `metrics.json`: pixel acc, mean acc, mIoU, PSNR, SSIM, colorfulness.
