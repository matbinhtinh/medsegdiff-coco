"""
Train MedSegDiff for joint colorization + semantic segmentation on COCO-Stuff 10k.

    python scripts/coco_train.py --data_dir C:/Users/Admin/Downloads/cocostuff-10k-v1.1 --out_dir ./results/coco

Single-GPU loop (no DDP/NCCL, so it also runs on Windows), bf16 autocast, gradient
accumulation, EMA, and MedSegDiff-style checkpoint names (savedmodelNNNNNN.pt, ...).
"""
import argparse
import copy
import glob
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch as th
from torch.optim import AdamW

from guided_diffusion import logger
from guided_diffusion.cocostuff_loader import CocoStuffDataset, NUM_CLASSES
from guided_diffusion.coco_util import coco_model_and_diffusion_defaults, AB_CH
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def create_argparser():
    defaults = dict(
        data_dir="C:/Users/Admin/Downloads/cocostuff-10k-v1.1",
        out_dir="./results/coco",
        max_items=0,              # >0: train on a subset (debug / overfit test)
        num_workers=4,
        schedule_sampler="uniform",
        lr=1e-4,
        hw_lr_mult=5.0,           # lr multiplier for the highway (cal) head
        weight_decay=0.0,
        batch_size=2,
        grad_accum=4,             # effective batch = batch_size * grad_accum
        ema_rate="0.9999",
        lambda_ce=1.0,
        lambda_ab=1.0,
        grad_clip=0.0,            # 0 = no clipping (as in the original MedSegDiff)
        bf16=True,
        max_steps=0,              # 0 = run forever
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",     # path to savedmodelNNNNNN.pt
        seed=0,
    )
    defaults.update(coco_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    args = create_argparser().parse_args()
    th.manual_seed(args.seed)
    logger.configure(dir=args.out_dir, format_strs=["stdout", "log", "csv"])
    dev = th.device("cuda" if th.cuda.is_available() else "cpu")

    logger.log("creating data loader...")
    ds = CocoStuffDataset(args.data_dir, "train", args.image_size, max_items=args.max_items or None)
    loader = th.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    logger.log(f"{len(ds)} training images")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dev)
    logger.log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    # the highway (nnUNet-style) segmentation/color head learns much faster with a larger lr
    hw_params = list(model.hwm.parameters())
    hw_ids = {id(p) for p in hw_params}
    main_params = [p for p in model.parameters() if id(p) not in hw_ids]
    opt = AdamW(
        [{"params": main_params, "lr": args.lr},
         {"params": hw_params, "lr": args.lr * args.hw_lr_mult}],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    ema_rates = [float(r) for r in args.ema_rate.split(",")]
    ema_params = [copy.deepcopy(list(model.parameters())) for _ in ema_rates]
    for params in ema_params:
        for p in params:
            p.requires_grad_(False)

    step = 0
    if args.resume_checkpoint:
        step = int(re.findall(r"(\d+)\.pt$", args.resume_checkpoint)[-1])
        ckdir = os.path.dirname(args.resume_checkpoint)
        logger.log(f"resuming from {args.resume_checkpoint} (step {step})")
        model.load_state_dict(th.load(args.resume_checkpoint, map_location=dev))
        for rate, params in zip(ema_rates, ema_params):
            ema_path = os.path.join(ckdir, f"emasavedmodel_{rate}_{step:06d}.pt")
            if os.path.exists(ema_path):
                sd = th.load(ema_path, map_location=dev)
                for p, (_, v) in zip(params, sorted_params(model, sd)):
                    p.copy_(v)
        opt_path = os.path.join(ckdir, f"optsavedmodel{step:06d}.pt")
        if os.path.exists(opt_path):
            opt.load_state_dict(th.load(opt_path, map_location=dev))

    def save():
        th.save(model.state_dict(), os.path.join(args.out_dir, f"savedmodel{step:06d}.pt"))
        for rate, params in zip(ema_rates, ema_params):
            sd = {name: p.detach() for (name, _), p in zip(model.named_parameters(), params)}
            sd.update({k: v for k, v in model.state_dict().items() if k not in sd})  # BN buffers
            th.save(sd, os.path.join(args.out_dir, f"emasavedmodel_{rate}_{step:06d}.pt"))
        th.save(opt.state_dict(), os.path.join(args.out_dir, f"optsavedmodel{step:06d}.pt"))
        logger.log(f"saved checkpoint at step {step}")

    logger.log("training...")
    data = infinite(loader)
    model.train()
    try:
        while not args.max_steps or step < args.max_steps:
            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                cond, target, label, _ = next(data)
                cond = cond.to(dev, non_blocking=True)
                target = target.to(dev, non_blocking=True)
                label = label.to(dev, non_blocking=True)
                t, weights = schedule_sampler.sample(cond.shape[0], dev)
                with th.autocast(device_type=dev.type, dtype=th.bfloat16, enabled=args.bf16):
                    losses = diffusion.training_losses_joint(
                        model, cond, target, label, t, NUM_CLASSES,
                        lambda_ce=args.lambda_ce, lambda_ab=args.lambda_ab, ab_channels=AB_CH,
                    )
                loss = (losses["loss"] * weights).mean() / args.grad_accum
                loss.backward()
                for k, v in losses.items():
                    logger.logkv_mean(k, v.mean().item())
            grad_norm = th.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
            logger.logkv_mean("grad_norm", grad_norm.item())
            opt.step()
            for rate, params in zip(ema_rates, ema_params):
                update_ema(params, list(model.parameters()), rate=rate)
            step += 1

            if step % args.log_interval == 0:
                logger.logkv("step", step)
                logger.logkv("samples", step * args.batch_size * args.grad_accum)
                if th.cuda.is_available():
                    logger.logkv("vram_gb", th.cuda.max_memory_allocated() / 2**30)
                logger.dumpkvs()
            if step % args.save_interval == 0:
                save()
    except KeyboardInterrupt:
        logger.log("interrupted")
    save()


def sorted_params(model, sd):
    return [(n, sd[n]) for n, _ in model.named_parameters()]


if __name__ == "__main__":
    main()
