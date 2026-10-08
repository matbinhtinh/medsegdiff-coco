"""
Train MedSegDiff for joint colorization + semantic segmentation on COCO-Stuff (10k or 164k).

Single GPU:
    python scripts/coco_train.py --data_dir <coco-stuff> --out_dir ./results/coco
Multi-GPU (DDP, e.g. Kaggle 2x T4):
    torchrun --standalone --nproc_per_node=2 scripts/coco_train.py --data_dir <coco-stuff> --out_dir /kaggle/working/run

Resume / fine-tune:
    --resume auto                       continue from <out_dir>/checkpoints/latest.pt if present
    --resume <path | http(s) | Google Drive file/folder link>
    --init_from <same>                  load weights only (fresh optimizer/step) for fine-tuning

Checkpoints (guided_diffusion/coco_ckpt.py) hold model + EMA + optimizer + scaler + step/epoch
+ all args + metrics + wandb run id, so a new Kaggle session can continue seamlessly.
"""
import argparse
import copy
import math
import os
import signal
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from guided_diffusion import coco_ckpt, logger
from guided_diffusion.coco_util import (
    AB_CH,
    coco_model_and_diffusion_defaults,
    fuse_predictions,
    sample_joint,
)
from guided_diffusion.cocostuff_loader import NUM_CLASSES, build_dataset, lab_to_rgb, label_palette
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

MODEL_KEYS = list(model_and_diffusion_defaults().keys())
# model/diffusion args that do not change the weights' layout: keep the CLI value on resume
RUNTIME_KEYS = {"timestep_respacing", "use_checkpoint", "use_fp16", "dpm_solver"}


def create_argparser():
    defaults = dict(
        data_dir="/kaggle/input/datasets/dntai2/cocostuf-2017",
        out_dir="./results/coco",
        train_split="train",
        val_split="val",          # "test" for COCO-Stuff 10k
        max_items=0,              # >0: train on the first N images (overfit / debug)
        augment=True,
        num_workers=2,            # per process
        schedule_sampler="uniform",
        lr=1e-4,
        hw_lr_mult=5.0,           # lr multiplier for the highway (cal) head
        lr_schedule="constant",   # constant | cosine (cosine needs total_steps)
        warmup_steps=1000,
        total_steps=0,
        lr_min=1e-6,
        weight_decay=0.0,
        batch_size=4,             # per GPU
        grad_accum=1,             # effective batch = batch_size * grad_accum * world_size
        ema_rate="0.9999",
        lambda_ce=1.0,
        lambda_ab=1.0,
        lambda_seg=1.0,           # CE weight of the main-UNet segmentation head (needs --seg_ch 183)
        self_cond=False,          # self-conditioning on the previous x0 estimate (Analog Bits)
        self_cond_prob=0.5,
        grad_clip=0.0,
        precision="auto",         # auto | bf16 | fp16 | fp32  (auto: bf16 on Ampere+, fp16 on T4/V100)
        sync_bn=True,             # SyncBatchNorm for the highway head under DDP
        max_steps=0,              # 0 = unlimited
        max_hours=11.5,           # stop + save before the Kaggle session limit (0 = unlimited)
        log_interval=50,
        save_interval=5000,       # steps
        save_minutes=30.0,        # also save every N minutes (0 = off)
        keep_ckpts=2,             # full checkpoints kept besides latest.pt
        vis_interval=2000,        # sample + log images every N steps (0 = off)
        vis_n=4,
        vis_steps="ddim25",
        vis_from="val",           # val | train (train = check memorisation when overfitting)
        resume="auto",
        init_from="",
        seed=0,
        wandb=False,
        wandb_project="medsegdiff-coco",
        wandb_entity="",
        wandb_name="",
    )
    defaults.update(coco_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


class ResumableSampler(DistributedSampler):
    """DistributedSampler that can skip the samples already seen in the current epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = 0

    def __iter__(self):
        indices = list(super().__iter__())[self.start:]
        self.start = 0
        return iter(indices)

    def __len__(self):
        return self.num_samples - self.start


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank, world, local = 0, 1, 0
    if th.cuda.is_available():
        th.cuda.set_device(local)
        dev = th.device("cuda", local)
    else:
        dev = th.device("cpu")
    return rank, world, dev


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_obj(obj, rank):
    if not dist.is_initialized():
        return obj
    box = [obj if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def resolve_precision(name, dev):
    if dev.type != "cuda" or name == "fp32":
        return None
    if name == "auto":
        name = "bf16" if th.cuda.get_device_capability(dev)[0] >= 8 else "fp16"
    return {"bf16": th.bfloat16, "fp16": th.float16}[name]


def resolve_ckpt(spec, ckpt_dir, rank):
    """'auto' | local path | URL -> local path (rank 0 downloads) or None."""
    path = None
    if rank == 0 and spec:
        if spec == "auto":
            path = coco_ckpt.find_latest(ckpt_dir)
        elif coco_ckpt.is_url(spec):
            path = coco_ckpt.download(spec, os.path.join(ckpt_dir, "_download"))
        else:
            path = spec
    path = broadcast_obj(path, rank)
    barrier()
    return path


def lr_at(step, args):
    if args.warmup_steps and step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if args.lr_schedule == "cosine" and args.total_steps > args.warmup_steps:
        p = min(1.0, (step - args.warmup_steps) / (args.total_steps - args.warmup_steps))
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * p))
    return args.lr


@th.no_grad()
def visualize(model, vis_diffusion, vis_batch, ema_params, raw_params, dev, amp_dtype, pal, self_cond=False):
    """Sample the fixed visualization batch with raw and EMA weights -> (grid uint8, metrics)."""
    cond, target, label, _ = vis_batch
    cond = cond.to(dev)
    model.eval()
    rows, metrics = [], {}
    rgb_gt = lab_to_rgb(cond, target[:, :AB_CH])
    gray = ((cond.cpu().numpy()[:, 0] + 1) / 2)[..., None].repeat(3, -1)
    backup = [p.detach().clone() for p in raw_params]
    for tag, params in (("raw", None), ("ema", ema_params)):
        if params is not None:
            for p, e in zip(raw_params, params):
                p.data.copy_(e.data)
        g = th.Generator(device=dev).manual_seed(1234)
        noise = th.randn(cond.shape[0], vis_diffusion.target_channels, *cond.shape[2:], device=dev, generator=g)
        with th.autocast(device_type=dev.type, dtype=amp_dtype or th.float32, enabled=amp_dtype is not None):
            sample, cal, seg = sample_joint(vis_diffusion, model, cond, noise=noise, self_cond=self_cond)
        pred = fuse_predictions(sample.float(), cal.float(), seg=seg)
        rgb = lab_to_rgb(cond, pred["ab"])
        lab_pred = pred["label"].cpu().numpy()
        gt = label.numpy()
        valid = gt > 0
        metrics[f"vis/{tag}_pixel_acc"] = float((lab_pred[valid] == gt[valid]).mean()) if valid.any() else 0.0
        mse = ((rgb - rgb_gt) ** 2).mean(axis=(1, 2, 3))
        metrics[f"vis/{tag}_psnr"] = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
        for i in range(cond.shape[0]):
            if tag == "raw":
                rows.append([gray[i], rgb_gt[i], None, None, pal[gt[i]] / 255.0, None, None])
            rows[i][2 if tag == "raw" else 3] = rgb[i]
            rows[i][5 if tag == "raw" else 6] = pal[lab_pred[i]] / 255.0
    for p, b in zip(raw_params, backup):
        p.data.copy_(b)
    model.train()
    # columns: gray | GT color | color raw | color EMA | GT seg | seg raw | seg EMA
    grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    return (grid * 255).round().astype(np.uint8), metrics


def main():
    args = create_argparser().parse_args()
    rank, world, dev = setup_dist()
    is_main = rank == 0
    th.backends.cudnn.benchmark = True
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
        logger.configure(dir=args.out_dir, format_strs=["stdout", "log", "csv"])
    barrier()
    log = logger.log if is_main else (lambda *a, **k: None)

    # ---------------- checkpoint to resume / initialise from ----------------
    resume_path = resolve_ckpt(args.resume, ckpt_dir, rank)
    init_path = None if resume_path else resolve_ckpt(args.init_from, ckpt_dir, rank)
    ckpt = coco_ckpt.load(resume_path or init_path) if (resume_path or init_path) else None
    if ckpt is not None and ckpt.get("args"):
        # the architecture must match the checkpoint; training hyper-params come from the CLI
        for k in MODEL_KEYS:
            if k in ckpt["args"] and k not in RUNTIME_KEYS:
                setattr(args, k, ckpt["args"][k])
        args.self_cond = ckpt["args"].get("self_cond", False)  # changes the input layout
        log(f"{'resuming' if resume_path else 'initialising'} from {resume_path or init_path} "
            f"(step {ckpt.get('step', 0)}, epoch {ckpt.get('epoch', 0)})")

    # network input = [L, (x0 estimate if self_cond), x_t]
    args.in_ch = args.cond_ch + args.target_ch * (2 if args.self_cond else 1)

    seed = args.seed + rank + (ckpt.get("step", 0) if ckpt and resume_path else 0)
    th.manual_seed(seed)
    np.random.seed(seed % 2**32)
    amp_dtype = resolve_precision(args.precision, dev)
    log(f"world_size={world}  device={dev}  precision={amp_dtype or 'fp32'}")

    # ---------------- data ----------------
    ds = build_dataset(args.data_dir, args.train_split, args.image_size,
                       augment=args.augment, max_items=args.max_items or None)
    sampler = ResumableSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    steps_per_epoch = len(sampler) // args.batch_size // args.grad_accum
    log(f"{len(ds)} training images, {steps_per_epoch} optimizer steps per epoch, "
        f"effective batch {args.batch_size * args.grad_accum * world}")

    # ---------------- model ----------------
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, MODEL_KEYS))
    if ckpt is not None:
        model.load_state_dict(ckpt["model"] if resume_path else coco_ckpt.model_weights(ckpt))
    if world > 1 and args.sync_bn:
        model = th.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(dev)
    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    ddp_model = DDP(model, device_ids=[dev.index], gradient_as_bucket_view=True) if world > 1 else model
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=args.diffusion_steps)

    hw_ids = {id(p) for p in model.hwm.parameters()}
    opt = AdamW(
        [{"params": [p for p in model.parameters() if id(p) not in hw_ids], "lr_mult": 1.0},
         {"params": list(model.hwm.parameters()), "lr_mult": args.hw_lr_mult}],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = th.amp.GradScaler("cuda", enabled=amp_dtype == th.float16)
    raw_params = list(model.parameters())
    param_names = [n for n, _ in model.named_parameters()]
    ema_rates = [float(r) for r in args.ema_rate.split(",")]
    ema_params = [[p.detach().clone().float() for p in raw_params] for _ in ema_rates]

    step = epoch = samples_seen = epoch_batches = 0
    elapsed_prev = 0.0
    wandb_id = None
    if ckpt is not None and resume_path:
        for rate, params in zip(ema_rates, ema_params):
            sd = (ckpt.get("ema") or {}).get(str(rate))
            if sd is not None:
                for p, n in zip(params, param_names):
                    p.copy_(sd[n])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        step, epoch = ckpt.get("step", 0), ckpt.get("epoch", 0)
        samples_seen, epoch_batches = ckpt.get("samples_seen", 0), ckpt.get("epoch_batches", 0)
        elapsed_prev = ckpt.get("elapsed_hours", 0.0)
        wandb_id = ckpt.get("wandb_run_id")
        if ckpt.get("world_size", world) == world and ckpt.get("batch_size") == args.batch_size:
            sampler.start = epoch_batches * args.batch_size
        else:
            epoch_batches = 0  # layout changed: restart the epoch
    elif ckpt is not None:
        # fine-tuning: EMA starts from the loaded weights
        for params in ema_params:
            for p, r in zip(params, raw_params):
                p.copy_(r.detach())
    del ckpt

    # ---------------- logging / visualisation ----------------
    run = None
    if is_main and args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                             name=args.wandb_name or None, id=wandb_id, resume="allow",
                             config=vars(args), dir=args.out_dir)
            wandb_id = run.id
        except Exception as e:
            log(f"wandb disabled: {e}")
            run = None

    vis_batch = vis_diffusion = None
    if is_main and args.vis_interval > 0:
        vis_split = args.train_split if args.vis_from == "train" else args.val_split
        vis_ds = build_dataset(args.data_dir, vis_split, args.image_size, augment=False,
                               max_items=args.vis_n)
        vis_batch = next(iter(DataLoader(vis_ds, batch_size=args.vis_n, shuffle=False)))
        vis_diffusion = create_gaussian_diffusion(
            steps=args.diffusion_steps, noise_schedule=args.noise_schedule,
            timestep_respacing=args.vis_steps, target_channels=args.target_ch,
            predict_xstart=args.predict_xstart, rescale_timesteps=args.rescale_timesteps,
        )
        pal = label_palette()

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    t_start = time.time()
    last_save = time.time()
    metrics = {}

    saved_step = {"step": -1}

    def save_ckpt(tag=""):
        if saved_step["step"] == step:
            return
        saved_step["step"] = step
        if is_main:
            path = os.path.join(ckpt_dir, f"ckpt_{step:07d}.pt")
            coco_ckpt.save(
                path, keep_last=args.keep_ckpts,
                model={n: p.detach().cpu() for n, p in model.state_dict().items()},
                ema={str(r): {n: p.detach().cpu() for n, p in zip(param_names, params)} |
                     {k: v.detach().cpu() for k, v in model.state_dict().items() if k not in param_names}
                     for r, params in zip(ema_rates, ema_params)},
                optimizer=opt.state_dict(),
                scaler=scaler.state_dict() if scaler.is_enabled() else None,
                step=step, epoch=epoch, epoch_batches=epoch_batches, samples_seen=samples_seen,
                steps_per_epoch=steps_per_epoch, world_size=world, batch_size=args.batch_size,
                args=vars(args), metrics=dict(metrics), wandb_run_id=wandb_id,
                elapsed_hours=elapsed_prev + (time.time() - t_start) / 3600,
                dataset={"dir": args.data_dir, "split": args.train_split, "size": len(ds)},
            )
            log(f"saved checkpoint {path} {tag}")
        barrier()

    # ---------------- training loop ----------------
    log("training...")
    ddp_model.train()
    data_iter = None
    sampler.set_epoch(epoch)
    data_iter = iter(loader)
    t_log, n_log = time.time(), 0
    sums = {}
    while True:
        if args.max_steps and step >= args.max_steps:
            break
        # every rank must take the same decision, otherwise one rank waits forever in an all-reduce:
        # the time limit is judged on rank 0's clock, SIGTERM on any rank
        out_of_time = is_main and args.max_hours and (time.time() - t_start) / 3600 >= args.max_hours
        flags = th.tensor([1.0 if out_of_time else 0.0, 1.0 if stop["flag"] else 0.0], device=dev)
        if dist.is_initialized():
            dist.all_reduce(flags)
        if flags[0].item() > 0:
            log("max_hours reached")
            break
        if flags[1].item() > 0:
            log("SIGTERM received")
            break

        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            try:
                cond, target, label, _ = next(data_iter)
            except StopIteration:
                epoch += 1
                epoch_batches = 0
                sampler.set_epoch(epoch)
                data_iter = iter(loader)
                cond, target, label, _ = next(data_iter)
            epoch_batches += 1
            cond = cond.to(dev, non_blocking=True)
            target = target[:, :args.target_ch].to(dev, non_blocking=True)  # target_ch=2: color-only diffusion
            label = label.to(dev, non_blocking=True)
            t, weights = schedule_sampler.sample(cond.shape[0], dev)
            sync = micro == args.grad_accum - 1 or world == 1
            ctx = ddp_model.no_sync() if (world > 1 and not sync) else th.enable_grad()
            with ctx:
                with th.autocast(device_type=dev.type, dtype=amp_dtype or th.float32, enabled=amp_dtype is not None):
                    losses = diffusion.training_losses_joint(
                        ddp_model, cond, target, label, t, NUM_CLASSES,
                        lambda_ce=args.lambda_ce, lambda_ab=args.lambda_ab, ab_channels=AB_CH,
                        self_cond=args.self_cond, self_cond_prob=args.self_cond_prob, lambda_seg=args.lambda_seg,
                    )
                loss = (losses["loss"] * weights).mean() / args.grad_accum
                scaler.scale(loss).backward()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.detach().mean().item()
            n_log += 1

        scaler.unscale_(opt)
        grad_norm = th.nn.utils.clip_grad_norm_(raw_params, args.grad_clip if args.grad_clip > 0 else float("inf"))
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
        if not skipped:
            for rate, params in zip(ema_rates, ema_params):
                update_ema(params, raw_params, rate=rate)
        step += 1
        samples_seen += args.batch_size * args.grad_accum * world
        sums["grad_norm"] = sums.get("grad_norm", 0.0) + (grad_norm.item() if math.isfinite(grad_norm.item()) else 0.0) * args.grad_accum
        sums["skipped"] = sums.get("skipped", 0.0) + float(skipped) * args.grad_accum

        if step % args.log_interval == 0:
            keys = sorted(sums)
            vec = th.tensor([sums[k] / max(n_log, 1) for k in keys], device=dev)
            if dist.is_initialized():
                dist.all_reduce(vec)
                vec /= world
            dt = time.time() - t_log
            metrics = {f"train/{k}": float(v) for k, v in zip(keys, vec.tolist())}
            metrics.update({
                "train/lr": lr, "train/epoch": epoch + epoch_batches * args.batch_size / max(len(sampler), 1),
                "train/samples_seen": samples_seen,
                "perf/img_per_sec": n_log * args.batch_size * world / dt,
                "perf/vram_gb": th.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0,
                "perf/loss_scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
                "perf/hours": elapsed_prev + (time.time() - t_start) / 3600,
            })
            if is_main:
                for k, v in metrics.items():
                    logger.logkv(k, v)
                logger.logkv("step", step)
                logger.dumpkvs()
                if run is not None:
                    run.log(metrics, step=step)
            sums, n_log, t_log = {}, 0, time.time()

        if args.vis_interval > 0 and step % args.vis_interval == 0:
            if is_main:  # other ranks wait at the barrier
                grid, vis_metrics = visualize(model, vis_diffusion, vis_batch, ema_params[-1], raw_params,
                                              dev, amp_dtype, pal, self_cond=args.self_cond)
                Image.fromarray(grid).save(os.path.join(args.out_dir, "samples", f"step_{step:07d}.jpg"),
                                           quality=90)
                log(" ".join(f"{k}={v:.4f}" for k, v in vis_metrics.items()))
                if run is not None:
                    import wandb
                    run.log({**vis_metrics, "vis/samples": wandb.Image(
                        grid, caption="gray | GT | color raw | color EMA | GT seg | seg raw | seg EMA")},
                        step=step)
            barrier()

        due = step % args.save_interval == 0
        if args.save_minutes > 0:
            due_t = th.tensor([1.0 if time.time() - last_save >= args.save_minutes * 60 else 0.0], device=dev)
            if dist.is_initialized():
                dist.broadcast(due_t, 0)
            due = due or due_t.item() > 0
        if due:
            save_ckpt()
            last_save = time.time()

    save_ckpt("(final)")
    if run is not None:
        run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
