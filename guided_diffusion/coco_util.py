"""
Shared helpers for the COCO-Stuff joint colorization + segmentation variant of MedSegDiff.

Channel layout of the network input x (11 channels):
    x[:, 0:1]   clean condition: L (Lab) in [-1, 1]
    x[:, 1:3]   noisy ab
    x[:, 3:11]  noisy analog label bits
Main (diffusion) output: eps for the 10 target channels.
Highway / calibration output (cal, 185 channels): 183 segmentation logits + 2 deterministic ab.
"""
import torch as th
import torch.nn.functional as F

from .cocostuff_loader import NUM_CLASSES, N_BITS
from .script_util import model_and_diffusion_defaults
from .utils import bits2int

AB_CH = 2
TARGET_CH = AB_CH + N_BITS            # 10
COND_CH = 1
CAL_CH = NUM_CLASSES + AB_CH          # 185


def coco_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res.update(
        image_size=256,
        num_channels=128,             # must stay 128: highway anchors add 32+32+64 channels
        num_res_blocks=2,
        num_heads=4,
        attention_resolutions="16,8",
        use_checkpoint=True,
        use_scale_shift_norm=True,
        in_ch=COND_CH + TARGET_CH,
        target_ch=TARGET_CH,
        cal_ch=CAL_CH,
        cond_ch=COND_CH,
        version="new",
        # eps-prediction gives almost no learning signal at high noise for analog bits
        # (eps ~= x_t there); predicting x0 forces the model to infer labels/colors from L
        predict_xstart=True,
        noise_schedule="cosine",
    )
    return res


def split_cal(cal):
    """cal [B, 185, H, W] -> (seg_logits [B, 183, H, W], ab [B, 2, H, W])."""
    return cal[:, :NUM_CLASSES], cal[:, NUM_CLASSES:NUM_CLASSES + AB_CH]


def bits_class_loglik(bits, k=4.0):
    """
    Soft decoding of analog bits: log-likelihood of every class code.
    :param bits: [B, N_BITS, H, W] predicted analog bits in [-1, 1].
    :return: [B, NUM_CLASSES, H, W]; each bit is treated as Bernoulli(sigmoid(k * bit)).
    """
    codes = ((th.arange(NUM_CLASSES, device=bits.device)[:, None]
              >> th.arange(N_BITS - 1, -1, -1, device=bits.device)[None]) & 1).float()   # [C, N_BITS]
    logp1 = F.logsigmoid(k * bits.float())
    logp0 = F.logsigmoid(-k * bits.float())
    return th.einsum("cb,bnhw->nchw", codes, logp1.transpose(0, 1)) + \
        th.einsum("cb,bnhw->nchw", 1 - codes, logp0.transpose(0, 1))


def fuse_predictions(sample, cal, alpha_seg=0.5, w_ab=0.5, beta_bits=1.0, seg=None):
    """
    Combine the diffusion sample with the highway calibration head.
    Labels: label_cal (highway argmax), label_diff (hard bit decoding), label_soft
    (argmax of log p_highway(c) + beta * log p_bits(c)), label (= label_soft).
    Colors: ab_diff, ab_cal and their mix ab.
    :param sample: [B, 10, H, W] final x_0 sample (ab + analog bits).
    :param cal: [B, 185, H, W] raw highway output.
    :param seg: optional dict from sample_joint with main-UNet segmentation probabilities
        ("first": at t=T, i.e. from L only; "mean": averaged over all sampling steps).
        When given, label = argmax of the averaged main-UNet probabilities.
    """
    seg_logits, cal_ab = split_cal(cal.float())
    diff_ab = sample[:, :AB_CH].float()
    logp = F.log_softmax(seg_logits, dim=1)
    logp[:, 0] = -1e4  # never predict "unlabeled"
    if sample.shape[1] >= AB_CH + N_BITS:
        bits = sample[:, AB_CH:AB_CH + N_BITS].float()
        diff_label = bits2int(bits, max_value=NUM_CLASSES - 1)
        soft_label = (logp + beta_bits * bits_class_loglik(bits)).argmax(dim=1)
    else:  # color-only diffusion (target_ch = 2): no label bits
        diff_label = soft_label = logp.argmax(dim=1)

    out = {
        "ab_diff": diff_ab,
        "ab_cal": cal_ab.clamp(-1, 1),
        "ab": ((1 - w_ab) * diff_ab + w_ab * cal_ab).clamp(-1, 1),
        "label_diff": diff_label,
        "label_cal": logp.argmax(dim=1),
        "label_soft": soft_label,
        "label": soft_label,
    }
    if seg is not None:
        for key in ("first", "mean"):
            probs = seg[key].float().clone()
            probs[:, 0] = 0
            out[f"label_main_{key}"] = probs.argmax(dim=1)
        # ensemble: main-UNet probabilities (averaged over sampling steps) + highway probabilities
        ens = seg["mean"].float() + logp.exp()
        ens[:, 0] = 0
        out["label_ens"] = ens.argmax(dim=1)
        out["label"] = out["label_ens"]
    return out


@th.no_grad()
def sample_joint(diffusion, model, cond, eta=0.0, progress=False, noise=None, self_cond=False):
    """
    DDIM (eta=0) / DDPM-like (eta=1) sampling over the (possibly respaced) diffusion.
    :param cond: [B, 1, H, W] L channel.
    :param self_cond: feed the previous step's x0 prediction ([cond, x0_prev, x_t] input).
    :return: (x0 sample [B, 10, H, W], cal of the last step [B, 185, H, W],
              seg: None or {"first", "mean"} main-UNet segmentation probabilities [B, 183, H, W])
    """
    B, _, H, W = cond.shape
    x = th.randn(B, diffusion.target_channels, H, W, device=cond.device) if noise is None else noise
    x0_prev = th.zeros_like(x)
    indices = list(range(diffusion.num_timesteps))[::-1]
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    out = None
    seg_first = seg_sum = None
    for i in indices:
        t = th.full((B,), i, device=cond.device, dtype=th.long)
        inp = th.cat((cond, x0_prev, x), dim=1) if self_cond else th.cat((cond, x), dim=1)
        out = diffusion.ddim_sample(model, inp, t, clip_denoised=True, eta=eta)
        x = out["sample"]
        x0_prev = out["pred_xstart"].to(x.dtype)
        if out.get("extra") is not None:
            probs = F.softmax(out["extra"][:, :NUM_CLASSES].float(), dim=1)
            seg_first = probs if seg_first is None else seg_first
            seg_sum = probs if seg_sum is None else seg_sum + probs
    seg = None if seg_sum is None else {"first": seg_first, "mean": seg_sum / len(indices)}
    return x, out["cal"], seg
