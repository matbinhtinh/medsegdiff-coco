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


def fuse_predictions(sample, cal, alpha_seg=0.5, w_ab=0.5):
    """
    Combine the diffusion sample with the highway calibration head, in the spirit of
    MedSegDiff's `cal_out` fusion.
    :param sample: [B, 10, H, W] final x_0 sample (ab + analog bits).
    :param cal: [B, 185, H, W] raw highway output.
    :return: dict with ab [B,2,H,W] and label maps [B,H,W] (diffusion / cal / fused).
    """
    seg_logits, cal_ab = split_cal(cal.float())
    diff_ab = sample[:, :AB_CH].float()
    diff_label = bits2int(sample[:, AB_CH:], max_value=NUM_CLASSES - 1)

    probs = F.softmax(seg_logits, dim=1)
    probs[:, 0] = 0  # never predict "unlabeled"
    onehot = F.one_hot(diff_label, NUM_CLASSES).permute(0, 3, 1, 2).float()
    onehot[:, 0] = 0
    fused_label = (probs + alpha_seg * onehot).argmax(dim=1)

    return {
        "ab_diff": diff_ab,
        "ab_cal": cal_ab.clamp(-1, 1),
        "ab": ((1 - w_ab) * diff_ab + w_ab * cal_ab).clamp(-1, 1),
        "label_diff": diff_label,
        "label_cal": probs.argmax(dim=1),
        "label": fused_label,
    }


@th.no_grad()
def sample_joint(diffusion, model, cond, eta=0.0, progress=False, noise=None):
    """
    DDIM (eta=0) / DDPM-like (eta=1) sampling over the (possibly respaced) diffusion.
    :param cond: [B, 1, H, W] L channel.
    :return: (x0 sample [B, 10, H, W], cal of the last step [B, 185, H, W])
    """
    B, _, H, W = cond.shape
    x = th.randn(B, diffusion.target_channels, H, W, device=cond.device) if noise is None else noise
    indices = list(range(diffusion.num_timesteps))[::-1]
    if progress:
        from tqdm.auto import tqdm
        indices = tqdm(indices)
    out = None
    for i in indices:
        t = th.full((B,), i, device=cond.device, dtype=th.long)
        out = diffusion.ddim_sample(model, th.cat((cond, x), dim=1), t, clip_denoised=True, eta=eta)
        x = out["sample"]
    return x, out["cal"]
