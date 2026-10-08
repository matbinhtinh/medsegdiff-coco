
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s


# ---- analog bits (Chen et al., "Analog Bits", 2022) for diffusing multi-class label maps ----

def int2bits(x, n_bits=8):
    """[..., H, W] integer tensor -> [..., n_bits, H, W] float tensor in {-1, 1} (MSB first)."""
    x = x.long()
    shifts = torch.arange(n_bits - 1, -1, -1, device=x.device)
    bits = (x.unsqueeze(-3) >> shifts.view(-1, 1, 1)) & 1
    return bits.float() * 2.0 - 1.0


def bits2int(bits, max_value=None):
    """[..., n_bits, H, W] analog bits -> [..., H, W] long tensor (threshold at 0).
    Codes above max_value (not a valid class) are mapped to 0."""
    n_bits = bits.shape[-3]
    weights = (2 ** torch.arange(n_bits - 1, -1, -1, device=bits.device)).view(-1, 1, 1)
    x = ((bits > 0).long() * weights).sum(dim=-3)
    if max_value is not None:
        x = torch.where(x > max_value, torch.zeros_like(x), x)
    return x
