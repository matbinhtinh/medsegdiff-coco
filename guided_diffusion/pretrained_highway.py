"""
Pretrained condition encoder ("highway") for MedSegDiff.

Drop-in replacement for the nnUNet-style Generic_UNet highway: same interface
    anchors, cal = highway(cond)
    anchors = (feat32 [B,32,H,W], feat64 [B,64,H,W])   -> injected into the diffusion UNet
    cal     = [B, out_channels, H, W]                   -> calibration / segmentation (+ color) head
but the encoder is an ImageNet-pretrained ResNet (torchvision) on the grayscale L channel,
which makes semantic segmentation learn far faster than a randomly initialised encoder.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d


def _gn(ch):
    return nn.GroupNorm(min(32, ch // 4), ch)


class _ConvGN(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), _gn(cout), nn.ReLU(inplace=True))


class PretrainedHighway(nn.Module):
    def __init__(self, in_channels, out_channels, arch="resnet50", pretrained=True, fpn_ch=128):
        super().__init__()
        weights = {"resnet50": "IMAGENET1K_V2", "resnet34": "IMAGENET1K_V1", "resnet18": "IMAGENET1K_V1"}[arch]
        net = getattr(torchvision.models, arch)(weights=weights if pretrained else None,
                                               norm_layer=FrozenBatchNorm2d)
        self.in_channels = in_channels
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)          # /2
        self.pool = net.maxpool                                          # /4
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])  # /4 /8 /16 /32
        chans = [m[-1].conv3.out_channels if hasattr(m[-1], "conv3") else m[-1].conv2.out_channels
                 for m in self.layers]
        stem_ch = net.conv1.out_channels
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in chans])
        self.smooth = nn.ModuleList([_ConvGN(fpn_ch, fpn_ch) for _ in chans])
        self.up2 = _ConvGN(fpn_ch + stem_ch, fpn_ch)                     # /2
        self.up1 = _ConvGN(fpn_ch + in_channels, 64)                     # /1
        self.anchor32 = nn.Conv2d(64, 32, 3, padding=1)
        self.anchor64 = nn.Conv2d(64, 64, 3, padding=1)
        self.head = nn.Sequential(_ConvGN(64, 64), nn.Conv2d(64, out_channels, 1))
        self.register_buffer("mean", th.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", th.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def encoder_parameters(self):
        for m in (self.stem, self.layers):
            yield from m.parameters()

    def forward(self, x, hs=None):
        x = x.float()
        # L in [-1, 1] -> gray in [0, 1] -> 3 channels, ImageNet normalisation
        g = ((x[:, :1] + 1) / 2).repeat(1, 3, 1, 1)
        g = (g - self.mean) / self.std
        s2 = self.stem(g)
        f = self.pool(s2)
        feats = []
        for layer in self.layers:
            f = layer(f)
            feats.append(f)
        top = None
        for i in range(len(feats) - 1, -1, -1):
            lat = self.lateral[i](feats[i])
            top = lat if top is None else lat + F.interpolate(top, size=lat.shape[-2:], mode="nearest")
            top = self.smooth[i](top)
        p2 = self.up2(th.cat([F.interpolate(top, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        p1 = self.up1(th.cat([F.interpolate(p2, size=x.shape[-2:], mode="bilinear", align_corners=False), x], 1))
        return (self.anchor32(p1), self.anchor64(p1)), self.head(p1)
