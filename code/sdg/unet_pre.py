"""
U-Net with an ImageNet-pretrained ResNet encoder.

Why this exists: ADA (MICCAI 2025) reports its numbers on an ImageNet-pretrained DeepLabv3+/
MobileNetV2, while our head-to-head used a U-Net trained from scratch. Their ERM on I2CVB is 64.46
Dice; ours on the same transfer is 0.123. A gap that size cannot be method-related, so **until this
arm runs, nothing may be said about absolute numbers or about beating a published method.**

The arm asks two questions the scratch arm cannot:
  1. Does pretraining change the *ranking* of the seven methods, or only the level?
  2. Does the cross-benchmark reversal survive a pretrained backbone?

Deliberate choices, recorded rather than hidden:
  * Encoder: torchvision ResNet (34 or 50), `IMAGENET1K_V2` where available. Weights must be
    pre-cached on the Sonic login node — GPU nodes have no internet.
  * 1-channel input: prostate MRI is grayscale, so the first convolution's weights are summed
    across the RGB axis and applied to the single channel. This is the standard adaptation and it
    preserves the filter response scale, unlike replicating the channel three times.
  * Normalisation: we keep the pipeline's own normalisation (per-case z-score for MRI, per-image
    z-score for fundus) rather than ImageNet statistics, because changing it would confound the
    backbone comparison with a preprocessing change. This is a deviation from the usual recipe and
    is stated in the results.
  * Style hooks sit after the stem and after layer1, i.e. on low-level features, matching where
    MixStyle/DSU/MaxStyle are placed in the scratch model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _stem_to_1ch(conv):
    w = conv.weight.data.sum(1, keepdim=True)
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    new.weight.data = w
    return new


class Up(nn.Module):
    def __init__(self, ci, cs, co):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ci + cs, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.01, True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.f(torch.cat([x, skip], 1))


class PretrainedUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, arch='resnet34', style=None, n_style=2, pretrained=True):
        super().__init__()
        w = {'resnet34': torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
             'resnet50': torchvision.models.ResNet50_Weights.IMAGENET1K_V2}[arch] if pretrained else None
        net = getattr(torchvision.models, arch)(weights=w)
        self.stem = nn.Sequential(_stem_to_1ch(net.conv1) if in_ch == 1 else net.conv1,
                                  net.bn1, net.relu)
        self.pool = net.maxpool
        self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
        c = [64, 256, 512, 1024, 2048] if arch == 'resnet50' else [64, 64, 128, 256, 512]
        self.u4 = Up(c[4], c[3], 256)
        self.u3 = Up(256, c[2], 128)
        self.u2 = Up(128, c[1], 64)
        self.u1 = Up(64, c[0], 64)
        self.head = nn.Conv2d(64, out_ch, 1)
        self.styles = nn.ModuleList([style() for _ in range(n_style)]) if style is not None else None

    def freeze_encoder(self, level):
        """level 0 = nothing frozen (ordinary fine-tuning);
           level 1 = stem + layer1 frozen (the low-level, most transferable features);
           level 2 = the whole encoder frozen, only the decoder and head train.

        The point of exposing this is a measurement: single-source fine-tuning may be *destroying*
        the domain robustness that pretraining supplied, in which case the SDG problem is not
        "manufacture diversity" (whose ceiling we measured at ~0) but "keep the representation".
        Frozen BatchNorm statistics matter too -- a frozen encoder must also stop updating its
        running statistics, or the source domain leaks in through them.
        """
        groups = {1: [self.stem, self.l1],
                  2: [self.stem, self.l1, self.l2, self.l3, self.l4]}.get(level, [])
        for g in groups:
            for p in g.parameters():
                p.requires_grad_(False)
            g.eval()                       # freeze BN running stats as well
        self._frozen = groups

    def train(self, mode=True):
        super().train(mode)
        for g in getattr(self, '_frozen', []):
            g.eval()
        return self

    def forward(self, x, return_feat=False, feat_override=None):
        if feat_override is not None:
            return self.head(feat_override)
        h, w = x.shape[-2:]
        s0 = self.stem(x)
        if self.styles is not None and len(self.styles) > 0:
            s0 = self.styles[0](s0)
        s1 = self.l1(self.pool(s0))
        if self.styles is not None and len(self.styles) > 1:
            s1 = self.styles[1](s1)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.u4(s4, s3)
        d = self.u3(d, s2)
        d = self.u2(d, s1)
        d = self.u1(d, s0)
        d = F.interpolate(d, size=(h, w), mode='bilinear', align_corners=False)
        return (self.head(d), d) if return_feat else self.head(d)
