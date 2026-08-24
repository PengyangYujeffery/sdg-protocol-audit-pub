"""
Segmentation head on a DINOv2 ViT encoder — the third backbone of the backbone arm.

Why a third backbone. The backbone axis is the largest protocol effect we measure: ImageNet
pretraining alone is worth more to ERM than the entire augmentation literature, and it compresses
the spread between methods. Two points (scratch CNN, ImageNet CNN) establish that the ranking moves;
they do not establish whether it keeps moving. DINOv2 is the natural third point because
self-supervised features are explicitly *sold* on domain robustness, so it is the case where the
method effects should shrink most.

Deliberate deviations, stated here rather than buried, because a backbone comparison is only
readable if the differences are known:

  * Decoder. A plain ViT emits every block at one resolution (H/14), so there is no CNN-style
    pyramid to build a U-Net over. We follow DINOv2's own segmentation protocol and use a simple
    decoder: four intermediate blocks are concatenated at H/14, projected, and upsampled. This is
    *not* the same decoder as the ResNet arm, and the comparison is therefore
    backbone-plus-decoder, not backbone alone. No claim is made that isolates the encoder.
  * Input size. DINOv2 requires a multiple of the patch size (14). Inputs are resized to the
    nearest multiple and the logits are resized back, so the loss and the metrics stay on the
    original grid.
  * 1-channel input. As in the ResNet arm, the patch-embedding weights are summed across the RGB
    axis rather than the channel being replicated, which preserves the filter response scale.
  * Style hooks. MixStyle/DSU/MaxStyle perturb channel statistics of low-level CNN features.
    There is no exact analogue in a ViT; we apply them to the reshaped feature map of the earliest
    selected block, which is the closest available position. This is a deviation and any
    style-method number on this backbone must be read with it in mind.
  * Normalisation. The pipeline's own normalisation is kept, not ImageNet/DINOv2 statistics,
    for the same reason as the ResNet arm: changing it would confound backbone with preprocessing.

Weights must be pre-cached on the login node; compute nodes have no internet. Set
`HF_HUB_OFFLINE=1` in the job script.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 14
BLOCKS = (2, 5, 8, 11)          # evenly spaced through the 12 blocks of dinov2-base


def _round14(n):
    return max(PATCH, int(round(n / PATCH)) * PATCH)


class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co),
            nn.LeakyReLU(0.01, True))

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.f(x)


class DinoUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, name='facebook/dinov2-base', style=None, n_style=2,
                 pretrained=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        if pretrained:
            self.enc = AutoModel.from_pretrained(name)
        else:
            # architecture control, exactly as `--pretrained 0` does for the ResNet arm
            self.enc = AutoModel.from_config(AutoConfig.from_pretrained(name))
        d = self.enc.config.hidden_size
        if in_ch == 1:
            proj = self.enc.embeddings.patch_embeddings.projection
            new = nn.Conv2d(1, proj.out_channels, proj.kernel_size, proj.stride, bias=proj.bias is not None)
            new.weight.data = proj.weight.data.sum(1, keepdim=True)
            if proj.bias is not None:
                new.bias.data = proj.bias.data.clone()
            self.enc.embeddings.patch_embeddings.projection = new
            self.enc.embeddings.patch_embeddings.num_channels = 1
        self.proj = nn.Conv2d(d * len(BLOCKS), 256, 1)
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        self.u3 = Up(64, 64)
        self.head = nn.Conv2d(64, out_ch, 1)
        self.style_proj = nn.Conv2d(d, d, 1) if style is not None else None
        self.styles = nn.ModuleList([style() for _ in range(n_style)]) if style is not None else None

    def freeze_encoder(self, level):
        """1 = patch embedding + the first third of the blocks; 2 = the whole encoder."""
        blocks = self.enc.encoder.layer
        groups = {1: [self.enc.embeddings] + list(blocks[:len(blocks) // 3]),
                  2: [self.enc]}.get(level, [])
        for g in groups:
            for p in g.parameters():
                p.requires_grad_(False)
            g.eval()
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
        H, W = _round14(h), _round14(w)
        xi = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) \
            if (H, W) != (h, w) else x
        out = self.enc(pixel_values=xi, output_hidden_states=True)
        gh, gw = H // PATCH, W // PATCH
        feats = []
        for i, b in enumerate(BLOCKS):
            t = out.hidden_states[b + 1][:, 1:, :]                  # drop the CLS token
            f = t.transpose(1, 2).reshape(t.shape[0], -1, gh, gw)
            if i == 0 and self.styles is not None and len(self.styles) > 0:
                f = self.styles[0](f)
            if i == 1 and self.styles is not None and len(self.styles) > 1:
                f = self.styles[1](f)
            feats.append(f)
        z = self.proj(torch.cat(feats, 1))
        z = self.u1(z)
        z = self.u2(z)
        z = self.u3(z)
        d = F.interpolate(z, size=(h, w), mode='bilinear', align_corners=False)
        return (self.head(d), d) if return_feat else self.head(d)
