"""
Improved SegFormer3D+ model for 3D medical image segmentation.
Enhancements:
- Hybrid wavelet + convolutional stem
- Dual attention with residual fusion
- Flexible decoder with deeper fusion and deep supervision
- Optional edge-aware loss and radiomics-guided fusion
- Lightweight modular blocks and optimized memory usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class WaveletStem(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.low  = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.high = nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        nn.init.constant_(self.low.weight, 1 / 27.0)
        nn.init.kaiming_uniform_(self.high.weight)

    def forward(self, x):
        low  = self.low(x)
        high = self.high(x) - low
        return torch.cat([low, high], dim=1)


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch, out_ch, patch):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, patch, stride=patch)

    def forward(self, x):
        return self.proj(x)


class MixFFN(nn.Module):
    def __init__(self, dim, ratio=3):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv3d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, x, D, H, W):
        x = self.drop(self.act(self.fc1(x)))
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop(self.act(self.fc2(x)))
        return x


class SegFormerMixerBlock(nn.Module):
    def __init__(self, dim, heads, ratio=3, use_ckpt=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, ratio)
        self.use_ckpt = use_ckpt

    def forward(self, x, D, H, W):
        def fn(y):
            y = y + self.attn(self.norm1(y), self.norm1(y), self.norm1(y))[0]
            y = y + self.mlp(self.norm2(y), D, H, W)
            return y

        return checkpoint(fn, x) if self.use_ckpt and self.training else fn(x)


class DualAttentionBlock3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_ch, in_ch // 8, 1),
            nn.ReLU(),
            nn.Conv3d(in_ch // 8, in_ch, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv3d(in_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


class SegFormer3DPlus(nn.Module):
    def __init__(self, in_ch=4, num_classes=4, depths=(2,2,2,2), dims=(48,96,192,384), heads=(4,4,6,8),
                 patch_sizes=((4,4,4),(2,2,2),(2,2,2),(2,2,2)), use_ckpt=False, use_radiomics=False, edge_supervision=False):
        super().__init__()
        self.use_radiomics = use_radiomics
        self.edge_supervision = edge_supervision

        self.stem = WaveletStem(in_ch)
        ch = in_ch * 2

        self.patch_embeds = nn.ModuleList([PatchEmbed3D(ch if i==0 else dims[i-1], dims[i], p) for i, p in enumerate(patch_sizes)])

        self.encoders = nn.ModuleList([
            nn.ModuleList([SegFormerMixerBlock(dims[i], heads[i], use_ckpt=use_ckpt) for _ in range(depths[i])])
            for i in range(len(depths))
        ])

        self.dual_attn = DualAttentionBlock3D(dims[-1])

        self.fuse_layers = nn.ModuleList([nn.Conv3d(d, 256, 1) for d in dims])
        self.class_head = nn.Conv3d(256, num_classes, 1)
        if edge_supervision:
            self.edge_head = nn.Conv3d(256, 1, 1)
        if use_radiomics:
            self.fuse_gate = nn.Sequential(nn.Linear(256, len(dims)), nn.Sigmoid())
            self.ema_proj = nn.Linear(256, 256)
            self.distill_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x, radiomics_vec=None):
        ori_sz = x.shape[-3:]
        x = self.stem(x)
        feats = []
        for pe, blocks in zip(self.patch_embeds, self.encoders):
            x = pe(x)
            B, C, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in blocks:
                x = blk(x, D, H, W)
            x = x.transpose(1, 2).reshape(B, C, D, H, W)
            feats.append(x)

        feats[-1] = self.dual_attn(feats[-1])
        tgt_sz = feats[0].shape[-3:]
        weights = self.fuse_gate(radiomics_vec) if self.use_radiomics and radiomics_vec is not None else None

        fused = 0
        for i, (f, conv) in enumerate(zip(feats, self.fuse_layers)):
            up = F.interpolate(conv(f), size=tgt_sz, mode='trilinear', align_corners=False)
            if weights is not None:
                up = up * weights[:, i].view(-1, 1, 1, 1, 1)
            fused += up

        fused = F.relu(fused)
        logits = self.class_head(fused)
        if logits.shape[-3:] != ori_sz:
            logits = F.interpolate(logits, size=ori_sz, mode='trilinear', align_corners=False)

        out = [logits]
        if self.edge_supervision:
            edge = self.edge_head(fused)
            edge = F.interpolate(edge, size=ori_sz, mode='trilinear', align_corners=False)
            out.append(edge)

        if self.use_radiomics and radiomics_vec is not None:
            student_vec = self.ema_proj(torch.mean(fused, dim=(-3, -2, -1)))
            distill = self.distill_loss(F.log_softmax(student_vec/2.0, dim=-1),
                                        F.softmax(radiomics_vec.detach()/2.0, dim=-1))
            out.append(distill)

        return tuple(out) if len(out) > 1 else out[0]


def get_segformer3d_plus(num_classes=4, use_checkpoint=True, use_radiomics=True, edge_supervision=True):
    return SegFormer3DPlus(
        in_ch=4,
        num_classes=num_classes,
        depths=(2,2,2,2),
        dims=(48,96,192,384),
        heads=(4,4,6,8),
        patch_sizes=((4,4,4),(2,2,2),(2,2,2),(2,2,2)),
        use_ckpt=use_checkpoint,
        use_radiomics=use_radiomics,
        edge_supervision=edge_supervision
    )
