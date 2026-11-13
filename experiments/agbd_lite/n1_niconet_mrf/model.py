"""N1 matched NicoNet models. The upstream backbone is left intact."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/AGBD-GFMs/model"))
from nico_net_film import NicoNet_FiLM


class ModalityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2_projection = nn.Conv2d(12, 16, 1)
        self.dsm_projection = nn.Conv2d(1, 8, 1)
        self.lc_projection = nn.Conv2d(26, 8, 1)
        self.dsm_gate = nn.Sequential(nn.Linear(24, 16), nn.GELU(), nn.Linear(16, 1))
        self.lc_gate = nn.Sequential(nn.Linear(24, 16), nn.GELU(), nn.Linear(16, 1))
        for mlp in (self.dsm_gate, self.lc_gate):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(self, s2, dsm, lc):
        # Cropping before a 1x1 convolution is exactly equivalent to cropping after it.
        center = (slice(None), slice(None), slice(11, 14), slice(11, 14))
        s = self.s2_projection(s2[center]).mean((2, 3))
        d = self.dsm_projection(dsm[center]).mean((2, 3))
        l = self.lc_projection(lc[center]).mean((2, 3))
        gd = 1 + 0.5 * torch.tanh(self.dsm_gate(torch.cat((s, d), 1)))
        gl = 1 + 0.5 * torch.tanh(self.lc_gate(torch.cat((s, l), 1)))
        return gd, gl


class N1NicoNet(nn.Module):
    def __init__(self, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.backbone = NicoNet_FiLM(
            in_features=39, num_outputs=1, emb_dim=8, biome_dim=64,
            num_sepconv_blocks=8, num_sepconv_filters=256, long_skip=True,
            returns="dense", patch_size=25, only_entry=True,
            linear_emb=False, padding_mode="zeros", sigreg_lambda=0.0)
        if adaptive:
            self.gate = ModalityGate()

    def forward(self, s2, dsm, lc, meta, return_gates=False):
        if self.adaptive:
            gd, gl = self.gate(s2, dsm, lc)
            x = torch.cat((s2, gd[:, :, None, None] * dsm,
                           gl[:, :, None, None] * lc), 1)
        else:
            x = torch.cat((s2, dsm, lc), 1)
            gd = gl = None
        prediction = self.backbone((x, meta))[:, 0, 12, 12]
        return (prediction, gd, gl) if return_gates else prediction
