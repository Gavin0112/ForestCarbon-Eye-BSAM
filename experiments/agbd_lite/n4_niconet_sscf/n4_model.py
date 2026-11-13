"""N4: spectral/spatial interaction at the unchanged NicoNet entry feature map."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

N1 = Path(__file__).resolve().parent.parent / "n1_niconet_mrf"
sys.path.insert(0, str(N1))
from model import N1NicoNet


class SpectralPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Linear(3, 32)
        self.band_position = nn.Parameter(torch.zeros(1, 12, 32))
        nn.init.normal_(self.band_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.band_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.coefficient = nn.Linear(32, 1)
        nn.init.normal_(self.coefficient.weight, std=0.02)
        nn.init.zeros_(self.coefficient.bias)
        self.projection = nn.Sequential(
            nn.Conv2d(12, 64, 1), nn.GELU(), nn.Conv2d(64, 128, 1))

    def forward(self, s2: torch.Tensor):
        # Descriptors are in FP32; the surrounding convolutions follow autocast.
        values = s2.float()
        centre = values[:, :, 11:14, 11:14].mean((2, 3))
        whole = values.mean((2, 3))
        spread = values.std((2, 3), correction=0)
        tokens = torch.stack((centre, whole, spread), -1)
        encoded = self.band_encoder(self.token_embed(tokens) + self.band_position)
        coefficients = 1.0 + 0.5 * torch.tanh(self.coefficient(encoded).squeeze(-1))
        feature = self.projection(s2 * coefficients[:, :, None, None].to(s2.dtype))
        return feature, coefficients


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(39, 64, 3, padding=1),
                                  nn.GroupNorm(8, 64), nn.GELU())
        self.dw3 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.dw5 = nn.Conv2d(64, 64, 5, padding=2, groups=64)
        self.projection = nn.Conv2d(64, 128, 1)

    def forward(self, x: torch.Tensor):
        z = self.stem(x)
        return self.projection(self.dw3(z) + self.dw5(z))


class NicoNet_SSCF(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct the original backbone first so its random initialization
        # exactly matches the seeded N1 M0 construction.
        self.base = N1NicoNet(False)
        self.spectral = SpectralPath()
        self.spatial = SpatialPath()
        self.interaction = nn.Sequential(
            nn.Conv2d(384, 256, 1), nn.GroupNorm(16, 256), nn.GELU())
        self.delta = nn.Conv2d(256, 256, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.capture = False
        self.aux = None
        self._entry_hook = self.base.backbone.body.entry_block.register_forward_hook(self._enhance_entry)

    def _enhance_entry(self, _module, inputs, entry):
        x = inputs[0]
        spec, coefficients = self.spectral(x[:, :12])
        spat = self.spatial(x)
        inter = self.interaction(torch.cat((spec, spat, spec * spat), 1))
        delta = self.delta(inter)
        if self.capture:
            self.aux = dict(spectral_rms=spec.detach().float().square().mean().sqrt(),
                            spatial_rms=spat.detach().float().square().mean().sqrt(),
                            interaction_rms=inter.detach().float().square().mean().sqrt(),
                            delta_rms=delta.detach().float().square().mean().sqrt(),
                            coefficient_std=coefficients.detach().float().std())
        return entry + delta

    def forward(self, s2, dsm, lc, meta):
        return self.base(s2, dsm, lc, meta)
