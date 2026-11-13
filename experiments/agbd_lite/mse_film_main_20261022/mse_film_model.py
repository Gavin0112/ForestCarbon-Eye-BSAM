"""Single-model modality-specific entry for the local N5/FCN-FiLM backbone."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/agbd_lite/n5_capacity_control'))
sys.path.insert(0, str(ROOT / 'third_party/AGBD-GFMs/model'))
from n5_model import NicoNet_CapacityControl
from nico_net_film import PointwiseBlock


class GroupedS2(nn.Module):
    """Wavelength-aware small groups, not a hyperspectral Mamba scan."""
    GROUPS = ((0, 1, 2, 3), (4, 5, 6), (7, 8), (10, 11))

    def __init__(self):
        super().__init__()
        self.group_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(len(ids), 8, 1), nn.GELU(),
                          nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.GELU())
            for ids in self.GROUPS
        ])
        # B09 is water-vapour, retained as absolute information outside four groups.
        self.mix = nn.Sequential(nn.Conv2d(12 + 1 + 32, 64, 1), nn.GELU())

    def forward(self, s2):
        groups = [layer(s2[:, ids]) for ids, layer in zip(self.GROUPS, self.group_layers)]
        return self.mix(torch.cat((s2, s2[:, 9:10], *groups), 1))


class TerrainEncoder(nn.Module):
    """AW3D30 DSM derivatives in metres on the 10 m resampled S2 grid."""
    def __init__(self, mean_m: float, std_m: float):
        super().__init__()
        self.register_buffer('mean_m', torch.tensor(float(mean_m)))
        self.register_buffer('std_m', torch.tensor(float(std_m)))
        self.stem = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.GELU())

    def forward(self, normalized_dem):
        z = normalized_dem.float() * self.std_m + self.mean_m
        p = F.pad(z, (1, 1, 1, 1), mode='replicate')
        dzdx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / 20.0
        dzdy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / 20.0
        slope = torch.atan(torch.sqrt(dzdx.square() + dzdy.square())) / (torch.pi / 2)
        p5 = F.pad(z, (2, 2, 2, 2), mode='replicate')
        local_mean = F.avg_pool2d(p5, 5, stride=1)
        relief = F.max_pool2d(p5, 5, stride=1) + F.max_pool2d(-p5, 5, stride=1)
        center = z[:, :, 12:13, 12:13]
        inputs = torch.cat((normalized_dem.float(), torch.tanh((z - center) / 100),
                            slope, torch.tanh((z - local_mean) / 100),
                            torch.tanh(relief / 100)), 1)
        return self.stem(inputs)


class LandCoverEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(23, 8)  # 22 observed classes + unknown
        self.stem = nn.Sequential(nn.Conv2d(8 + 3 + 22, 16, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())

    def forward(self, lc):
        onehot = lc[:, :22]
        cls = onehot.argmax(1)
        cls = torch.where(onehot.sum(1) > 0, cls, 22)
        emb = self.embedding(cls).permute(0, 3, 1, 2)
        composition = F.avg_pool2d(onehot, 5, stride=1, padding=2)
        # Original 3 validity/confidence channels remain available.
        return self.stem(torch.cat((emb, composition, lc[:, 23:26]), 1))


class RadarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.GELU())

    def forward(self, radar, dn):
        # DN>0: numeric input exists. DN=1 remains physically unresolved and is
        # excluded only from texture estimation, while its absolute value/flag stay.
        valid = (dn > 0).all(dim=1, keepdim=True).float()
        reliable_texture = (dn > 1).all(dim=1, keepdim=True).float()
        local_count = F.avg_pool2d(reliable_texture, 3, 1, 1).clamp_min(1 / 9)
        local = F.avg_pool2d(radar * reliable_texture, 3, 1, 1) / local_count
        texture = (radar - local) * reliable_texture
        hh, hv = radar[:, :1], radar[:, 1:2]
        ratio = (hh - hv) / (hh + hv + 0.05)
        features = torch.cat((radar, hh - hv, ratio, texture, valid,
                              reliable_texture, ((dn == 1).any(1, keepdim=True)).float()), 1)
        return self.stem(features)


class MSEFilm(nn.Module):
    def __init__(self, dem_mean_m: float, dem_std_m: float):
        super().__init__()
        base = NicoNet_CapacityControl()
        self.backbone = base.base.backbone
        self.adapter = base.adapter
        self.s2 = GroupedS2()
        self.dem = TerrainEncoder(dem_mean_m, dem_std_m)
        self.lc = LandCoverEncoder()
        self.alos = RadarEncoder()
        self.backbone.body.entry_block = PointwiseBlock(112, [128, 256, 256], biome_dim=64)
        self.backbone.in_features = 112
        self.backbone.body.entry_block.register_forward_hook(self._adapt)

    def _adapt(self, _module, _inputs, output):
        return self.adapter(output)

    def forward(self, s2, dem, lc, meta, radar, dn):
        f = torch.cat((self.s2(s2), self.dem(dem), self.lc(lc), self.alos(radar, dn)), 1)
        assert f.shape[1:] == (112, 25, 25) and meta.shape[1] == 8
        return self.backbone((f, meta))[:, 0, 12, 12]
