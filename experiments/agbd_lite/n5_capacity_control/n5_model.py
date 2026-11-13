"""N5: parameter-matched pointwise residual capacity control for N4."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

N1_DIR = Path(__file__).resolve().parent.parent / 'n1_niconet_mrf'
sys.path.insert(0, str(N1_DIR))
from model import N1NicoNet


class CapacityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv2d(256, 421, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(421, 256, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class NicoNet_CapacityControl(nn.Module):
    def __init__(self):
        super().__init__()
        # Build the N1 backbone first to preserve its seeded initial tensors.
        self.base = N1NicoNet(False)
        self.adapter = CapacityAdapter()
        self.base.backbone.body.entry_block.register_forward_hook(self._adapt_entry)

    def _adapt_entry(self, _module, _inputs, output):
        return self.adapter(output)

    def forward(self, s2, dsm, lc, meta):
        return self.base(s2, dsm, lc, meta)
