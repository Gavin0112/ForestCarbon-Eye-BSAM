"""Independent BSAM two-pass update; weighted ascent, original-MSE descent."""
from __future__ import annotations
import torch
from torch import nn


class BalancedSharpnessStep:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, rho: float):
        self.model, self.optimizer, self.rho = model, optimizer, float(rho)
        assert self.rho > 0
        self.offsets = None

    @torch.no_grad()
    def perturb(self):
        grads = [p.grad.detach().float() for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        assert grads
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        assert bool(torch.isfinite(norm)) and float(norm) > 0
        self.offsets = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                eps = (self.rho / norm) * p.grad.detach()
                p.add_(eps)
                self.offsets.append((p, eps))
        actual = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(e.float()) for _, e in self.offsets]))
        assert bool(torch.isfinite(actual))
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm), float(actual)

    @torch.no_grad()
    def restore(self):
        assert self.offsets is not None
        for p, eps in self.offsets:
            p.sub_(eps)
        self.offsets = None


class BatchNormSecondPass:
    """Keep BN batch normalization in train mode; prevent a second stats update."""
    def __init__(self, model: nn.Module):
        self.layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.saved = None

    @torch.no_grad()
    def freeze_running_update(self):
        assert self.saved is None
        self.saved = [(m, m.momentum, m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None)
                      for m in self.layers]
        for m in self.layers:
            assert m.training
            m.momentum = 0.0

    @torch.no_grad()
    def restore_running_update(self):
        assert self.saved is not None
        for m, momentum, counter in self.saved:
            m.momentum = momentum
            if counter is not None:
                m.num_batches_tracked.copy_(counter)
        self.saved = None
