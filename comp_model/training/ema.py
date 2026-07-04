"""Exponential moving average of model weights.

Keeps a shadow copy of every floating-point tensor in ``model.state_dict()`` (parameters + float
buffers) and, after each optimizer step, pulls it toward the live weights: ``s ← d·s + (1−d)·θ``.
The averaged weights are the ones validated and shipped. Non-float buffers (int step counters, bool
masks) are left to the live module. ``decay <= 0`` disables the tracker: every method becomes a no-op
and ``applied_to`` / ``merged_state_dict`` fall through to the live weights.
"""

from __future__ import annotations

import contextlib

import torch


class ModelEMA:
    def __init__(self, model, decay: float, warmup: bool = True):
        self.decay = float(decay)
        self.enabled = self.decay > 0.0
        self.warmup = warmup           # ramp the decay in early
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        if self.enabled:
            self.shadow = {k: v.detach().clone().float()
                           for k, v in model.state_dict().items()
                           if torch.is_floating_point(v)}

    @torch.no_grad()
    def update(self, model) -> None:
        if not self.enabled:
            return
        self.num_updates += 1
        d = self.decay
        if self.warmup:                # min(decay, (1+t)/(10+t)) — warmup ramp
            d = min(d, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is not None:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def merged_state_dict(self, module) -> dict:
        """A full ``module`` state_dict with the EMA floats substituted in (non-float buffers kept).

        Returns detached clones so the result is safe to ``torch.save`` without aliasing live tensors.
        """
        sd = module.state_dict()
        if not self.enabled:
            return sd
        return {k: (self.shadow[k].to(v.dtype).clone() if k in self.shadow else v)
                for k, v in sd.items()}

    @contextlib.contextmanager
    def applied_to(self, module):
        """Temporarily swap the EMA weights into ``module`` (restored on exit). No-op if disabled."""
        if not self.enabled:
            yield
            return
        live = module.state_dict()
        backup = {k: live[k].detach().clone() for k in self.shadow if k in live}
        for k, s in self.shadow.items():
            if k in live:
                live[k].copy_(s.to(live[k].dtype))
        try:
            yield
        finally:
            cur = module.state_dict()
            for k, b in backup.items():
                cur[k].copy_(b)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, sd: dict | None) -> None:
        if not self.enabled or not sd:
            return
        self.num_updates = int(sd.get("num_updates", 0))
        for k, v in (sd.get("shadow") or {}).items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
