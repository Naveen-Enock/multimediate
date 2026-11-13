"""Training loop — masked MSE + CCC (RegressionCCCLoss), NaN guard, grad clip, weight EMA."""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from .ema import ModelEMA
from .losses import RegressionCCCLoss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: dict, device) -> dict:
    """Move all tensors in the (nested) batch dict to device; leave metadata."""
    def mv(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: mv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [mv(v) for v in x]
        return x
    return {k: mv(v) for k, v in batch.items()}


def make_optimizer(model, config):
    name = config["optimizer"]["name"].lower()
    lr = float(config["optimizer"]["lr"])
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def lr_scale(step: int, warmup_steps: int, total_steps: int | None,
             min_lr_ratio: float = 0.0, cycle_steps: int | None = None) -> float:
    """LR multiplier on ``base_lr``: linear warmup then cosine decay (optionally with warm restarts).

    * ``step < warmup_steps``  -> linear ramp ``(step+1)/warmup_steps`` from ~0 to 1.
    * after warmup, if ``total_steps`` is known -> cosine anneal from 1 down to ``min_lr_ratio``.
      With ``cycle_steps`` set (SGDR-style warm restarts, DAPA MM'25 uses ``T_max`` = 10 epochs over
      a 40-epoch run), the cosine resets to 1 every ``cycle_steps`` post-warmup steps instead of
      decaying once over the whole run; ``total_steps`` still bounds the last (possibly partial)
      cycle. ``cycle_steps`` None/0 -> single decay over ``total_steps - warmup_steps`` (unchanged
      behavior).
    * if ``total_steps`` is None -> hold flat at 1 after warmup (warmup-then-constant).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps is None or total_steps <= warmup_steps:
        return 1.0
    post_warmup = step - warmup_steps
    span = total_steps - warmup_steps
    if cycle_steps:
        progress = (post_warmup % cycle_steps) / cycle_steps
    else:
        progress = post_warmup / span
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))          # 1 -> 0
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class Trainer:
    def __init__(self, model, config, device, logger, rank0: bool = True):
        self.model = model
        # Unwrapped module the EMA reads from: ``model`` here is the pre-DDP module, so its
        # ``state_dict`` keys stay unprefixed and match the checkpoints.
        self.ema_module = model
        self.cfg = config
        self.device = device
        self.logger = logger
        self.rank0 = rank0
        self.opt = make_optimizer(model, config)
        sched = config.get("lr_schedule", {}) or {}
        self.warmup_steps = int(sched["warmup_steps"])
        self.min_lr_ratio = float(sched.get("min_lr_ratio", 0.0))
        # Cosine warm-restart period in epochs (SGDR-style; DAPA MM'25 uses T_max=10 over 40 epochs).
        # None/0 -> single decay over the whole run. Converted to steps once steps_per_epoch is known
        # (configure_schedule()).
        self.cycle_epochs = sched.get("cycle_epochs") or None
        self.cycle_steps: int | None = None
        # Total optimizer steps for the cosine decay denominator. Set by configure_schedule()
        # once the loader lengths are known; None -> warmup-then-constant (no decay).
        self.total_steps: int | None = None
        self.base_lr = float(config["optimizer"]["lr"])
        self.clip = float(config.get("grad_clip_norm", 1.0))
        self.amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.global_step = 0
        closs = config.get("continuous_loss", {}) or {}
        lam = float(closs.get("ccc_lambda", 1.0))
        mse_weight = float(closs.get("mse_weight", 1.0))
        # Horizon-decay weighting on the continuous loss (``continuous_loss.horizon_decay_kappa``,
        # default 0 ⇒ off): a ``(1, 1, K)`` ``exp(−κ·Δt)`` weight broadcast over (B,N,K), monotone
        # from the chunk's own start. Off by default with the centered K-frame chunk (context on
        # both sides, no one-sided reliability gradient); a nonzero value here weights the chunk's
        # leading edge over its trailing edge, which only makes sense if you have a reason to trust
        # the leading side more.
        hdk = float(closs.get("horizon_decay_kappa", 0.0))
        self.horizon_weight = None
        if hdk > 0:
            K = int(config["window"]["K"])
            hw = torch.exp(-hdk * torch.arange(K, dtype=torch.float32, device=device))
            self.horizon_weight = hw.view(1, 1, K)
        # Continuous regression loss: μ·MSE + λ·(1−CCC). μ (mse_weight) toward 0 approaches the
        # pure-CCC objective DAPA (MM'25) trains with.
        self.regression_loss = RegressionCCCLoss(lam=lam, mse_weight=mse_weight)
        # Weight EMA (``train.ema_decay``, 0 ⇒ off). The averaged weights are what validation and the
        # shipped checkpoints read; ``ema_module`` (unwrapped) is stepped after every optimizer update.
        self.ema = ModelEMA(self.ema_module, float((config.get("train", {}) or {}).get("ema_decay", 0.0)))
        self.grad_accum = max(1, int((config.get("train", {}) or {}).get("grad_accum_steps", 1)))

    def configure_schedule(self, total_steps: int, steps_per_epoch: int | None = None):
        """Tell the LR scheduler the full training length (steps) so cosine decay can run.

        ``total_steps`` is the total number of optimizer steps across the whole run
        (steps/epoch × epochs, after any ``max_steps`` cap). Without this the LR holds flat
        at ``base_lr`` after warmup. ``steps_per_epoch`` converts ``lr_schedule.cycle_epochs``
        (if set) into the warm-restart period in steps.
        """
        self.total_steps = int(total_steps)
        if self.cycle_epochs and steps_per_epoch:
            self.cycle_steps = max(1, int(round(steps_per_epoch * self.cycle_epochs)))

    def _set_lr(self):
        scale = lr_scale(self.global_step, self.warmup_steps, self.total_steps,
                         self.min_lr_ratio, self.cycle_steps)
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * scale

    def _dataset_group_ids(self, batch: dict, like: torch.Tensor) -> torch.Tensor:
        """Per-sample integer dataset ids ``(B, 1, 1)`` for CCC segregation.

        ``batch["dataset_id"]`` is a length-B list of dataset-name strings (all nodes of a session
        share its dataset). The actual integer values only need to distinguish the datasets present
        in this batch, so a per-batch enumeration suffices; the shape broadcasts over the ``(N, K)``
        node/horizon axes of ``like`` in the loss.
        """
        ds = batch["dataset_id"]
        order = {name: i for i, name in enumerate(sorted(set(ds)))}
        ids = torch.tensor([order[d] for d in ds], device=like.device, dtype=torch.long)
        return ids.view(-1, 1, 1)

    def _loss_for(self, batch: dict, out: dict):
        """Continuous loss over scored target nodes. Returns ``(loss, parts)``.

        Outputs/masks carry a node axis ``(B, N, K)``; the validity mask is gated by ``is_target``
        so padded and partner-only nodes carry no loss.
        """
        vm = batch["valid_mask"] * batch["is_target"].unsqueeze(-1)   # (B, N, K)
        gid = self._dataset_group_ids(batch, out["pred"])
        loss, rparts = self.regression_loss(
            out["pred"], batch["target_chunk"], vm, group_id=gid,
            horizon_weight=self.horizon_weight, return_parts=True)
        parts = {"loss_cont_mse": float(rparts["mse"]),
                 "loss_cont_ccc": float(1.0 - rparts["ccc"])}
        return loss, parts

    def _do_step(self, loss: torch.Tensor, log_extras: dict | None = None) -> float:
        """Shared backward + optimizer mechanics (AMP, grad clip, logging)."""
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {self.global_step}: {loss.item()}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.ema.update(self.ema_module)
        self.global_step += 1
        if self.rank0 and self.global_step % 5 == 0:
            entry = {"train/loss": loss.item(),
                     "train/grad_norm": float(grad_norm),
                     "train/lr": self.opt.param_groups[0]["lr"]}
            if log_extras:
                entry.update(log_extras)
            self.logger.log(entry, step=self.global_step)
        if self.rank0 and self.global_step % 20 == 0:
            lr = self.opt.param_groups[0]["lr"]
            extras_str = ""
            if log_extras:
                parts = [f"{k.split('/')[-1]}={v:.4f}" for k, v in log_extras.items()]
                extras_str = "  " + "  ".join(parts)
            print(f"  step {self.global_step:>6}  loss={loss.item():.5f}  "
                  f"gnorm={float(grad_norm):.3f}  lr={lr:.2e}{extras_str}", flush=True)
        return loss.item()

    def _step(self, batch: dict) -> float:
        batch = move_to_device(batch, self.device)
        self._set_lr()
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=self.amp):
            loss, parts = self._loss_for(batch, self.model(batch))
        return self._do_step(loss, log_extras={f"train/{k}": v for k, v in parts.items()})

    def train_epoch(self, loader, max_steps=None) -> float:
        """One pass over ``loader`` (capped at ``max_steps`` optimizer steps). Returns mean loss."""
        self.model.train()
        it = iter(loader)
        total, n = 0.0, 0

        if self.grad_accum == 1:
            while True:
                if max_steps is not None and n >= max_steps:
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    break
                total += self._step(batch)
                n += 1
        else:
            # Gradient accumulation: collect grad_accum micro-batches, backward each
            # (loss / grad_accum so the gradient scale is identical to a single large batch),
            # then do one optimizer step. Optimizer steps are the unit for max_steps and
            # global_step — identical to a DDP run with world_size == grad_accum.
            micro_count = 0
            window_loss = 0.0
            self.opt.zero_grad(set_to_none=True)
            while True:
                if max_steps is not None and n >= max_steps:
                    break
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = move_to_device(batch, self.device)
                if micro_count == 0:
                    self._set_lr()
                with torch.autocast("cuda", enabled=self.amp):
                    loss, _ = self._loss_for(batch, self.model(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {self.global_step}: {loss.item()}")
                self.scaler.scale(loss / self.grad_accum).backward()
                window_loss += loss.item()
                micro_count += 1

                if micro_count == self.grad_accum:
                    self.scaler.unscale_(self.opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.ema.update(self.ema_module)
                    self.global_step += 1
                    self.opt.zero_grad(set_to_none=True)

                    avg_loss = window_loss / self.grad_accum
                    if self.rank0 and self.global_step % 5 == 0:
                        self.logger.log(
                            {"train/loss": avg_loss,
                             "train/grad_norm": float(grad_norm),
                             "train/lr": self.opt.param_groups[0]["lr"]},
                            step=self.global_step)
                    if self.rank0 and self.global_step % 20 == 0:
                        print(f"  step {self.global_step:>6}  loss={avg_loss:.5f}  "
                              f"gnorm={float(grad_norm):.3f}  "
                              f"lr={self.opt.param_groups[0]['lr']:.2e}  "
                              f"(accum×{self.grad_accum})", flush=True)
                    total += avg_loss
                    n += 1
                    micro_count = 0
                    window_loss = 0.0

        return total / max(n, 1)
