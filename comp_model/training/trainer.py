"""Training loop — joint multi-task, NaN guard, grad clip.

Continuous batches -> masked MSE (diffusion noise-pred or regression); categorical (PInSoRo)
batches -> an unweighted cross-entropy on the social/task heads, scored on a 1 Hz subsample
of the horizon.

``train_epoch`` accepts one or more loaders (one per label kind). When multiple loaders are
given, ``_joint_step`` pulls one batch from *each* loader and sums their scaled losses before
a single backward + optimizer step. The shorter loader is cycled so every step sees all task types.
"""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch


def _endless(loader):
    """Yield batches forever by re-iterating the loader.

    Unlike ``itertools.cycle``, this does not retain a copy of each yielded batch (which would pin
    the DataLoader IPC tensors in /dev/shm for the whole epoch).
    """
    while True:
        for batch in loader:
            yield batch

from ..data.class_weights import pinsoro_class_priors
from .ema import ModelEMA
from .losses import DiffusionHybridCCCLoss, RegressionCCCLoss, masked_ce, masked_mse


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _valid_class_array(logits: torch.Tensor, target: torch.Tensor):
    """Argmax predicted class indices over valid (non -1) positions, as a 1-D numpy array."""
    valid = target.reshape(-1) != -1
    pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)[valid]
    return pred.detach().to("cpu").numpy()


def _valid_target_array(target: torch.Tensor):
    """Ground-truth class indices over valid (non -1) positions, as a 1-D numpy array."""
    flat = target.reshape(-1)
    return flat[flat != -1].detach().to("cpu").numpy()


def _class_count_hist(indices: np.ndarray, num_classes: int):
    """Per-class count histogram (one bin per class) as a ``(counts, bin_edges)`` pair.

    ``bin_edges`` are integer boundaries ``[0, 1, …, num_classes]`` so class ``i`` lands in bin
    ``i`` — a clean discrete histogram rather than continuous auto-binning of the index values."""
    counts = np.bincount(indices.astype(np.int64), minlength=num_classes)[:num_classes]
    return counts, np.arange(num_classes + 1)


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
    # Only optimize trainable params (some are frozen per phase).
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def lr_scale(step: int, warmup_steps: int, total_steps: int | None,
             min_lr_ratio: float = 0.0) -> float:
    """LR multiplier on ``base_lr``: linear warmup then cosine decay to ``min_lr_ratio``.

    * ``step < warmup_steps``  -> linear ramp ``(step+1)/warmup_steps`` from ~0 to 1.
    * after warmup, if ``total_steps`` is known -> cosine anneal from 1 down to ``min_lr_ratio``
      over the remaining ``total_steps - warmup_steps`` steps (``progress`` clamped to [0, 1] so it
      holds at the floor past the end).
    * if ``total_steps`` is None -> hold flat at 1 after warmup (warmup-then-constant).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps is None or total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
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
        # Total optimizer steps for the cosine decay denominator. Set by configure_schedule()
        # once the loader lengths are known; None -> warmup-then-constant (no decay).
        self.total_steps: int | None = None
        self.base_lr = float(config["optimizer"]["lr"])
        self.clip = float(config.get("grad_clip_norm", 1.0))
        self.amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.global_step = 0
        lam = float((config.get("diffusion", {}) or {}).get("ccc_lambda", 1.0))
        # The loss un-standardizes the recovered ŷ₀ back to raw scale using the per-sample stats the
        # model emits in ``out["y_mean"]/["y_std"]`` and segregates the CCC by dataset.
        self.diffusion_loss = DiffusionHybridCCCLoss(lam=lam)
        # Horizon-decay weighting on the continuous loss (``diffusion.horizon_decay_kappa``, 0 ⇒ off):
        # a ``(1, 1, K)`` ``exp(−κ·Δt)`` weight broadcast over (B,N,K), mirroring the inference
        # ensemble decay (``ensemble.decay_kappa``).
        hdk = float((config.get("diffusion", {}) or {}).get("horizon_decay_kappa", 0.0))
        self.horizon_weight = None
        if hdk > 0:
            K = int(config["window"]["K"])
            hw = torch.exp(-hdk * torch.arange(K, dtype=torch.float32, device=device))
            self.horizon_weight = hw.view(1, 1, K)
        # Deterministic-regression continuous loss (``model.use_diffusion=false``): MSE + λ·(1−CCC).
        self.regression_loss = RegressionCCCLoss(lam=lam)
        # Weight EMA (``train.ema_decay``, 0 ⇒ off). The averaged weights are what validation and the
        # shipped checkpoints read; ``ema_module`` (unwrapped) is stepped after every optimizer update.
        self.ema = ModelEMA(self.ema_module, float((config.get("train", {}) or {}).get("ema_decay", 0.0)))
        # PInSoRo categorical objective: a (train-time logit-adjusted) cross-entropy, scored on a
        # 1 Hz subsample of the K-frame horizon (every ``pred_stride``-th frame).
        pin = config.get("pinsoro", {}) or {}
        self.pred_stride = max(1, int(pin.get("pred_stride", 25)))
        # Cadence (in steps) for the PInSoRo pred/GT class histograms (collapse detection).
        self.hist_every = int((config.get("train", {}) or {}).get("hist_every", 50))
        self.grad_accum = max(1, int((config.get("train", {}) or {}).get("grad_accum_steps", 1)))
        # Train-time logit adjustment, set per domain: the categorical CE is scored on
        # ``f_c − τ_d·α·log π_c^nat`` where ``π^nat`` is each domain's natural class prior over the
        # training splits, ``α`` the sampler exponent, and ``τ_d`` (``train_logit_adjust_tau_{cc,cr}``)
        # the per-domain strength. ``τ_d`` is baked into the stored log-prior rows, so ``masked_ce``
        # applies a uniform ``−α·log_prior``. Rows are keyed by ``self._la_dom``.
        self.sampler_alpha = float(pin.get("sampler_alpha", 0.5))
        tau_by_dom = {"cc": float(pin.get("train_logit_adjust_tau_cc", 0.0)),
                      "cr": float(pin.get("train_logit_adjust_tau_cr", 0.0))}
        self._la_active = any(t > 0 for t in tau_by_dom.values())
        self._la_dom, self._la_social, self._la_task = None, None, None
        if self._la_active:
            domains = ["cc", "cr"]
            roots = config["data"]["roots"]
            # Count the natural prior over the same splits the model trains on (train+val by default).
            splits = tuple(config["data"].get("train_splits", ["train"]))
            soc_rows, tsk_rows = [], []
            for d in domains:
                sp, tp = pinsoro_class_priors(roots, ["pinsoro"], domain=d, splits=splits)
                soc_rows.append(tau_by_dom[d] * np.log(sp))     # per-domain τ_d baked into the row
                tsk_rows.append(tau_by_dom[d] * np.log(tp))
            self._la_dom = {d: i for i, d in enumerate(domains)}
            self._la_social = torch.tensor(np.stack(soc_rows), dtype=torch.float32, device=device)
            self._la_task = torch.tensor(np.stack(tsk_rows), dtype=torch.float32, device=device)

    def configure_schedule(self, total_steps: int):
        """Tell the LR scheduler the full training length (steps) so cosine decay can run.

        ``total_steps`` is the total number of optimizer steps across the whole run
        (steps/epoch × epochs, after any ``max_steps`` cap). Without this the LR holds flat
        at ``base_lr`` after warmup.
        """
        self.total_steps = int(total_steps)

    def _set_lr(self):
        scale = lr_scale(self.global_step, self.warmup_steps, self.total_steps,
                         self.min_lr_ratio)
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

    def _logit_adjust_log_prior(self, batch: dict, B: int, N: int, device):
        """Per-(B·N) natural class log-prior rows ``(soc, tsk)`` or ``(None, None)``.

        Each session's domain (``batch["domain"]`` — ``"cc"``/``"cr"``) selects its pre-scaled prior
        row ``τ_d·log π_c^nat``; the per-session ``(B, C)`` rows are expanded b-major/n-minor to match
        the ``(B·N, …)`` reshape the categorical loss uses. ``masked_ce`` turns these into the
        ``−α·(τ_d·log π)`` margin. Unknown/missing domains (and τ_d=0 domains) get a zero row (== plain
        CE for that session, since the margin then vanishes).
        """
        if not self._la_active or self._la_social is None:
            return None, None
        dom = batch.get("domain") or [None] * B
        idx = torch.tensor([self._la_dom.get(d, -1) for d in dom],
                           device=device, dtype=torch.long)
        soc = torch.zeros(B, self._la_social.shape[1], device=device)
        tsk = torch.zeros(B, self._la_task.shape[1], device=device)
        known = idx >= 0
        if known.any():
            soc[known] = self._la_social[idx[known]]          # τ_d·log π_c^nat; α applied in masked_ce
            tsk[known] = self._la_task[idx[known]]
        return soc.repeat_interleave(N, dim=0), tsk.repeat_interleave(N, dim=0)

    def _loss_for(self, batch: dict, out: dict):
        """Route loss by the batch's (homogeneous) label kind.

        Returns ``(loss, parts)`` where ``parts`` is a flat dict of granular log values: scalar
        loss components + valid-token counts under bare keys, and (on the histogram cadence) the
        PInSoRo pred/GT per-class count histograms under ``hist/...`` keys.

        Outputs/masks carry a node axis ``(B, N, …)``; loss is reduced only over scored target
        nodes — continuous masks are gated by ``is_target``, categorical relies on the ``-1`` pad
        (non-target/padded nodes have no labels) which ``masked_ce`` already ignores.
        """
        parts: dict = {}
        if batch["label_kind"][0] == "continuous":
            vm = batch["valid_mask"] * batch["is_target"].unsqueeze(-1)   # (B, N, K)
            if "v_pred" in out:     # diffusion: hybrid velocity-prediction MSE + CCC on recovered ŷ₀
                gid = self._dataset_group_ids(batch, out["v_pred"])
                loss, dparts = self.diffusion_loss(
                    out["v_pred"], out["v_target"], out["y_t"],
                    batch["target_chunk"], out["alpha_bar_t"], vm,
                    y_mean=out["y_mean"], y_std=out["y_std"], group_id=gid,
                    horizon_weight=self.horizon_weight, return_parts=True)
                parts["loss_cont_v_mse"] = float(dparts["v_mse"])
                # Log the CCC *loss term* (1 − CCC), matching how it enters L = v_mse + λ·(1 − CCC).
                parts["loss_cont_ccc"] = float(1.0 - dparts["ccc"])
                return loss, parts
            gid = self._dataset_group_ids(batch, out["pred"])
            loss, rparts = self.regression_loss(
                out["pred"], batch["target_chunk"], vm, group_id=gid,
                horizon_weight=self.horizon_weight, return_parts=True)
            parts["loss_cont_mse"] = float(rparts["mse"])
            parts["loss_cont_ccc"] = float(1.0 - rparts["ccc"])
            return loss, parts
        sl, tl = out["social_logits"], out["task_logits"]
        B, N, K, Cs = sl.shape
        Ct = tl.shape[-1]
        # Subsample the horizon to 1 Hz (every pred_stride-th frame) before scoring: the head is
        # trained at macro-granularity and forward-filled back to 25 fps at eval (see ensemble.py).
        st = self.pred_stride
        soc_logits = sl[:, :, ::st, :].reshape(B * N, -1, Cs)
        tsk_logits = tl[:, :, ::st, :].reshape(B * N, -1, Ct)
        soc_tgt = batch["target_chunk"]["social"][:, :, ::st].reshape(B * N, -1)
        tsk_tgt = batch["target_chunk"]["task"][:, :, ::st].reshape(B * N, -1)
        # Per-domain τ-scaled log-prior rows (None when both domains' τ=0); masked_ce turns them into
        # the power-sampler-harmonized margin −α·(τ_d·log π_c^nat) before the softmax.
        soc_lp, tsk_lp = self._logit_adjust_log_prior(batch, B, N, sl.device)
        soc_f = masked_ce(soc_logits, soc_tgt, log_prior=soc_lp,
                          sampler_alpha=self.sampler_alpha)
        tsk_f = masked_ce(tsk_logits, tsk_tgt, log_prior=tsk_lp,
                          sampler_alpha=self.sampler_alpha)
        loss = soc_f + tsk_f
        parts["loss_cat_social"] = float(soc_f.detach())
        parts["loss_cat_task"] = float(tsk_f.detach())
        parts["cat_valid_social"] = float((soc_tgt != -1).sum().item())
        parts["cat_valid_task"] = float((tsk_tgt != -1).sum().item())
        # Pred/GT class histograms (collapse detection) — only on rank0, on the cadence, to avoid
        # a per-step GPU→CPU sync.
        if self.rank0 and (self.global_step + 1) % self.hist_every == 0:
            parts["hist/pinsoro_pred_social"] = _class_count_hist(_valid_class_array(soc_logits, soc_tgt), Cs)
            parts["hist/pinsoro_gt_social"] = _class_count_hist(_valid_target_array(soc_tgt), Cs)
            parts["hist/pinsoro_pred_task"] = _class_count_hist(_valid_class_array(tsk_logits, tsk_tgt), Ct)
            parts["hist/pinsoro_gt_task"] = _class_count_hist(_valid_target_array(tsk_tgt), Ct)
        return loss, parts

    def _do_step(self, loss: torch.Tensor, log_extras: dict | None = None,
                 hist_extras: dict | None = None) -> float:
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
        if self.rank0 and hist_extras:
            self.logger.log_histograms(hist_extras, step=self.global_step)
        if self.rank0 and self.global_step % 20 == 0:
            lr = self.opt.param_groups[0]["lr"]
            extras_str = ""
            if log_extras:
                parts = [f"{k.split('/')[-1]}={v:.4f}" for k, v in log_extras.items()]
                extras_str = "  " + "  ".join(parts)
            print(f"  step {self.global_step:>6}  loss={loss.item():.5f}  "
                  f"gnorm={float(grad_norm):.3f}  lr={lr:.2e}{extras_str}", flush=True)
        return loss.item()

    @staticmethod
    def _split_parts(parts: dict):
        """Split a granular-parts dict into (train/-prefixed scalars, bare-keyed hist arrays)."""
        scalars = {f"train/{k}": v for k, v in parts.items() if not k.startswith("hist/")}
        hists = {k[len("hist/"):]: v for k, v in parts.items() if k.startswith("hist/")}
        return scalars, hists

    def _step(self, batch: dict) -> float:
        """Single-task step (one homogeneous batch)."""
        batch = move_to_device(batch, self.device)
        self._set_lr()
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=self.amp):
            loss, parts = self._loss_for(batch, self.model(batch))
        scalars, hists = self._split_parts(parts)
        return self._do_step(loss, log_extras=scalars, hist_extras=hists)

    def _joint_step(self, batches: list) -> float:
        """Joint multi-task step: forward all task batches, average losses, one backward.

        The shared backbone receives gradients from all task loss surfaces before the weight update.
        Each individual batch remains homogeneous (routing inside _loss_for still works); we
        accumulate into one loss tensor and backward once.
        """
        batches = [move_to_device(b, self.device) for b in batches]
        self._set_lr()
        self.opt.zero_grad(set_to_none=True)
        merged: dict = {}
        with torch.autocast("cuda", enabled=self.amp):
            task_losses = []
            for b in batches:
                l, parts = self._loss_for(b, self.model(b))
                task_losses.append(l)
                merged.update(parts)
            loss = sum(task_losses) / len(task_losses)
        for b, tl in zip(batches, task_losses):
            merged[f"loss_{b['label_kind'][0]}"] = tl.item()
        scalars, hists = self._split_parts(merged)
        return self._do_step(loss, log_extras=scalars, hist_extras=hists)

    def train_epoch(self, loaders, max_steps=None) -> float:
        """Joint multi-task epoch.

        Single loader -> original per-batch step loop (no change).
        Multiple loaders -> zip them; shorter one is cycled so every step pulls one
        batch from each task type. Epoch length = longest loader.
        """
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        self.model.train()

        if len(loaders) == 1:
            it = iter(loaders[0])
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

        # Multi-task: cycle all but the longest so every step sees all task types.
        # Use _endless (not itertools.cycle) for the shorter loaders — cycle would pin their IPC
        # tensors in /dev/shm.
        steps = max(len(ldr) for ldr in loaders)
        iters = [_endless(ldr) if len(ldr) < steps else iter(ldr) for ldr in loaders]
        total, n = 0.0, 0
        t_epoch_start = time.time()
        for _ in range(steps):
            if max_steps is not None and n >= max_steps:
                break
            batches = [next(it) for it in iters]
            if n == 0 and self.rank0:
                print(f"[train] first batch ready in {time.time() - t_epoch_start:.1f}s  "
                      f"(worker spawn + cold session load)", flush=True)
            total += self._joint_step(batches)
            n += 1
        return total / max(n, 1)
