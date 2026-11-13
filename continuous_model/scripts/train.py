#!/usr/bin/env python3
"""torchrun entry point — training.

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
        continuous_model/scripts/train.py --config continuous_model/configs/default.yaml

Single-process (plain `python ...`) also works.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from continuous_model.config import load_config, apply_overrides
from continuous_model.data import registry
from continuous_model.data.build import build_loader
from continuous_model.data.normalize import Normalizer, empty as empty_normalizer
from continuous_model.eval.validate import evaluate_continuous
from continuous_model.models.engagement_model import EngagementModel
from continuous_model.training import distributed as dist_utils
from continuous_model.training.trainer import Trainer, seed_everything
from continuous_model.training.wandb_logger import WandbLogger


def _validate(model, cfg, normalizer, device, cont_datasets, rank0, logger, step):
    """Run validation on rank0 and log per-domain CCC plus ``val/ccc_mean``.

    Returns the full ``log`` dict (so callers can select a checkpoint by any individual metric);
    returns ``{}`` off rank0.
    """
    if not rank0:
        return {}
    log = {}
    ccc_by_dom = evaluate_continuous(model, cfg, normalizer, device, cont_datasets, split="val")
    for dom, v in ccc_by_dom.items():
        log[f"val/ccc_{dom}"] = v
    if ccc_by_dom:
        log["val/ccc_mean"] = sum(ccc_by_dom.values()) / len(ccc_by_dom)
        print("  val CCC: " +
              "  ".join(f"{d}={v:.4f}" for d, v in sorted(ccc_by_dom.items())) +
              f"   mean={log['val/ccc_mean']:.4f}")
    if log:
        logger.log(log, step=step)
    return log


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def make_run_name(cfg: dict, git: str) -> str:
    """Run identity shared by the W&B run and the on-disk checkpoint dir.

    Built from the monotonic run counter + ``wandb.run_name_pattern`` (same source the legacy
    WandbLogger used). Called once on rank0 so the counter is bumped exactly once per run; the
    name is then passed to both the logger and the ``checkpoints/<run_name>/`` directory.
    """
    from datetime import date

    from continuous_model.training.wandb_logger import _next_run_id
    pattern = (cfg.get("wandb", {}) or {}).get("run_name_pattern", "run{run_id:03d}-{date}")
    return pattern.format(run_id=_next_run_id(),
                          date=date.today().strftime("%Y%m%d"),
                          git=(git[:7] or "nogit"))


def _log(rank0: bool, msg: str, t0: float | None = None) -> None:
    if not rank0:
        return
    suffix = f"  (+{time.time() - t0:.1f}s)" if t0 is not None else ""
    print(f"[init] {msg}{suffix}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="continuous_model/configs/default.yaml")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap steps/epoch (smoke runs)")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override global train batch size (smoke runs)")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="override dataloader workers (use 0 for light smoke runs)")
    ap.add_argument("--max_sessions", type=int, default=None,
                    help="cap sessions per dataset (smoke runs: keeps cache warm, avoids disk thrash)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.epochs")
    ap.add_argument("--resume", default=None,
                    help="resume a previous run from a checkpoint (.pt): restores weights + "
                         "global_step (+ optimizer/scaler if present) and continues the epoch loop "
                         "and LR schedule into the same run dir. Pass the same --config/"
                         "--epochs/--batch_size as the original; only dataloader-perf settings "
                         "(num_workers, prefetch_factor) are safe to change. Use "
                         "checkpoints/<run>/resume.pt for a faithful resume, or a lean "
                         "epoch_<NNN>.pt to resume with a fresh optimizer.")
    ap.add_argument("--run_name", default=None,
                    help="pin the W&B run name + checkpoints/<run_name>/ dir (else auto-incrementing "
                         "run counter)")
    ap.add_argument("--val_every", type=int, default=None,
                    help="override train.val_every_epochs (finer cadence → finer best-epoch selection)")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted config override, repeatable (e.g. --set continuous_loss.mse_weight=0 "
                         "--set train.ema_decay=0.999). Values are parsed as YAML (true/false/ints/floats).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, {"train.max_steps": args.max_steps,
                          "train.epochs": args.epochs,
                          "train.val_every_epochs": args.val_every,
                          "batch_size.train": args.batch_size,
                          "data.num_workers": args.num_workers})
    if args.overrides:
        import yaml as _yaml
        # Parse each --set KEY=VALUE, coercing VALUE via YAML so booleans/numbers aren't left as strings.
        apply_overrides(cfg, {k: _yaml.safe_load(v)
                              for k, v in (o.split("=", 1) for o in args.overrides)})
    if args.no_wandb:
        cfg["wandb"]["enabled"] = False

    train_run(cfg, max_sessions=args.max_sessions, no_wandb=args.no_wandb,
              resume_ckpt=args.resume, run_name_override=args.run_name)


def train_run(cfg, *, max_sessions=None,
              no_wandb=False, wandb_run=None, run_name_override=None, resume_ckpt=None):
    """Run one training job from a fully-resolved config.

    When ``wandb_run`` is given (an already-open W&B run) the logger attaches to it rather than
    starting a second run, and the run/checkpoint identity is taken from its name.

    ``run_name_override`` pins the W&B run name and the ``checkpoints/<run_name>/`` directory to a
    fixed string; ``None`` keeps the auto-incrementing run counter.

    The run keeps a single ``best.pt`` holding the checkpoint with the highest value of
    ``cfg["train"]["select_metric"]`` (default ``val/ccc_mean``) across all validation points.
    """
    t_start = time.time()

    info = dist_utils.setup()
    rank0 = dist_utils.is_rank0(info)
    seed_everything(cfg["seed"] + info["rank"])

    _log(rank0, f"DDP ready  world={info['world_size']}  "
         f"rank={info['rank']}  device={info['device']}", t_start)

    # Normalization stats (train-only). Falls back to no-op if not yet fit.
    stats_path = cfg["data"]["norm_stats"]
    normalizer = (Normalizer.load(stats_path)
                  if os.path.exists(stats_path) else empty_normalizer())
    if rank0 and not os.path.exists(stats_path):
        print(f"[norm] {stats_path} missing — using identity normalization "
              f"(run fit_norm_stats.py).")

    train_datasets = list(cfg["data"]["train_datasets"])
    _log(rank0, f"building dataset index  ({train_datasets})", t_start)
    t_loaders = time.time()
    train_splits = list(cfg["data"].get("train_splits", ["train"]))
    _, train_loader = build_loader(
        cfg, train_datasets, train_splits, train=True, normalizer=normalizer,
        world_size=info["world_size"], rank=info["rank"], seed=cfg["seed"],
        max_sessions=max_sessions)
    if train_loader is None:
        print("No training records found — check dataset roots.")
        dist_utils.cleanup()
        return
    _log(rank0, f"loader ready  {len(train_loader.dataset.records)} sessions  "
         f"{len(train_loader.dataset.windows)} windows  "
         f"{len(train_loader)} steps/epoch/rank", t_loaders)

    t_model = time.time()
    _log(rank0, "building model ...")
    model = EngagementModel(cfg, modality_dims=normalizer.effective_dims).to(info["device"])
    _log(rank0, "model on device", t_model)

    # Resume: load a previous run's weights here (before compile/DDP). global_step,
    # optimizer, scaler and the start epoch are restored after the Trainer is built (below). The LR
    # schedule resumes exactly because it is a pure function of the restored global_step. A lean
    # epoch_<NNN>.pt (no optimizer/scaler) resumes with a fresh Adam/scaler — moments re-warm in a
    # few steps; resume.pt carries the full state for a faithful resume.
    resume_state, start_epoch = None, 0
    if resume_ckpt:
        resume_state = torch.load(resume_ckpt, map_location=info["device"], weights_only=False)
        state = resume_state.get("model", resume_state) if isinstance(resume_state, dict) \
            else resume_state
        model.load_state_dict(state, strict=True)
        if rank0:
            print(f"[resume] loaded weights from {resume_ckpt} "
                  f"(epoch {resume_state.get('epoch', '?')}, step {resume_state.get('step', '?')})")

    # torch.compile the trunk forward in place (opt-in). model.compile() rewrites forward without
    # wrapping the module, so state_dict keys stay unprefixed (checkpoint save/load unchanged) and
    # it composes with the DDP wrap below. Automatic-dynamic shapes absorb the variable node count
    # and last-batch B: expect one recompile per new shape, then it reuses the graph — the first
    # few steps pay a one-time compile cost.
    if bool(cfg["model"].get("compile", False)) and info["device"].type == "cuda":
        mode = str(cfg["model"].get("compile_mode", "default"))
        _log(rank0, f"torch.compile(mode={mode}) — first steps compile (one-time cost) ...")
        model.compile(mode=mode)

    if info["world_size"] > 1:
        model = DDP(model, device_ids=[info["local_rank"]]
                    if info["device"].type == "cuda" else None,
                    static_graph=True)

    # Run identity (shared by the W&B run + the checkpoint dir). If a run is already open, reuse
    # its name (and attach the logger to it) rather than bumping the counter.
    if wandb_run is not None:
        run_name = wandb_run.name if rank0 else None
        logger = WandbLogger.attach(wandb_run, rank0)
    else:
        if rank0:
            # On resume, continue into the original run's dir (so its epoch_<NNN>.pt keep accruing)
            # unless an explicit name override is given.
            run_name = (run_name_override
                        or (resume_state.get("run_name") if resume_state else None)
                        or make_run_name(cfg, git_hash()))
        else:
            run_name = None
        logger = WandbLogger(cfg, rank0, git_hash(), run_name=run_name) if not no_wandb else \
            WandbLogger({"wandb": {"enabled": False}}, rank0, run_name=run_name)

    inner = model.module if isinstance(model, DDP) else model
    trainer = Trainer(inner, cfg, info["device"], logger, rank0=rank0)
    # Keep DDP grad sync by stepping the wrapped module's params via the wrapper.
    if isinstance(model, DDP):
        trainer.model = model

    # Total optimizer steps for the cosine LR decay (same on every rank: global_step is the
    # per-rank step count, and len(loader) is already per-rank under DistributedSampler).
    steps_per_epoch = len(train_loader)
    max_steps = cfg["train"].get("max_steps") or steps_per_epoch
    eff_steps_per_epoch = min(steps_per_epoch, max_steps) // trainer.grad_accum
    trainer.configure_schedule(eff_steps_per_epoch * cfg["train"]["epochs"],
                               steps_per_epoch=eff_steps_per_epoch)

    # Resume: restore the step counter (drives the LR schedule), optimizer/scaler if present, and the
    # epoch to continue from. After Trainer build so trainer.opt/scaler exist.
    if resume_state is not None:
        trainer.global_step = int(resume_state.get("step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        if "optimizer" in resume_state:
            trainer.opt.load_state_dict(resume_state["optimizer"])
        elif rank0:
            print("[resume] checkpoint has no optimizer state — Adam starts fresh "
                  "(moments re-warm in a few steps)")
        if "scaler" in resume_state:
            trainer.scaler.load_state_dict(resume_state["scaler"])
        if "model_ema" in resume_state:
            trainer.ema.load_state_dict(resume_state["model_ema"])
        if rank0:
            print(f"[resume] continuing from epoch {start_epoch + 1}/{cfg['train']['epochs']}, "
                  f"global_step {trainer.global_step}")

    if rank0:
        n_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        nw = cfg["data"].get("num_workers", 0)
        print(f"\n{'='*60}")
        print(f"  model params : {n_params/1e6:.1f}M")
        print(f"  world size   : {info['world_size']}  device: {info['device']}")
        print(f"  epochs       : {cfg['train']['epochs']}  steps/epoch: {min(max_steps, steps_per_epoch)}")
        print(f"  batch        : {cfg['batch_size']['train']} sessions/step (global)")
        print(f"  window       : W={cfg['window']['W']} S={cfg['window']['S']} K={cfg['window']['K']}")
        print(f"  num_workers  : {nw}  cache_roles: {registry.MAX_PARTNERS + 1}")
        print(f"  amp          : {cfg['train'].get('amp', True)}  grad_clip: {cfg.get('grad_clip_norm', 1.0)}")
        print(f"  setup time   : {time.time() - t_start:.1f}s")
        print(f"{'='*60}\n")

    nw = int(cfg["data"].get("num_workers", 0))
    epochs = cfg["train"]["epochs"]
    # One directory per run; every validation-interval checkpoint is kept here (no best/last).
    run_dir = os.path.join("checkpoints", run_name) if rank0 else None
    if rank0:
        os.makedirs(run_dir, exist_ok=True)
        print(f"[ckpt] saving checkpoints to {run_dir}/")
    # Validation datasets are an independent axis from training: cfg["data"]["val_datasets"] lets a
    # dataset (e.g. mpii) be scored without being trained on (evaluate_continuous builds its own
    # val-split loader per dataset, so it doesn't need to be in train_datasets/train_splits).
    val_datasets_cfg = list(cfg["data"].get("val_datasets", train_datasets))
    cont_val_datasets = [d for d in val_datasets_cfg if registry.label_kind(d) == "continuous"]
    best_ccc = None      # running max of val/ccc_mean
    # Checkpoint selection: keep a single best.pt at the highest value of ``select_metric``.
    select_metric = cfg["train"].get("select_metric") or "val/ccc_mean"
    best_select = None   # running max of select_metric

    for ep in range(start_epoch, epochs):
        samp = getattr(train_loader, "sampler", None)
        if hasattr(samp, "set_epoch"):
            samp.set_epoch(ep)
        if rank0:
            # Workers persist across epochs (persistent_workers=True); the only per-epoch cost is a
            # cold role-cache reload — set_epoch reshuffles window order, so the first windows
            # point at sessions not in the workers' LRU cache (cache_roles). Not a re-spawn.
            verb = "spawning" if ep == 0 else "reusing"
            print(f"[train] epoch {ep+1}/{epochs} — {verb} {nw} persistent workers, warming session cache (~15–30s) ...", flush=True)
        t0 = time.time()
        avg = trainer.train_epoch(train_loader, max_steps=cfg["train"]["max_steps"])
        elapsed = time.time() - t0
        if rank0:
            logger.log({"train/epoch_loss": avg, "epoch": ep},
                       step=trainer.global_step)
            print(f"epoch {ep+1}/{epochs}  loss={avg:.5f}  "
                  f"steps={trainer.global_step}  time={elapsed:.0f}s")

        # ── checkpoint save (every save_every_epochs) + validation/best (every val_every_epochs) ──
        if rank0:
            save_every = max(1, int(cfg["train"].get("save_every_epochs", 1)))
            val_every = int(cfg["train"].get("val_every_epochs", 10))
            is_last = (ep + 1) == epochs
            # Eval/submit checkpoints carry the EMA weights (== the weights validated below); a run
            # without EMA falls through to the live weights. resume.pt keeps the *live* weights (plus
            # optimizer/scaler/EMA shadow) for a faithful continuation.
            ckpt = {"model": trainer.ema.merged_state_dict(inner), "config": cfg,
                    "epoch": ep + 1, "step": trainer.global_step,
                    "git_hash": git_hash(), "run_name": run_name}
            if (ep + 1) % save_every == 0 or is_last:
                path = os.path.join(run_dir, f"epoch_{ep + 1:03d}.pt")
                torch.save(ckpt, path)
                print(f"  saved checkpoint -> {path}")
                # Rolling full-state checkpoint for --resume (adds optimizer + scaler + EMA shadow).
                # Overwrites each interval; the lean epoch_<NNN>.pt above stays weights-only.
                torch.save({**ckpt, "model": inner.state_dict(),
                            "optimizer": trainer.opt.state_dict(),
                            "scaler": trainer.scaler.state_dict(),
                            "model_ema": trainer.ema.state_dict()},
                           os.path.join(run_dir, "resume.pt"))
            if (ep + 1) % val_every == 0 or is_last:
                # Validate on the EMA weights so best-checkpoint selection matches what is shipped.
                with trainer.ema.applied_to(inner):
                    val_log = _validate(inner, cfg, normalizer, info["device"],
                                        cont_val_datasets, rank0, logger, trainer.global_step)
                ccc_mean = val_log.get("val/ccc_mean")
                if ccc_mean is not None:
                    best_ccc = ccc_mean if best_ccc is None else max(best_ccc, ccc_mean)
                    logger.log({"val/ccc_best": best_ccc}, step=trainer.global_step)
                # Keep best.pt at the highest value of select_metric.
                cur = val_log.get(select_metric)
                if cur is not None and (best_select is None or cur > best_select):
                    best_select = cur
                    best_path = os.path.join(run_dir, "best.pt")
                    torch.save({**ckpt, "select_metric": select_metric, "select_value": cur},
                               best_path)
                    logger.log({"val/best": best_select}, step=trainer.global_step)
                    print(f"  new best {select_metric}={cur:.4f} -> {best_path}")

    if rank0:
        print("training done")
        logger.finish()
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
