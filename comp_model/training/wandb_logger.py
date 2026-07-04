"""Weights & Biases logging (rank-0 only). W&B is the tracker — no JSONL fallback.

Activates when `wandb.enabled` is true AND credentials exist — either `WANDB_API_KEY`
in the env or a `wandb login` entry in ~/.netrc. If W&B is unavailable the logger
no-ops (prints to stdout) so smoke runs don't block on auth.
"""

from __future__ import annotations

import os
from datetime import date

# Monotonic run counter file.
_DEFAULT_COUNTER_PATH = "comp_model/configs/run_counter.txt"


def _next_run_id(counter_path: str = _DEFAULT_COUNTER_PATH) -> int:
    """Read, increment, and persist a monotonic run counter. Thread-safe for single-node use."""
    os.makedirs(os.path.dirname(counter_path) or ".", exist_ok=True)
    try:
        with open(counter_path) as f:
            n = int(f.read().strip())
    except (OSError, ValueError):
        n = 0
    n += 1
    with open(counter_path, "w") as f:
        f.write(str(n))
    return n


def _has_credentials() -> bool:
    """True if wandb can authenticate: env key or a ~/.netrc login entry."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        with open(os.path.join(os.path.expanduser("~"), ".netrc")) as f:
            return "api.wandb.ai" in f.read()
    except OSError:
        return False


class WandbLogger:
    def __init__(self, config: dict, rank0: bool, git_hash: str = "",
                 run_name: str | None = None):
        """``run_name``: pre-computed run name (so the W&B run and the on-disk checkpoint dir share
        one identity). If None, a name is generated here from the run counter + ``run_name_pattern``."""
        self.active = False
        self.run = None
        wcfg = config.get("wandb", {})
        if not rank0 or not wcfg.get("enabled", False):
            return
        if not _has_credentials():
            print("[wandb] no credentials (set WANDB_API_KEY or run `wandb login`) "
                  "— logging to stdout only.")
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] wandb not installed — logging to stdout only.")
            return
        if run_name is None:
            run_name = wcfg.get("run_name_pattern", "run{run_id:03d}-{date}-{git}").format(
                run_id=_next_run_id(),
                date=date.today().strftime("%Y%m%d"),
                git=(git_hash[:7] or "nogit"),
            )
        self.run = wandb.init(
            project=wcfg.get("project", "comp_model"),
            name=run_name, config={**config, "git_hash": git_hash},
        )
        self.active = True
        print(f"[wandb] run: {run_name}")

    @classmethod
    def attach(cls, run, rank0: bool = True) -> "WandbLogger":
        """Wrap an already-open W&B run (e.g. a `wandb agent` sweep trial's run) instead of
        calling `wandb.init()` again. `log`/`finish` then write to the existing run."""
        self = cls.__new__(cls)
        self.active = bool(rank0 and run is not None)
        self.run = run if rank0 else None
        return self

    def log(self, metrics: dict, step: int | None = None):
        if self.active and self.run is not None:
            self.run.log(metrics, step=step)
        else:
            kv = " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in metrics.items())
            print(f"[step {step}] {kv}")

    def log_histograms(self, hists: dict, step: int | None = None):
        """Log per-class count histograms (collapse detection).

        ``hists``: ``{name: (counts, bin_edges)}`` — a precomputed discrete histogram with one bin
        per class (as returned by the trainer's ``_class_count_hist``). Empty histograms (no valid
        samples) are skipped. Keys are prefixed ``train/``. No-ops (prints per-class counts) when
        W&B is inactive."""
        nonempty = {k: v for k, v in hists.items() if v is not None and int(v[0].sum())}
        if not nonempty:
            return
        if self.active and self.run is not None:
            import wandb
            self.run.log(
                {f"train/{k}": wandb.Histogram(np_histogram=(counts.tolist(), edges.tolist()))
                 for k, (counts, edges) in nonempty.items()},
                step=step)
        else:
            counts = {k: v[0].tolist() for k, v in nonempty.items()}
            print(f"[step {step}] hist {counts}")

    def finish(self):
        if self.active and self.run is not None:
            self.run.finish()
