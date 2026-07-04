"""Build DataLoaders from config (shared by train + validate)."""

from __future__ import annotations

from torch.utils.data import DataLoader

from . import registry
from .collate import make_collate
from .dataset import EngagementDataset
from .normalize import Normalizer
from .sampler import ClassBalancedSampler, SessionGroupedSampler
from .sessions import build_index


def build_loader(config, datasets, splits, train: bool,
                 normalizer: Normalizer | None = None,
                 batch_size: int | None = None,
                 world_size: int = 1, rank: int = 0,
                 seed: int = 40, num_workers: int | None = None,
                 max_sessions: int | None = None):
    """Construct (dataset, loader). Returns (None, None) if no records found.

    Sampler choice: categorical (PInSoRo) *training* uses ``ClassBalancedSampler`` (per-slot
    power-balanced draw on both label axes). Continuous training and all validation use
    ``SessionGroupedSampler``: windows are yielded in session-contiguous blocks so each session
    stays in the per-worker LRU cache while its windows are consumed.
    """
    roots = config["data"]["roots"]
    records = build_index(roots, datasets, splits)
    if max_sessions is not None:
        from collections import defaultdict
        by_ds: dict = defaultdict(list)
        for r in records:
            by_ds[r.dataset].append(r)
        records = [r for ds_recs in by_ds.values() for r in ds_recs[:max_sessions]]
    if not records:
        return None, None

    ds = EngagementDataset(records, config, normalizer=normalizer,
                           train=train, seed=seed)
    if batch_size is None:
        batch_size = config["batch_size"]["train" if train else "val"]
    per_gpu = max(1, batch_size // world_size) if world_size > 1 else batch_size

    if train and registry.label_kind(datasets[0]) == "categorical":
        sampler = ClassBalancedSampler(
            ds, num_social=registry.PINSORO_SOCIAL_CLASSES,
            num_task=registry.PINSORO_TASK_CLASSES,
            alpha=float(config.get("pinsoro", {}).get("sampler_alpha", 0.5)),
            seed=seed, num_replicas=world_size, rank=rank)
    else:
        sampler = SessionGroupedSampler(
            ds, shuffle=train, seed=seed,
            num_replicas=world_size, rank=rank)

    if num_workers is None:
        num_workers = int(config["data"].get("num_workers", 0))

    collate = make_collate(max_partners=registry.MAX_PARTNERS,
                           window_W=config["window"]["W"],
                           modality_dims=normalizer.effective_dims if normalizer else None)
    loader = DataLoader(
        ds, batch_size=per_gpu, sampler=sampler,
        num_workers=num_workers, collate_fn=collate, drop_last=train,
        # pin_memory omitted: pin_memory=True deadlocks with the NCCL backend in DDP.
        persistent_workers=(num_workers > 0),
        prefetch_factor=(int(config["data"].get("prefetch_factor", 2))
                         if num_workers > 0 else None),
    )
    return ds, loader
