"""Build the flat sample index from the registry.

A "sample record" is one **session** (all present roles), not one target. The PyTorch Dataset
turns each record into one window per slide step; every window carries *all* nodes in the session
so the model can predict engagement for everyone in the frame from a single shared trunk pass.

``node_roles`` are every role present in the session (PInSoRo ``env``/robot included as a node).
``target_roles`` is the scored subset (``registry.target_roles``): NoXi expert+novice, PInSoRo
purple+yellow, MPII all detected subjects. Nodes that are not targets (e.g. the PInSoRo robot)
still participate in graph cross-attention but carry no loss/metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import registry


@dataclass(frozen=True)
class SampleRecord:
    ref: registry.SessionRef
    node_roles: tuple        # every role present in the session (graph nodes)
    target_roles: tuple      # scored subset of node_roles

    @property
    def dataset(self) -> str:
        return self.ref.dataset

    @property
    def domain(self) -> str:
        """Per-domain reporting + conditioning key.

        PInSoRo resolves to its interaction type (``"cc"``/``"cr"``) so the train-time
        logit-adjustment margin and per-domain kappa key on the right natural prior; every other
        dataset reports at dataset granularity (language-level NoXi splits are not encoded on disk).
        """
        if self.ref.dataset == "pinsoro":
            return registry.pinsoro_domain(self.ref.feature_dir) or self.ref.dataset
        return self.ref.dataset


def build_index(roots: dict, datasets, splits) -> list:
    """Enumerate one (session) record per dataset/split.

    Args:
        roots: {dataset: root_path} from config.
        datasets: iterable of dataset keys to include.
        splits: iterable of logical split names to include (filtered to those
            that exist for each dataset).
    """
    records: list = []
    splits = list(splits)
    for dataset in datasets:
        root = roots[dataset]
        valid_splits = [s for s in splits if s in registry.splits_for(dataset)]
        for split in valid_splits:
            for ref in registry.iter_sessions(dataset, root, split):
                targets = registry.target_roles(dataset, ref.roles, ref.feature_dir)
                if not targets:
                    continue
                records.append(SampleRecord(ref, tuple(ref.roles), tuple(targets)))
    return records
