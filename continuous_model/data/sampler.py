"""Window sampler.

``SessionGroupedSampler``: shuffle *sessions*, then yield all windows from each session
consecutively to maximise per-worker LRU cache hits.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class SessionGroupedSampler(Sampler):
    """Yield window indices grouped by session to maximise per-worker LRU cache hits.

    Args:
        dataset: an ``EngagementDataset`` whose ``.windows`` list is ``[(rec_idx, start), …]``.
        shuffle: shuffle session order (and windows within each session) each epoch.
            Set ``False`` for val/test — stable ordering, no randomness.
        seed: base random seed; actual seed = ``seed + epoch`` so each epoch is different.
        num_replicas: DDP world size (1 = single-process).
        rank: DDP rank of this process.
    """

    def __init__(self, dataset, shuffle: bool = True, seed: int = 40,
                 num_replicas: int = 1, rank: int = 0):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        # Group window indices by session (rec_idx preserves insertion order).
        groups: dict[int, list[int]] = {}
        for win_idx, (rec_idx, _) in enumerate(dataset.windows):
            groups.setdefault(rec_idx, []).append(win_idx)
        self.groups: list[list[int]] = [groups[k] for k in sorted(groups)]

        # Pre-compute per-rank length (padded so all ranks have the same count).
        total = sum(len(g) for g in self.groups)
        padded = math.ceil(total / num_replicas) * num_replicas
        self._len = padded // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            session_order = torch.randperm(len(self.groups), generator=g).tolist()
        else:
            session_order = list(range(len(self.groups)))

        # Build the full ordered index list: consecutive windows per session.
        indices: list[int] = []
        for s in session_order:
            wins = list(self.groups[s])
            if self.shuffle:
                perm = torch.randperm(len(wins), generator=g).tolist()
                wins = [wins[i] for i in perm]
            indices.extend(wins)

        # Pad with wrap-around so all ranks get the same number of indices.
        total = len(indices)
        padded = self._len * self.num_replicas
        if padded > total:
            indices = indices + indices[: padded - total]

        # Give this rank a contiguous block — preserves session locality within rank.
        start = self.rank * self._len
        return iter(indices[start: start + self._len])
