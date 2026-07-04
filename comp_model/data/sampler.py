"""Window samplers.

``SessionGroupedSampler`` (continuous datasets + all validation): shuffle *sessions*, then yield all
windows from each session consecutively to maximise per-worker LRU cache hits. ``ClassBalancedSampler``
(categorical/PInSoRo training): a randomized, **power-balanced** draw — each slot picks a class
weighted by a square-root power law over the whole PInSoRo pool.
"""

from __future__ import annotations

import math

import numpy as np
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


def _build_class_pool(dataset, axis: str, num_classes: int) -> list[dict[int, list[int]]]:
    """Index windows by the classes present in their non-overlapping horizon slice, for one axis.

    Returns ``pool`` where ``pool[c]`` maps ``rec_idx -> [window_idx, …]`` for every window whose
    membership slice contains class ``c`` on ``axis`` (``"social"`` | ``"task"``). The slice is the
    leading ``S`` frames of the target chunk (consecutive chunks overlap by ``K - S`` frames); for
    the default ``W=96, S=32, K=64`` that is window-local offsets ``[32, 64)``. Membership uses every
    valid (``>= 0``) frame of the slice, so a window with several classes appears in several buckets.
    Labels come from the dataset's own loader (annotation CSVs, already resampled to 25 fps).
    """
    pool: list[dict[int, list[int]]] = [dict() for _ in range(num_classes)]
    W, K, S = dataset.W, dataset.K, dataset.S
    chunk_off = W - K                 # target chunk = arr[start+chunk_off : start+W]
    seg_len = min(S, K)               # leading slice of the chunk not shared with the next window
    # windows are enumerated record-by-record, so one record's labels are loaded once and dropped
    # when the record changes.
    cur_rec, cur_arrs = -1, []
    for win_idx, (rec_idx, start) in enumerate(dataset.windows):
        if rec_idx != cur_rec:
            rec = dataset.records[rec_idx]
            T = dataset._T[rec_idx]
            cur_arrs = [dataset._load_role_labels(rec.ref, role, T)[axis]
                        for role in rec.target_roles]
            cur_rec = rec_idx
        cs = start + chunk_off
        present: set[int] = set()
        for arr in cur_arrs:
            seg = arr[cs:cs + seg_len]
            for c in np.unique(seg):
                if c >= 0:
                    present.add(int(c))
        for c in present:
            pool[c].setdefault(rec_idx, []).append(win_idx)
    return pool


class ClassBalancedSampler(Sampler):
    """Power-balanced window sampler for the categorical (PInSoRo) loader.

    Decouples batch composition from the natural class + session frequencies via a three-stage draw
    per slot, run **in parallel on both label axes** (social 5-class, task 4-class) since each frame
    carries a social *and* a task label:

      1. **Power-balance the class.** Pick a class with probability ``∝ f_c^(1-α)`` over the classes
         present in the PInSoRo pool (``f_c`` = the class's total window count across *all* sessions,
         cc and cr together). With ``α = 0.5`` this is square-root balancing (``α = 1`` is strict
         uniform, ``α = 0`` natural frequencies).
      2. **Stratify by session uniformly.** Among the sessions holding ``>= 1`` window of the class,
         pick one uniformly — ignoring how many such windows it holds.
      3. **Select the window.** Pick one of that session's windows containing the class uniformly.

    The class *marginal* stays ``∝ f_c^(1-α)`` (set by stage 1); stages 2+3 only redistribute the
    within-class mass evenly across sessions. The cc/cr interaction domain is not stratified — the
    domain mix follows the data; domain-level calibration is handled by the train-time
    logit-adjustment margin (``losses.masked_ce``).

    A window "contains" a class iff it appears in the window's non-overlapping horizon slice (the
    leading ``S`` frames of the target chunk); see ``_build_class_pool``.

    The social- and task-targeted draws are interleaved across slots, so the emitted index stream
    is balanced on both the social and task class marginals at once. Sampling is with replacement;
    ``__len__`` equals the per-rank window count so epoch length matches the grouped sampler.

    NB: this draws windows from random sessions, so it forfeits the per-worker session-cache locality.

    Args:
        dataset: an ``EngagementDataset`` over categorical (PInSoRo) records.
        num_social, num_task: class counts for the two axes (``registry.PINSORO_*_CLASSES``).
        alpha: power-law balancing exponent on ``1/f_c`` (``0.5`` = square-root, ``1`` = strict
            uniform, ``0`` = natural frequencies).
        seed: base RNG seed (actual seed mixes in ``epoch`` and ``rank`` so draws differ).
        num_replicas: DDP world size; rank: DDP rank of this process.
    """

    def __init__(self, dataset, num_social: int, num_task: int, alpha: float = 0.5,
                 seed: int = 40, num_replicas: int = 1, rank: int = 0):
        super().__init__()
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.alpha = float(alpha)

        # _axes[axis] -> (classes, sessions, probs): classes present in the pool, a per-class list of
        # session window-index arrays (for the uniform session→window draw), and the power-balanced
        # class-selection probabilities ∝ f_c^(1-α) (f_c = total windows of the class, all sessions).
        self._axes = []
        for axis, n in (("social", num_social), ("task", num_task)):
            pool = _build_class_pool(dataset, axis, n)
            classes, sessions, probs = [], [], []
            for c, rec_map in enumerate(pool):
                sess_wins = [np.asarray(ws, dtype=np.int64) for ws in rec_map.values()]
                if sess_wins:
                    classes.append(c)
                    sessions.append(sess_wins)
                    f_c = sum(len(w) for w in sess_wins)
                    probs.append(f_c ** (1.0 - self.alpha))
            p = np.asarray(probs, dtype=np.float64)
            self._axes.append((classes, sessions, p / p.sum() if p.size else p))

        total = len(dataset.windows)
        padded = math.ceil(total / num_replicas) * num_replicas
        self._len = padded // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        # rank in the seed so each rank draws an independent (with-replacement) stream.
        g = np.random.default_rng(self.seed + self.epoch * 100003 + self.rank)
        indices: list[int] = []
        for i in range(self._len):
            classes, sessions, probs = self._axes[i % len(self._axes)]   # alternate social/task
            if not classes:
                continue
            ci = g.choice(len(classes), p=probs)                         # 1: power-balanced class
            sess_wins = sessions[ci]
            wins = sess_wins[g.integers(len(sess_wins))]                 # 2: uniform session
            indices.append(int(wins[g.integers(len(wins))]))            # 3: uniform window
        return iter(indices)
