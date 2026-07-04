"""PInSoRo class-count accounting for class-weighted CE + the balanced window sampler.

Counts are taken over the **scored target roles** of the train split only (children: purple/yellow;
never the ``env``/robot partner-only nodes), matching exactly which positions carry a label in the
categorical loss. Raw 30 fps annotation frames are counted (nearest-neighbour resampling to 25 fps
preserves class proportions).
"""

from __future__ import annotations

import os

import numpy as np

from . import registry, streams
from .sessions import build_index


def load_pinsoro_counts(stats_path: str):
    """Load precomputed ``(social_counts[5], task_counts[4])`` from a norm_stats ``.npz``.

    Written by ``fit_norm_stats.py``; returns ``None`` if the file or the count keys are absent
    (e.g. stats fit before this was added), so callers can fall back to ``pinsoro_class_counts``.
    """
    if not stats_path or not os.path.exists(stats_path):
        return None
    stats = np.load(stats_path)
    if "pinsoro_social_counts" not in stats.files:
        return None
    return (stats["pinsoro_social_counts"].astype(np.int64),
            stats["pinsoro_task_counts"].astype(np.int64))


def pinsoro_class_counts(roots: dict, datasets, splits=("train",), domain=None):
    """Return ``(social_counts[5], task_counts[4])`` over target roles of the categorical splits.

    Scans ``{role}.social_engagement.annotation.csv`` / ``{role}.task_engagement.annotation.csv``
    for every scored target role and tallies class indices (unknown ``-1`` dropped). Datasets that
    are not categorical are ignored, so passing the full ``train_datasets`` list is safe. ``domain``
    (``"cc"``/``"cr"``) restricts the tally to one interaction type; ``None`` pools both.
    """
    soc = np.zeros(registry.PINSORO_SOCIAL_CLASSES, dtype=np.int64)
    tsk = np.zeros(registry.PINSORO_TASK_CLASSES, dtype=np.int64)
    cat = [d for d in datasets if registry.label_kind(d) == "categorical"]
    if not cat:
        return soc, tsk
    for rec in build_index(roots, cat, list(splits)):
        ref = rec.ref
        if not ref.label_dir:
            continue
        if domain is not None and registry.pinsoro_domain(ref.feature_dir) != domain:
            continue
        for role in rec.target_roles:
            sp = os.path.join(ref.label_dir, f"{role}.social_engagement.annotation.csv")
            tp = os.path.join(ref.label_dir, f"{role}.task_engagement.annotation.csv")
            if os.path.exists(sp):
                idx = streams.read_str_csv_to_idx(sp, registry.SOCIAL_CLASSES)
                np.add.at(soc, idx[idx >= 0], 1)
            if os.path.exists(tp):
                idx = streams.read_str_csv_to_idx(tp, registry.TASK_CLASSES)
                np.add.at(tsk, idx[idx >= 0], 1)
    return soc, tsk


def pinsoro_class_priors(roots: dict, datasets, domain=None, splits=("train",), eps=1e-12):
    """Natural (train) class base rates per axis, optionally restricted to one ``domain``.

    Returns ``(social_prior[5], task_prior[4])``, each summing to 1, for the train-time logit
    adjustment margin ``f_c − τ_d·α·log π_c^nat``. Classes absent from the requested domain (e.g. cr
    has no ``cooperative``) get ``eps`` mass, so their log-prior is a large negative value → a large
    positive margin that suppresses their raw logit.
    """
    soc, tsk = pinsoro_class_counts(roots, datasets, splits, domain=domain)
    sp = soc.astype(np.float64) + eps
    tp = tsk.astype(np.float64) + eps
    return sp / sp.sum(), tp / tp.sum()
