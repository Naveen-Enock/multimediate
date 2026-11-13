"""Per-modality train-only normalization.

Stats are fit on the train split ONLY (see scripts/fit_norm_stats.py) and applied
everywhere.

Channel selection:
    fit_norm_stats.py writes ``{m}.keep_channels`` (sorted int64 index array) into the
    .npz alongside mean/std that are already aligned to those kept channels.  Normalizer
    selects the channels BEFORE z-scoring, so the projection layer sees ``D_effective``
    features per modality rather than the raw ``D_m`` from registry.MODALITY_DIMS.

    Structural drops (frame counters, dead channels, …) are declared in
    ``registry.MODALITY_DROP_CHANNELS``; data-dead channels (std=0 across the train
    corpus) are merged in during fitting.  Callers that need the effective dim per
    modality use ``normalizer.effective_dims``.

Normalization mode (registry.MODALITY_NORM_MODE; all modes store a per-channel
center/scale pair so apply() is identical — they differ only at fit time):
    * "z_score"  (default): center = per-dim mean, scale = per-dim std.
    * "global_std" (xlm_roberta): center = per-dim mean, scale = one global scalar
      (= sqrt(mean over dims of per-dim variance)) replicated to (D_eff,).
    * "robust" (egemapsv2): center = per-dim median, scale = per-dim NIQR =
      (p75 - p25)/1.349.
    * "per_session" (openface2, openpose): center = each stream's OWN per-channel mean
      (computed in apply() at load time), scale = train within-session std. Instance
      normalization for landmark/pose coordinates.

Post-normalization clip:
    After z-scoring, values are clamped to [-10, 10].

Continuous engagement target:
    ``fit_norm_stats.py`` also writes ``label_continuous.{dataset}.mean/.std`` (plus a pooled
    ``label_continuous.mean/.std`` fallback). ``standardize_label``/``destandardize_label`` z-score
    the raw target by its own dataset's stats — NoXi and NoXi-J sit on different absolute
    mean/scale, so the regression head is trained on a shared, dataset-neutral scale instead.
"""

from __future__ import annotations

import os

import numpy as np

from .registry import (MODALITY_ORDER, MODALITY_DIMS, MODALITY_VALID_MIN,
                       MODALITY_NORM_MODE)
from .streams import FEATURE_CLIP, valid_feature_mask

NORM_CLIP = 10.0  # post-zscore clamp magnitude


class Normalizer:
    """Holds per-modality mean/std and keep_channels; applies channel selection + (x-mean)/std + clip."""

    def __init__(self, stats: dict | None = None, eps: float = 1e-6):
        self.eps = eps
        self.mean: dict = {}
        self.std: dict = {}
        self.keep_channels: dict = {}   # m -> int64 index array into raw D_m
        self.label_mean: dict = {}      # dataset -> float, continuous engagement target
        self.label_std: dict = {}
        self.label_mean_pooled: float | None = None
        self.label_std_pooled: float | None = None
        if stats:
            for m in MODALITY_ORDER:
                if f"{m}.mean" in stats:
                    self.mean[m] = np.asarray(stats[f"{m}.mean"], dtype=np.float32)
                    self.std[m] = np.asarray(stats[f"{m}.std"], dtype=np.float32)
                if f"{m}.keep_channels" in stats:
                    self.keep_channels[m] = np.asarray(stats[f"{m}.keep_channels"], dtype=np.int64)
            if "label_continuous.mean" in stats and "label_continuous.std" in stats:
                self.label_mean_pooled = float(np.asarray(stats["label_continuous.mean"]).reshape(-1)[0])
                self.label_std_pooled = max(
                    float(np.asarray(stats["label_continuous.std"]).reshape(-1)[0]), eps)
            prefix, suffix = "label_continuous.", ".mean"
            for k in stats:
                if not (k.startswith(prefix) and k.endswith(suffix)):
                    continue
                ds = k[len(prefix):-len(suffix)]
                if not ds:                       # the pooled "label_continuous.mean" has no dataset segment
                    continue
                std_key = f"{prefix}{ds}.std"
                if std_key not in stats:
                    continue
                self.label_mean[ds] = float(np.asarray(stats[k]).reshape(-1)[0])
                self.label_std[ds] = max(float(np.asarray(stats[std_key]).reshape(-1)[0]), eps)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        return cls(dict(np.load(path)))

    @property
    def effective_dims(self) -> dict[str, int]:
        """Effective feature dimension per modality after channel selection.

        Falls back to raw MODALITY_DIMS when no keep_channels are stored (e.g. before
        the first refit, or for modalities where all channels are kept).
        """
        out = {}
        for m in MODALITY_ORDER:
            if m in self.keep_channels:
                out[m] = int(len(self.keep_channels[m]))
            elif m in self.mean:
                out[m] = int(self.mean[m].shape[0])
            else:
                out[m] = MODALITY_DIMS[m]
        return out

    def apply(self, modality: str, x: np.ndarray) -> np.ndarray:
        """Select channels, normalize, neutralize invalid frames, clip to [-10, 10].

        ``x: (T, D_raw)`` raw (un-sanitized) stream -> ``(T, D_effective)`` float32, all finite.

        Invalid frames (non-finite, |x|>FEATURE_CLIP, or below the per-modality sentinel floor —
        e.g. OpenPose -1) are mapped to the *neutral* post-norm value 0 (the channel center).
        """
        x = np.asarray(x)
        if modality in self.keep_channels:
            x = x[:, self.keep_channels[modality]]
        valid = valid_feature_mask(x, FEATURE_CLIP, MODALITY_VALID_MIN.get(modality))

        if modality not in self.mean:
            # no stats (pre-fit / missing modality): passthrough, invalid -> 0.
            return np.where(valid, x, 0.0).astype(np.float32)

        std = np.maximum(self.std[modality], self.eps)
        if MODALITY_NORM_MODE.get(modality) == "per_session":
            # Instance centering: subtract this stream's OWN per-channel mean over valid frames
            # (fall back to the stored global mean for channels with no valid frame), scale by the
            # stored within-session std.
            n = valid.sum(axis=0, keepdims=True)
            ssum = np.where(valid, x, 0.0).sum(axis=0, keepdims=True)
            center = np.where(n > 0, ssum / np.maximum(n, 1.0), self.mean[modality])
        else:
            center = self.mean[modality]
        out = (x - center) / std
        out = np.where(valid, out, 0.0)                  # invalid -> neutral (post-norm center)
        return np.clip(out, -NORM_CLIP, NORM_CLIP).astype(np.float32)

    def _label_moments(self, dataset: str) -> tuple[float, float] | None:
        if dataset in self.label_mean:
            return self.label_mean[dataset], self.label_std[dataset]
        if self.label_mean_pooled is not None:
            return self.label_mean_pooled, self.label_std_pooled
        return None

    def standardize_label(self, dataset: str, y):
        """Per-dataset z-score for the continuous engagement target (falls back to the pooled
        stats, then to identity, when ``dataset`` has no stats — e.g. before the first refit).

        NaN entries (invalid/padded frames) pass through as NaN under this affine transform, so
        callers that mask on ``np.isfinite`` (``windowing.chunk_continuous``) are unaffected.
        """
        m = self._label_moments(dataset)
        if m is None:
            return y
        mean, std = m
        return (y - mean) / std

    def destandardize_label(self, dataset: str, y):
        """Inverse of :meth:`standardize_label` — maps a model's standardized-scale prediction
        back to raw ``[0, 1]``-ish engagement units. Identity when ``dataset`` has no stats."""
        m = self._label_moments(dataset)
        if m is None:
            return y
        mean, std = m
        return y * std + mean


def empty() -> Normalizer:
    """A no-op normalizer (used before stats are fit, e.g. during fitting)."""
    return Normalizer(None)


