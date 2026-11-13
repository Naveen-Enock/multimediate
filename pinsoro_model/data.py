"""Load the cached window tables into model matrices.

One row per (session, target child, window). Sessions -- not children and not windows -- are the
grouping unit for cross-validation: the two children of a child-child session share a recording,
a room, a pair of cameras and one shared microphone, so splitting them across folds would leak.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from . import sessions

from .extract import NPZ_SUFFIX, SCHEMA_PATH
from .feature_sets import V2_SUFFIX

PINSORO_ROOT = "/path/to/pinsoro"
AXES = ("social", "task")
N_CLASSES = {"social": sessions.NUM_SOCIAL_CLASSES, "task": sessions.NUM_TASK_CLASSES}
CLASS_NAMES = {
    "social": [n for n, _ in sorted(sessions.SOCIAL_CLASSES.items(), key=lambda kv: kv[1])],
    "task": [n for n, _ in sorted(sessions.TASK_CLASSES.items(), key=lambda kv: kv[1])],
}


@dataclass
class Tables:
    """Stacked window tables plus everything decoding and grouping need.

    ``X`` is ``(n_win, n_feat)``; ``y`` / ``purity`` are per-axis ``(n_win,)``; ``group`` is the
    session id per row; ``table_id`` indexes ``tables`` (one entry per session/role file), which
    carries the per-session frame bookkeeping used to rebuild per-frame predictions.
    ``win_idx`` is each row's position within its own table, so a subset of rows can be mapped
    back to the frame spans it covers.
    """
    X: np.ndarray
    y: dict
    purity: dict
    group: np.ndarray
    domain: np.ndarray
    split: np.ndarray
    table_id: np.ndarray
    win_idx: np.ndarray
    feat_names: list
    tables: list

    def __len__(self):
        return len(self.X)

    def mask(self, *, domain=None, split=None):
        m = np.ones(len(self.X), bool)
        if domain is not None:
            m &= self.domain == domain
        if split is not None:
            m &= np.isin(self.split, [split] if isinstance(split, str) else list(split))
        return m


def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def tier_columns(feat_names, tier: str, schema: dict | None = None) -> np.ndarray:
    """Column indices of a feature tier (``T0`` / ``T1`` / ``T2``)."""
    schema = schema or load_schema()
    if tier == "T2":
        keep = set(feat_names)
    elif tier == "T1":
        keep = {n for n in feat_names if schema["tiers"].get(n) == "T1"}
    elif tier == "T0":
        keep = set(schema["T0"])
    else:
        raise ValueError(f"unknown tier {tier!r}")
    idx = np.array([i for i, n in enumerate(feat_names) if n in keep], dtype=np.int64)
    if len(idx) == 0:
        raise ValueError(f"tier {tier} selected no columns")
    return idx


# Which cached table each feature set reads: T0-T2 are column subsets of the original 442-feature
# tables; V2 has its own tables written by extract_v2.py.
FEATURE_SET_SUFFIX = {"T0": NPZ_SUFFIX, "T1": NPZ_SUFFIX, "T2": NPZ_SUFFIX, "V2": V2_SUFFIX}


def columns_for(feat_names, feature_set: str, domain: str, schema: dict | None = None) -> np.ndarray:
    """Column indices a model for ``domain`` uses under ``feature_set``.

    V2 differs by domain: cr drops the dyadic block, because the robot partner is barely tracked.
    """
    if feature_set == "V2":
        from .feature_sets import V2_CC, V2_CR
        want = V2_CC if domain == "cc" else V2_CR
        pos = {n: i for i, n in enumerate(feat_names)}
        missing = [n for n in want if n not in pos]
        if missing:
            raise ValueError(f"V2 columns missing from the table: {missing[:5]}")
        return np.array([pos[n] for n in want], dtype=np.int64)
    return tier_columns(feat_names, feature_set, schema)


def iter_split_sessions(root: str, split: str, domain: str):
    """Yield ``(session_id, session_dir)`` straight from ``<root>/<split>-<domain>/``.

    This package uses PInSoRo's own on-disk train/val/test split exactly as shipped. PInSoRo keeps
    annotations beside the features, so the session dir is both feature and label dir.
    """
    d = os.path.join(root, f"{split}-{domain}")
    if not os.path.isdir(d):
        return
    for sid in sorted(os.listdir(d)):
        sd = os.path.join(d, sid)
        if os.path.isdir(sd):
            yield sid, sd


def load(root: str = PINSORO_ROOT, splits=("train", "val"),
         domains=("cc", "cr"), suffix: str = NPZ_SUFFIX) -> Tables:
    """Stack every cached ``<role>.<suffix>`` table for the requested on-disk splits/domains."""
    Xs, groups, doms, spl, tid, widx, tables = [], [], [], [], [], [], []
    ys = {a: [] for a in AXES}
    ps = {a: [] for a in AXES}
    names = schema_hash = None

    for split in splits:
        for dom in domains:
            for sid, sdir in iter_split_sessions(root, split, dom):
                roles = sessions.detect_roles(sdir)
                for role in sessions.target_roles(roles, sdir):
                    path = os.path.join(sdir, f"{role}.{suffix}")
                    if not os.path.exists(path):
                        continue
                    z = np.load(path, allow_pickle=False)
                    nm, h = [str(x) for x in z["feat_names"]], str(z["schema_hash"])
                    if names is None:
                        names, schema_hash = nm, h
                    elif h != schema_hash:
                        raise RuntimeError(f"schema hash mismatch in {path} -- re-run extract")
                    k = len(tables)
                    tables.append({
                        "session": sid, "role": role, "domain": dom,
                        "split": split, "path": path,
                        "frame_start": z["frame_start"], "frame_end": z["frame_end"],
                        "n_frames": int(z["n_frames_30fps"]), "label_dir": sdir,
                    })
                    n = len(z["X"])
                    Xs.append(z["X"])
                    for a in AXES:
                        ys[a].append(z[f"y_{a}"])
                        ps[a].append(z[f"purity_{a}"])
                    groups.append(np.full(n, sid))
                    doms.append(np.full(n, dom))
                    spl.append(np.full(n, split))
                    tid.append(np.full(n, k, np.int64))
                    widx.append(np.arange(n, dtype=np.int64))

    if not Xs:
        raise RuntimeError(f"no cached *.{suffix} tables -- run the matching extractor "
                           "(pinsoro_model.extract or .extract_v2)")
    return Tables(
        X=np.concatenate(Xs), y={a: np.concatenate(ys[a]) for a in AXES},
        purity={a: np.concatenate(ps[a]) for a in AXES},
        group=np.concatenate(groups), domain=np.concatenate(doms),
        split=np.concatenate(spl), table_id=np.concatenate(tid),
        win_idx=np.concatenate(widx), feat_names=names, tables=tables)


def class_weights(y: np.ndarray, n_classes: int, alpha: float) -> np.ndarray:
    """Per-class weight ``freq^-alpha``, normalised to mean 1 over the observed classes.

    ``alpha = 0`` leaves the natural marginal alone; ``alpha = 1`` is fully balanced. Unweighted
    nominal kappa penalises both under- and over-prediction of a rare class, so the best alpha is
    usually interior and is tuned by grouped CV rather than assumed.
    """
    counts = np.bincount(y[y >= 0], minlength=n_classes).astype(np.float64)
    freq = np.where(counts > 0, counts / max(counts.sum(), 1.0), np.nan)
    w = np.where(np.isfinite(freq), freq ** (-alpha), 0.0)
    seen = np.isfinite(freq)
    if seen.any():
        w = w / max(w[seen].mean(), 1e-12)
    return w


def sample_weights(y: np.ndarray, purity: np.ndarray, n_classes: int, alpha: float) -> np.ndarray:
    """Class weight x window purity -- boundary-straddling windows count less than clean ones."""
    cw = class_weights(y, n_classes, alpha)
    w = np.where(y >= 0, cw[np.clip(y, 0, n_classes - 1)], 0.0) * np.clip(purity, 0.0, 1.0)
    return w.astype(np.float64)


def frame_labels(table: dict, axis: str) -> np.ndarray:
    """Ground-truth per-frame labels (30 fps, ``-1`` = unannotated) for one cached table.

    Read position-preservingly -- unannotated frames are blank lines in the annotation CSV, and
    dropping them would shift every later label (see ``extract.read_frame_labels``).
    """
    from .extract import read_frame_labels
    cmap = sessions.SOCIAL_CLASSES if axis == "social" else sessions.TASK_CLASSES
    path = os.path.join(table["label_dir"] or "", f"{table['role']}.{axis}_engagement.annotation.csv")
    return read_frame_labels(path if table["label_dir"] else "", cmap, table["n_frames"])
