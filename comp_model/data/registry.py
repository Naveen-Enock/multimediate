"""Dataset registry — structural facts grounded in the on-disk layout.

Modality availability is uneven; the loader zero-fills missing modalities and sets a
mask bit rather than crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Audio: egemapsv2, whisper.  Text (transcript): xlm_roberta.  Visual: swin, clip, openface2, openpose.
MODALITY_ORDER = ["egemapsv2", "whisper", "xlm_roberta", "swin", "clip",
                  "openface2", "openpose"]
MODALITY_DIMS = {
    "egemapsv2": 88, "whisper": 1280, "xlm_roberta": 768, "swin": 768,
    "clip": 512, "openface2": 714, "openpose": 139,
}

MODALITY_FILE = {
    "egemapsv2": ("{role}.audio.egemapsv2.stream",          "stream"),
    "whisper":   ("{role}.audio_whisper.npy",               "npy"),
    "xlm_roberta": ("{role}.audio.xlm_roberta_embeddings.stream", "stream"),
    "swin":      ("{role}.swin.stream",                     "stream"),
    "clip":      ("{role}.clip.stream",                     "stream"),
    "openface2": ("{role}.openface2.stream",                "stream"),
    "openpose":  ("{role}.openpose.stream",                 "stream"),
}

# Integer class maps for the PInSoRo CE head (ordered like the ordinal maps in
# engagement_analysis/engagement_label_report.py, but mapped to nominal indices).
SOCIAL_CLASSES = {
    "solitary": 0, "onlooker": 1, "parallel": 2, "associative": 3, "cooperative": 4,
}
TASK_CLASSES = {
    "noplay": 0, "aimless": 1, "goaloriented": 2, "adultseeking": 3,
}

# A modality is "present" iff this file exists in the resolved dir; openpose is
# present for every role across all datasets, so it is used to detect roles.
ROLE_PROBE_MODALITY = "openpose"
LABEL_FPS_CONTINUOUS = 25.0   # video + feature grid for NoXi, NoXi-J, MPII
LABEL_FPS_PINSORO = 30.0      # PInSoRo annotations are at 30 fps (resampled to 25)

# Number of social / task classes defined by the PInSoRo annotation protocol.
PINSORO_SOCIAL_CLASSES = len(SOCIAL_CLASSES)  # 5
PINSORO_TASK_CLASSES = len(TASK_CLASSES)       # 4

# Maximum interactive participants in any session across all datasets (determines collate padding
# and graph slot embeddings).
MAX_PARTNERS = 3

# ── Domain-adaptation metadata ids ────────────────────────────────────────────
ROLE_CLASSES = {"expert": 0, "novice": 1, "child": 2, "robot": 3, "group": 4, "environment": 5}
NUM_ROLES = len(ROLE_CLASSES)

# Session-level language of the interaction. Parsed from ``language.annotation.csv`` (field 2).
# MPII is recorded in German; PInSoRo is in English. Datasets without the file get "unknown".
LANGUAGE_CLASSES = {
    "English": 0, "French": 1, "German": 2, "Spanish": 3,
    "Italian": 4, "Arabic": 5, "Indonesian": 6,
    "Japanese": 7, "Chinese": 8,
    "unknown": 9,
}
NUM_LANGUAGES = len(LANGUAGE_CLASSES)

LABEL_KIND_IDS = {"continuous": 0, "categorical": 1}
NUM_LABEL_KINDS = len(LABEL_KIND_IDS)

NUM_PARTNER_COUNTS = MAX_PARTNERS + 1   # 0..MAX_PARTNERS partners present

# Per-node spatial framing / camera viewpoint class — a hidden domain factor distinct from the
# dataset id
FRAMING_CLASSES = {"closeup": 0, "wide": 1, "dynamic": 2, "scene": 3}
NUM_FRAMINGS = len(FRAMING_CLASSES)
_ROLE_CLASS_FRAMING = {
    ROLE_CLASSES["expert"]: "closeup",
    ROLE_CLASSES["novice"]: "closeup",
    ROLE_CLASSES["child"]: "dynamic",
    ROLE_CLASSES["robot"]: "dynamic",
    ROLE_CLASSES["group"]: "wide",
    ROLE_CLASSES["environment"]: "scene",
}

# Active-sensor multi-hot width: the per-node modality present-mask doubles as a modality-config
# bitmask fed to DomainFiLM.
NUM_MODALITIES = len(MODALITY_ORDER)

# Channels to unconditionally drop before any normalization or model input, keyed by modality.
# Data-dead channels (std=0 across the train corpus) are detected automatically in fit_norm_stats.py
# and merged here at fit time.
#   openface2 ch0: frame counter (1..N)
#   openface2 ch1: all-zero dead channel
#   openface2 ch2: timestamp in seconds
MODALITY_DROP_CHANNELS: dict[str, list[int]] = {
    "openface2": [0, 1, 2],
}

# Per-modality lower bound below which a value is a sentinel, not real data. OpenPose marks
# undetected keypoints/confidences with -1 while every genuine value (pixel coordinate, confidence)
# is non-negative.
MODALITY_VALID_MIN: dict[str, float] = {
    "openpose": 0.0,
}

# Normalization mode per modality.  All modes store a per-channel (center, scale) pair of shape
# (D_eff,) so Normalizer.apply() — (x - center) / scale, then clip ±10 — needs no special-casing;

MODALITY_NORM_MODE: dict[str, str] = {
    "xlm_roberta": "global_std",
    "egemapsv2": "robust",
    "openface2": "per_session",
    "openpose": "per_session",
}

# (dataset, on-disk role) -> semantic role class. PInSoRo is handled session-aware in
# role_class_id (env -> environment; the robot is detected per session). mpii subjectPos* and any
# unmapped role fall through to the dataset-level / "unknown" defaults below.
_ROLE_SEMANTIC = {
    ("noxi", "expert"): "expert",   ("noxi", "novice"): "novice",
    ("noxi_j", "expert"): "expert", ("noxi_j", "novice"): "novice",
}


def _pinsoro_is_robot(feature_dir: str, role: str) -> bool:
    """True if a PInSoRo purple/yellow node is the robot (child-robot sessions only).

    Robots carry no ``age``/``gender`` annotation, while children always do — so the target role
    whose ``{role}.age.annotation.csv`` is missing is the robot. In child-child sessions both
    children have the annotation, so this is False for both. Requires ``feature_dir``; without it
    the role defaults to "child".
    """
    if not feature_dir:
        return False
    return not os.path.exists(os.path.join(feature_dir, f"{role}.age.annotation.csv"))


def role_class_id(dataset: str, role: str, feature_dir: str | None = None) -> int:
    """Semantic role class id for an on-disk (dataset, role).

    PInSoRo is session-dependent: ``env`` is the shared scene view ("environment"); ``purple`` /
    ``yellow`` are children, except in child-robot sessions where the robot (the target role with no
    ``age`` annotation) maps to "robot". Pass ``feature_dir`` so the robot can be detected; without
    it both default to "child". mpii subjects -> 'group'. Every dataset role maps to a concrete
    class, so an unmapped role is a data/registry mismatch and raises.
    """
    if dataset == "pinsoro":
        if role == "env":
            return ROLE_CLASSES["environment"]
        if role in ("purple", "yellow"):
            return ROLE_CLASSES["robot" if _pinsoro_is_robot(feature_dir, role) else "child"]
    semantic = _ROLE_SEMANTIC.get((dataset, role))
    if semantic is None:
        if dataset == "mpii":
            semantic = "group"
        else:
            raise ValueError(f"unmapped role {role!r} for dataset {dataset!r}")
    return ROLE_CLASSES[semantic]


def label_kind_id(kind: str) -> int:
    return LABEL_KIND_IDS[kind]


def pinsoro_domain(feature_dir: str | None) -> str | None:
    """PInSoRo interaction domain — ``"cc"`` (child-child) / ``"cr"`` (child-robot).

    The dataset fans each split out to ``{split}-cc`` / ``{split}-cr`` directories (see
    ``_DATASET_SPLITS``), so the domain is the suffix of the split subdir that holds the session.
    Returns ``None`` for non-PInSoRo paths (any layout without the ``-cc``/``-cr`` subdir).
    """
    if not feature_dir:
        return None
    sub = os.path.basename(os.path.dirname(feature_dir))
    if sub.endswith("-cc"):
        return "cc"
    if sub.endswith("-cr"):
        return "cr"
    return None


def language_id(feature_dir: str, dataset: str) -> int:
    """Session-level language class id, read from ``language.annotation.csv``.

    Falls back to known per-dataset defaults (MPII → German, PInSoRo → English) when the file
    is absent, and to ``unknown`` for anything else missing.
    """
    import os
    path = os.path.join(feature_dir, "language.annotation.csv")
    if os.path.exists(path):
        with open(path) as f:
            parts = f.read().strip().split(";")
        if len(parts) >= 3:
            lang = parts[2].strip()
            return LANGUAGE_CLASSES.get(lang, LANGUAGE_CLASSES["unknown"])
    # fallback for datasets without the annotation file
    if dataset == "mpii":
        return LANGUAGE_CLASSES["German"]
    if dataset == "pinsoro":
        return LANGUAGE_CLASSES["English"]
    return LANGUAGE_CLASSES["unknown"]


def framing_for_role_class(role_class: int) -> int:
    """Per-node spatial-framing class id given a semantic role class id."""
    return FRAMING_CLASSES[_ROLE_CLASS_FRAMING[role_class]]


@dataclass(frozen=True)
class SessionRef:
    """One session of one dataset/split, with resolved directories."""
    dataset: str
    split: str            # logical split name (train/val/test/test_base/...)
    session_id: str
    feature_dir: str      # where the .stream files live
    label_dir: str        # where annotation csvs live (== feature_dir for NoXi)
    whisper_dir: str      # where {role}.audio_whisper.npy lives ("" if dataset has none)
    label_kind: str       # "continuous" | "categorical"
    roles: tuple = field(default_factory=tuple)  # roles present (detected)


# Logical split -> (feature_subdir, label_subdir, whisper_subdir). A None whisper
# subdir means "same as feature dir"; an empty string means "no whisper".
_DATASET_SPLITS = {
    "noxi": {
        "train": ("train", "train", None),
        "val": ("val", "val", None),
        "test_base": ("test-base", "test-base", None),
        "test_additional": ("test-additional", "test-additional", None),
    },
    "noxi_j": {
        "train": ("train", "train", None),
        "val": ("val", "val", None),
        "test": ("test", "test", None),
    },
    "mpii": {
        # Whisper lives in originalAudioVideo-{split}; labels only exist for val.
        "val": ("precomputed-features-val", "engagement-annotations-val",
                "originalAudioVideo-val"),
        "test": ("precomputed-features-test", "", "originalAudioVideo-test"),
    },
    # PInSoRo logical splits fan out to two on-disk dirs (child-child / child-robot).
    # ``val_cc``/``val_cr`` isolate one interaction type so validation kappa can be reported
    # per sub-split; ``val`` keeps the pooled view (used for the GT/pred histograms).
    "pinsoro": {
        "train": [("train-cc", "train-cc", ""), ("train-cr", "train-cr", "")],
        "val": [("val-cc", "val-cc", ""), ("val-cr", "val-cr", "")],
        "val_cc": [("val-cc", "val-cc", "")],
        "val_cr": [("val-cr", "val-cr", "")],
        "test": [("test-cc", "test-cc", ""), ("test-cr", "test-cr", "")],
        "test_cc": [("test-cc", "test-cc", "")],
        "test_cr": [("test-cr", "test-cr", "")],
    },
}

_LABEL_KIND = {
    "noxi": "continuous", "noxi_j": "continuous",
    "mpii": "continuous", "pinsoro": "categorical",
}

# Primary (scored) target roles. PInSoRo is handled session-aware in target_roles: only the
# children are scored — `env` (scene) and, in child-robot sessions, the robot are partner-only.
_TARGET_ROLES = {
    "noxi": ("expert", "novice"),
    "noxi_j": ("expert", "novice"),
    # mpii roles are detected per session (subjectPos1..4).
}


def label_kind(dataset: str) -> str:
    return _LABEL_KIND[dataset]


def detect_roles(feature_dir: str) -> tuple:
    """Roles present in a session = those with an openpose stream."""
    if not os.path.isdir(feature_dir):
        return tuple()
    roles = []
    suffix = f".{ROLE_PROBE_MODALITY}.stream"
    for fn in os.listdir(feature_dir):
        if fn.endswith(suffix) and not fn.endswith(".stream~"):
            roles.append(fn[: -len(suffix)])
    return tuple(sorted(roles))


def target_roles(dataset: str, present_roles: tuple, feature_dir: str | None = None) -> tuple:
    """Which present roles may serve as the scored target node.

    PInSoRo scores **children only**: ``purple``/``yellow`` that are children, never ``env`` (scene)
    and never the robot. In child-child both children are targets; in child-robot the robot (the
    child-position role with no ``age`` annotation) is a partner, leaving just the child. Pass
    ``feature_dir`` so the robot can be detected.
    """
    if dataset == "pinsoro":
        return tuple(r for r in present_roles
                     if r in ("purple", "yellow") and not _pinsoro_is_robot(feature_dir, r))
    if dataset in _TARGET_ROLES:
        allowed = set(_TARGET_ROLES[dataset])
        return tuple(r for r in present_roles if r in allowed)
    # mpii: any detected subject position can be the target.
    return present_roles


def iter_sessions(dataset: str, root: str, split: str):
    """Yield ``SessionRef`` for every session of ``dataset`` in logical ``split``."""
    spec = _DATASET_SPLITS[dataset][split]
    sub_specs = spec if isinstance(spec, list) else [spec]
    kind = _LABEL_KIND[dataset]
    for feat_sub, label_sub, whis_sub in sub_specs:
        feat_root = os.path.join(root, feat_sub)
        if not os.path.isdir(feat_root):
            continue
        for session_id in sorted(os.listdir(feat_root)):
            feature_dir = os.path.join(feat_root, session_id)
            if not os.path.isdir(feature_dir):
                continue
            roles = detect_roles(feature_dir)
            if not roles:
                continue
            label_dir = (os.path.join(root, label_sub, session_id)
                         if label_sub else "")
            if whis_sub is None:
                whisper_dir = feature_dir
            elif whis_sub == "":
                whisper_dir = ""
            else:
                whisper_dir = os.path.join(root, whis_sub, session_id)
            yield SessionRef(
                dataset=dataset, split=split, session_id=session_id,
                feature_dir=feature_dir, label_dir=label_dir,
                whisper_dir=whisper_dir, label_kind=kind, roles=roles,
            )


def splits_for(dataset: str) -> list:
    return list(_DATASET_SPLITS[dataset].keys())
