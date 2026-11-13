"""PInSoRo on-disk layout: label classes, splits, session discovery and scored roles.

Each logical split fans out to a child-child (``<split>-cc``) and a child-robot (``<split>-cr``)
directory; annotations live beside the feature streams, so a session dir is both feature and label
dir. Only children are scored — ``env`` (scene camera) and the robot are partner nodes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Integer class maps for the social / task labels (nominal indices).
SOCIAL_CLASSES = {
    "solitary": 0, "onlooker": 1, "parallel": 2, "associative": 3, "cooperative": 4,
}
TASK_CLASSES = {
    "noplay": 0, "aimless": 1, "goaloriented": 2, "adultseeking": 3,
}
NUM_SOCIAL_CLASSES = len(SOCIAL_CLASSES)  # 5
NUM_TASK_CLASSES = len(TASK_CLASSES)      # 4

# A role is present iff its openpose stream exists (every node carries one).
ROLE_PROBE_SUFFIX = ".openpose.stream"

# Logical split -> on-disk subdirs.
SPLITS = {
    "train": ("train-cc", "train-cr"),
    "val": ("val-cc", "val-cr"),
    "test": ("test-cc", "test-cr"),
}


@dataclass(frozen=True)
class SessionRef:
    """One PInSoRo session: its split, id, directory and the roles present."""
    split: str
    session_id: str
    feature_dir: str
    label_dir: str
    roles: tuple = field(default_factory=tuple)


def detect_roles(feature_dir: str) -> tuple:
    """Roles present in a session = those with an openpose stream."""
    if not os.path.isdir(feature_dir):
        return tuple()
    return tuple(sorted(fn[: -len(ROLE_PROBE_SUFFIX)] for fn in os.listdir(feature_dir)
                        if fn.endswith(ROLE_PROBE_SUFFIX)))


def is_robot(feature_dir: str, role: str) -> bool:
    """True if a ``purple``/``yellow`` node is the robot (child-robot sessions only).

    Robots carry no ``age`` annotation while children always do, so the role whose
    ``{role}.age.annotation.csv`` is missing is the robot.
    """
    if not feature_dir:
        return False
    return not os.path.exists(os.path.join(feature_dir, f"{role}.age.annotation.csv"))


def target_roles(present_roles: tuple, feature_dir: str) -> tuple:
    """Scored roles: ``purple``/``yellow`` children, never ``env`` and never the robot."""
    return tuple(r for r in present_roles
                 if r in ("purple", "yellow") and not is_robot(feature_dir, r))


def domain_of(feature_dir: str | None) -> str | None:
    """Interaction domain — ``"cc"`` (child-child) / ``"cr"`` (child-robot) — from the split dir."""
    if not feature_dir:
        return None
    sub = os.path.basename(os.path.dirname(feature_dir))
    if sub.endswith("-cc"):
        return "cc"
    if sub.endswith("-cr"):
        return "cr"
    return None


def iter_sessions(root: str, split: str):
    """Yield ``SessionRef`` for every session of logical ``split`` under ``root``."""
    for sub in SPLITS[split]:
        split_dir = os.path.join(root, sub)
        if not os.path.isdir(split_dir):
            continue
        for session_id in sorted(os.listdir(split_dir)):
            feature_dir = os.path.join(split_dir, session_id)
            if not os.path.isdir(feature_dir):
                continue
            roles = detect_roles(feature_dir)
            if roles:
                yield SessionRef(split=split, session_id=session_id, feature_dir=feature_dir,
                                 label_dir=feature_dir, roles=roles)
