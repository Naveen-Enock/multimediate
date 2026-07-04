"""Submission-format writers for challenge prediction files.

Writes per-frame predictions per (dataset, session, participant) under an output dir.
"""

from __future__ import annotations

import csv
import os

import numpy as np


def write_continuous_submission(out_dir: str, dataset: str, session_id: str,
                                participant_id: str, frame_preds: np.ndarray):
    """Write one CSV of per-frame engagement predictions.

    frame_preds: 1-D array at 25 fps. Path:
        <out_dir>/<dataset>/<session_id>/<participant_id>.engagement.pred.csv
    """
    sess_dir = os.path.join(out_dir, dataset, session_id)
    os.makedirs(sess_dir, exist_ok=True)
    path = os.path.join(sess_dir, f"{participant_id}.engagement.pred.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for v in np.asarray(frame_preds).reshape(-1):
            w.writerow([f"{float(v):.6f}"])
    return path


def write_categorical_submission(out_dir: str, dataset: str, session_id: str,
                                 participant_id: str, social: np.ndarray,
                                 task: np.ndarray):
    """Write per-frame PInSoRo class predictions (social + task) as two columns."""
    sess_dir = os.path.join(out_dir, dataset, session_id)
    os.makedirs(sess_dir, exist_ok=True)
    path = os.path.join(sess_dir, f"{participant_id}.engagement.pred.csv")
    social = np.asarray(social).reshape(-1)
    task = np.asarray(task).reshape(-1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["social", "task"])
        for s, t in zip(social, task):
            w.writerow([int(s), int(t)])
    return path
