"""Turn window-level class probabilities into per-frame predictions.

Two stages, both cheap and both worth a lot of kappa:

1. **Overlap averaging.** Windows are 10 s wide with a 2 s hop, so every 30 fps frame is covered by
   about five of them. Their probabilities are accumulated with a triangular weight peaked at each
   window's centre, which is where its statistics are most representative.
2. **Viterbi smoothing.** PInSoRo labels are extremely persistent (mean run 44-237 s), so an HMM
   whose transition matrix is estimated from the training label runs removes the isolated flicker
   that unweighted kappa punishes hardest. The classifier posterior already contains the class
   prior, so it is divided out before decoding (a hybrid scaled likelihood) -- otherwise the prior
   is counted twice, once by the model and once by the chain's stationary distribution.
"""

from __future__ import annotations

import os

import numpy as np

from .columns import VIDEO_FPS

EPS = 1e-12


def frame_posterior(frame_start, frame_end, proba: np.ndarray, n_frames: int,
                    n_classes: int) -> np.ndarray:
    """Accumulate window probabilities onto the 30 fps grid -> ``(n_frames, n_classes)``.

    Rows for frames covered by no window (only possible if the caller passes a subset of a
    session's windows) are left uniform.
    """
    acc = np.zeros((n_frames, n_classes), np.float64)
    wsum = np.zeros(n_frames, np.float64)
    for s, e, p in zip(frame_start, frame_end, proba):
        s, e = int(s), int(e)
        if e <= s:
            continue
        w = 1.0 - np.abs(np.arange(s, e) - (s + e - 1) / 2.0) / max((e - s) / 2.0, 1.0)
        w = np.clip(w, 1e-3, None)
        acc[s:e] += w[:, None] * p[None, :]
        wsum[s:e] += w
    covered = wsum > 0
    acc[covered] /= wsum[covered][:, None]
    acc[~covered] = 1.0 / n_classes
    return acc


def transition_matrix(label_seqs, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    """Row-stochastic 30 fps transition matrix estimated from annotated label sequences.

    Counts transitions only between consecutive *annotated* frames, so unannotated gaps do not
    fabricate a jump. Laplace ``smoothing`` keeps unseen transitions reachable -- without it a
    class pair absent from training could never be decoded.
    """
    cnt = np.full((n_classes, n_classes), smoothing, np.float64)
    for lab in label_seqs:
        lab = np.asarray(lab)
        ok = (lab[:-1] >= 0) & (lab[1:] >= 0)
        if ok.any():
            np.add.at(cnt, (lab[:-1][ok], lab[1:][ok]), 1.0)
    return cnt / cnt.sum(axis=1, keepdims=True)


def class_prior(label_seqs, n_classes: int, smoothing: float = 1.0) -> np.ndarray:
    cnt = np.full(n_classes, smoothing, np.float64)
    for lab in label_seqs:
        lab = np.asarray(lab)
        v = lab[lab >= 0]
        if v.size:
            cnt += np.bincount(v, minlength=n_classes)
    return cnt / cnt.sum()


def viterbi(log_emit: np.ndarray, log_trans: np.ndarray, log_start: np.ndarray) -> np.ndarray:
    """Maximum-likelihood state path for ``(T, K)`` log emissions. Returns ``(T,)`` int64."""
    T, K = log_emit.shape
    if T == 0:
        return np.zeros(0, np.int64)
    delta = log_start + log_emit[0]
    psi = np.zeros((T, K), np.int32)
    for t in range(1, T):
        m = delta[:, None] + log_trans          # (K_prev, K_next)
        psi[t] = m.argmax(axis=0)
        delta = m.max(axis=0) + log_emit[t]
    path = np.empty(T, np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def smooth(posterior: np.ndarray, trans: np.ndarray, prior: np.ndarray,
           divide_prior: bool = True, stickiness: float = 1.0) -> np.ndarray:
    """Viterbi-decode a per-frame posterior into a class sequence.

    ``stickiness`` raises the transition matrix to a power (>1 makes the chain more persistent,
    <1 less); 1.0 uses the empirical matrix as estimated.
    """
    p = np.clip(posterior, EPS, None)
    if divide_prior:
        p = p / np.clip(prior, EPS, None)[None, :]
    log_emit = np.log(p)
    log_trans = stickiness * np.log(np.clip(trans, EPS, None))
    return viterbi(log_emit, log_trans, np.log(np.clip(prior, EPS, None)))


def apply_bias(posterior: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Additive logit bias on a probability array, renormalised (used for prior calibration)."""
    z = np.log(np.clip(posterior, EPS, None)) + bias[None, :]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ── submission ────────────────────────────────────────────────────────────────

def write_prediction_csv(out_dir: str, session: str, role: str, axis: str,
                         frame_pred: np.ndarray, class_names) -> str:
    """One class string per 30 fps frame, matching ``submission/pinsoro-*/<sess>/<role>.<axis>``."""
    d = os.path.join(out_dir, session)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{role}.{axis}_engagement.prediction.csv")
    with open(path, "w") as f:
        f.write("\n".join(class_names[int(c)] for c in frame_pred) + "\n")
    return path


def frames_to_seconds(n_frames: int) -> float:
    return n_frames / VIDEO_FPS
