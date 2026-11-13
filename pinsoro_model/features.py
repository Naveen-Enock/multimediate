"""Per-window statistics over the raw PInSoRo streams (Blocks A-H of the plan).

Design notes that matter for correctness:

* Every per-frame series is built **once per session** (:class:`NodeSeries`) with sentinels turned
  into ``NaN``; window statistics are then plain slices. Nothing is resampled -- audio keeps its
  25 fps indexing and is sliced by the matching time interval.
* A statistic with fewer than ``MIN_VALID`` usable frames is emitted as ``NaN``, never ``0``.
  XGBoost learns a per-split default direction for ``NaN``, which is the right semantics for
  "face not tracked" / "keypoint not detected"; zero-filling would drop those samples into the
  middle of the feature's real range.
* **The audio is a shared room microphone, not a per-child signal.** Verified across cc and cr
  sessions: the two children's ``audio.egemapsv2`` streams are 33-58% bit-identical and their
  continuous channels correlate at r = 0.93-0.96. Audio therefore cannot attribute speech to a
  speaker, so every acoustic feature is emitted under the ``room.`` prefix and no turn-taking /
  speaker-overlap feature is computed -- those would be pure artifacts. Vision *is* per-child
  (OpenFace2 3% identical; OpenPose body keypoints 0.02-2% identical, mean |diff| 52-207 px), so
  the dyadic block is built from vision alone.
* The audio stream's channels are opaque (see :mod:`columns`), so speech activity is derived
  structurally: ``valid_frac`` is the per-frame share of defined channels (functionals go
  undefined when the analysis window holds no voiced speech) and ``energy`` is the mean |value|
  over the continuous channels. ``active`` thresholds ``energy`` at a quantile of the pooled
  distribution over *both children of the session*, so the child who vocalises more gets the
  higher active fraction while the threshold itself stays label-free and session-relative
  (the streams are standardized per recording, so no absolute threshold exists).
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from .streams import FEATURE_CLIP, read_stream

from . import columns as C
from .windows import time_slice

MIN_VALID = 10          # fewer usable frames than this -> NaN
LAG_SEC = 2.0           # +-window for the lagged-synchrony search
LAG_STEP_SEC = 0.2

warnings.filterwarnings("ignore", r"(Mean|Degrees of freedom|All-NaN|invalid value)", RuntimeWarning)


# ── small NaN-aware statistics ────────────────────────────────────────────────

def _n(a):
    return int(np.isfinite(a).sum())


def nmean(a):
    return float(np.nanmean(a)) if _n(a) >= MIN_VALID else np.nan


def nstd(a):
    return float(np.nanstd(a)) if _n(a) >= MIN_VALID else np.nan


def npct(a, p):
    return float(np.nanpercentile(a, p)) if _n(a) >= MIN_VALID else np.nan


def nfrac(mask):
    """Fraction of ``True`` in a boolean array (always defined; no validity floor)."""
    return float(np.mean(mask)) if mask.size else np.nan


def nmean_abs_diff(a):
    """Mean |first difference| -- the series' motion energy."""
    d = np.abs(np.diff(a))
    return float(np.nanmean(d)) if _n(d) >= MIN_VALID else np.nan


def sign_change_rate(a):
    """Fraction of consecutive valid pairs whose sign flips (gaze left/right shift rate)."""
    ok = np.isfinite(a)
    if ok.sum() < MIN_VALID:
        return np.nan
    v = a[ok]
    return float(np.mean(np.sign(v[:-1]) != np.sign(v[1:]))) if len(v) > 1 else np.nan


def slope(a, fps=C.VIDEO_FPS):
    """OLS slope per second over the window (drift: leaning in / pulling back)."""
    ok = np.isfinite(a)
    if ok.sum() < MIN_VALID:
        return np.nan
    t = np.flatnonzero(ok) / fps
    v = a[ok]
    t = t - t.mean()
    denom = float((t * t).sum())
    return float((t * (v - v.mean())).sum() / denom) if denom > 0 else np.nan


def corr(a, b):
    """Pearson r over the pairwise-valid frames of two equal-length series."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_VALID:
        return np.nan
    x, y = a[ok], b[ok]
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def best_lag(a, b, fps):
    """Lag (seconds) maximising |r(a, b shifted)| within +-``LAG_SEC``; NaN if undefined.

    A positive lag means the partner's series leads the target's.
    """
    step = max(1, int(round(LAG_STEP_SEC * fps)))
    span = int(round(LAG_SEC * fps))
    best_r, best_k = -1.0, np.nan
    for k in range(-span, span + 1, step):
        if k < 0:
            r = corr(a[-k:], b[:len(b) + k])
        elif k > 0:
            r = corr(a[:len(a) - k], b[k:])
        else:
            r = corr(a, b)
        if np.isfinite(r) and abs(r) > best_r:
            best_r, best_k = abs(r), k / fps
    return float(best_k) if np.isfinite(best_k) else np.nan


def runs_of(mask):
    """Lengths of the ``True`` runs in a boolean array."""
    if mask.size == 0 or not mask.any():
        return np.empty(0, np.int64)
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


# ── per-session derived series ────────────────────────────────────────────────

class NodeSeries:
    """All per-frame series for one (session, role), sentinels replaced by NaN.

    Video series are on the 30 fps grid (``T30``), audio on the 25 fps grid (``T25``). ``has_audio``
    is False for the PInSoRo ``env`` node, which carries no microphone.
    """

    def __init__(self, feature_dir: str, role: str):
        self.role = role
        of2, sr = read_stream(os.path.join(feature_dir, f"{role}.openface2.stream"))
        assert of2.shape[1] == C.OF2_DIM and sr == C.VIDEO_FPS
        op, sr = read_stream(os.path.join(feature_dir, f"{role}.openpose.stream"))
        assert op.shape[1] == C.OP_DIM and sr == C.VIDEO_FPS
        self.T30 = min(len(of2), len(op))
        of2, op = of2[:self.T30], op[:self.T30]

        # -- OpenFace: mask every derived quantity by a successful, confident track.
        # A plausible head distance is part of "tracked": see the OF2_TZ_* note in columns.py.
        self.of_ok = ((of2[:, C.OF2_SUCCESS] > 0.5)
                      & (of2[:, C.OF2_CONFIDENCE] > C.OF2_TRACK_CONF_MIN)
                      & (of2[:, C.OF2_POSE_TZ] >= C.OF2_TZ_MIN_MM)
                      & (of2[:, C.OF2_POSE_TZ] <= C.OF2_TZ_MAX_MM))
        m = np.where(self.of_ok, 1.0, np.nan).astype(np.float32)
        self.confidence = of2[:, C.OF2_CONFIDENCE].copy()
        self.gaze_x = of2[:, C.OF2_GAZE_ANGLE_X] * m
        self.gaze_y = of2[:, C.OF2_GAZE_ANGLE_Y] * m
        self.gaze_vec = of2[:, C.OF2_GAZE_VEC] * m[:, None]
        self.pose_t = of2[:, C.OF2_POSE_T] * m[:, None]
        self.pose_r = of2[:, C.OF2_POSE_R] * m[:, None]
        self.au_r = of2[:, C.OF2_AU_R] * m[:, None]
        self.au_c = of2[:, C.OF2_AU_C] * m[:, None]
        self.head_trans_speed = _speed(self.pose_t)
        self.head_rot_speed = _speed(self.pose_r)

        # -- OpenPose: neck-centred, shoulder-width-normalised keypoints.
        x, y, c = C.op_xyc(op)
        self.kp_valid = C.op_valid(x, y, c)
        km = np.where(self.kp_valid, 1.0, np.nan).astype(np.float32)
        self.kp_x, self.kp_y, self.kp_conf = x * km, y * km, c * km
        self.pose_score = np.where(op[:, C.OP_POSE_SCORE_COL] > C.OP_SENTINEL,
                                   op[:, C.OP_POSE_SCORE_COL], np.nan).astype(np.float32)
        rs, ls, nk = C.KP["RShoulder"], C.KP["LShoulder"], C.KP["neck"]
        self.shoulder_w = np.hypot(self.kp_x[:, rs] - self.kp_x[:, ls],
                                   self.kp_y[:, rs] - self.kp_y[:, ls])
        self.shoulder_angle = np.arctan2(self.kp_y[:, ls] - self.kp_y[:, rs],
                                         self.kp_x[:, ls] - self.kp_x[:, rs])
        ref = np.nanmedian(self.shoulder_w) if np.isfinite(self.shoulder_w).any() else np.nan
        scale = np.where(np.isfinite(self.shoulder_w), self.shoulder_w, ref)
        scale = np.where(np.isfinite(scale) & (scale > 1.0), scale, np.nan).astype(np.float32)
        self.torso_scale = scale
        self.xn = (self.kp_x - self.kp_x[:, nk:nk + 1]) / scale[:, None]
        self.yn = (self.kp_y - self.kp_y[:, nk:nk + 1]) / scale[:, None]
        self.kp_speed = np.full_like(self.xn, np.nan)
        self.kp_speed[1:] = np.hypot(np.diff(self.xn, axis=0), np.diff(self.yn, axis=0))
        self.hand_vis_r = (op[:, C.OP_HAND_R] > C.OP_SENTINEL).any(axis=1)
        self.hand_vis_l = (op[:, C.OP_HAND_L] > C.OP_SENTINEL).any(axis=1)
        rw, lw = C.KP["RWrist"], C.KP["LWrist"]
        self.hand_motion = np.nanmean(self.kp_speed[:, [rw, lw]], axis=1)

        # -- Audio (absent for the env node).
        eg_path = os.path.join(feature_dir, f"{role}.audio.egemapsv2.stream")
        self.has_audio = os.path.exists(eg_path) and os.path.exists(eg_path + "~")
        if self.has_audio:
            eg, sr = read_stream(eg_path)
            assert eg.shape[1] == C.EGEMAPS_DIM and sr == C.AUDIO_FPS
            bad = ~np.isfinite(eg) | (np.abs(eg) > FEATURE_CLIP)
            self.eg = np.where(bad, np.nan, eg).astype(np.float32)
            self.T25 = len(eg)
            self.valid_frac = (~bad).mean(axis=1).astype(np.float32)
            self.energy = np.nanmean(np.abs(self.eg[:, C.EG_CONT]), axis=1).astype(np.float32)
            self.active = np.zeros(self.T25, bool)   # set by set_activity_threshold()
            self.energy_thr = np.nan
        else:
            self.eg = None
            self.T25 = 0
            self.active = np.zeros(0, bool)
            self.valid_frac = self.energy = np.zeros(0, np.float32)
            self.energy_thr = np.nan

    def audio_slice(self, s30: int, e30: int):
        return time_slice(s30, e30, C.AUDIO_FPS, self.T25)


def set_activity_threshold(nodes, q: float = 0.6) -> None:
    """Mark vocal activity on every node using one threshold pooled over the session's children.

    A per-node quantile would force every child to the same active fraction and destroy the "who
    talks more" contrast; pooling the two children's energy keeps that contrast while staying
    session-relative (the streams are standardized per recording) and label-free.
    """
    pool = np.concatenate([n.energy[np.isfinite(n.energy)] for n in nodes if n.has_audio]) \
        if any(n.has_audio for n in nodes) else np.empty(0)
    thr = float(np.quantile(pool, q)) if pool.size else np.inf
    for n in nodes:
        if n.has_audio:
            n.energy_thr = thr
            n.active = np.isfinite(n.energy) & (n.energy > thr)


def _speed(v: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean speed of a ``(T, D)`` series; NaN on the first frame and gaps."""
    out = np.full(len(v), np.nan, np.float32)
    out[1:] = np.sqrt(np.sum(np.diff(v, axis=0) ** 2, axis=1))
    return out


# ── Block A: target child, OpenFace2 (108) ────────────────────────────────────

def block_a(s: NodeSeries, lo: int, hi: int, prefix: str) -> dict:
    f = {}
    f[f"{prefix}.of2.success.frac"] = nfrac(s.of_ok[lo:hi])
    f[f"{prefix}.of2.confidence.mean"] = nmean(s.confidence[lo:hi])
    f[f"{prefix}.of2.confidence.std"] = nstd(s.confidence[lo:hi])
    for name, arr in (("gaze_angle_x", s.gaze_x[lo:hi]), ("gaze_angle_y", s.gaze_y[lo:hi])):
        f[f"{prefix}.of2.{name}.mean"] = nmean(arr)
        f[f"{prefix}.of2.{name}.std"] = nstd(arr)
        f[f"{prefix}.of2.{name}.p10"] = npct(arr, 10)
        f[f"{prefix}.of2.{name}.p90"] = npct(arr, 90)
        f[f"{prefix}.of2.{name}.mean_abs_delta"] = nmean_abs_diff(arr)
        f[f"{prefix}.of2.{name}.sign_change_rate"] = sign_change_rate(arr)
    for j, gname in enumerate(["gaze0_x", "gaze0_y", "gaze0_z", "gaze1_x", "gaze1_y", "gaze1_z"]):
        col = s.gaze_vec[lo:hi, j]
        f[f"{prefix}.of2.{gname}.mean"] = nmean(col)
        f[f"{prefix}.of2.{gname}.std"] = nstd(col)
    for j, pname in enumerate(["pose_Tx", "pose_Ty", "pose_Tz"]):
        col = s.pose_t[lo:hi, j]
        f[f"{prefix}.of2.{pname}.mean"] = nmean(col)
        f[f"{prefix}.of2.{pname}.std"] = nstd(col)
        f[f"{prefix}.of2.{pname}.mean_abs_delta"] = nmean_abs_diff(col)
    f[f"{prefix}.of2.pose_Tz.slope"] = slope(s.pose_t[lo:hi, 2])
    f[f"{prefix}.of2.head_trans_speed.mean"] = nmean(s.head_trans_speed[lo:hi])
    f[f"{prefix}.of2.head_trans_speed.std"] = nstd(s.head_trans_speed[lo:hi])
    f[f"{prefix}.of2.head_rot_speed.mean"] = nmean(s.head_rot_speed[lo:hi])
    f[f"{prefix}.of2.head_rot_speed.std"] = nstd(s.head_rot_speed[lo:hi])
    for j, pname in enumerate(["pose_Rx", "pose_Ry", "pose_Rz"]):
        col = s.pose_r[lo:hi, j]
        f[f"{prefix}.of2.{pname}.mean"] = nmean(col)
        f[f"{prefix}.of2.{pname}.std"] = nstd(col)
        f[f"{prefix}.of2.{pname}.mean_abs_delta"] = nmean_abs_diff(col)
    au = s.au_r[lo:hi]
    for j, aname in enumerate(C.AU_R_NAMES):
        f[f"{prefix}.of2.{aname}_r.mean"] = nmean(au[:, j])
        f[f"{prefix}.of2.{aname}_r.std"] = nstd(au[:, j])
    for aname in C.SALIENT_AU_R:
        col = au[:, C.AU_R_INDEX[aname]]
        f[f"{prefix}.of2.{aname}_r.p90"] = npct(col, 90)
        f[f"{prefix}.of2.{aname}_r.frac_gt_1"] = (float(np.nanmean(col > 1.0))
                                                  if _n(col) >= MIN_VALID else np.nan)
    auc = s.au_c[lo:hi]
    for j, aname in enumerate(C.AU_C_NAMES):
        col = auc[:, j]
        f[f"{prefix}.of2.{aname}_c.active_frac"] = (float(np.nanmean(col))
                                                    if _n(col) >= MIN_VALID else np.nan)
    return f


# ── Block B: target child, OpenPose (69) ──────────────────────────────────────

def block_b(s: NodeSeries, lo: int, hi: int, prefix: str) -> dict:
    f = {}
    val = s.kp_valid[lo:hi]
    for j, kname in enumerate(C.KP_NAMES):
        f[f"{prefix}.op.{kname}.detect_frac"] = nfrac(val[:, j])
    for kname in C.CORE_KP:
        j = C.KP[kname]
        f[f"{prefix}.op.{kname}.xn.mean"] = nmean(s.xn[lo:hi, j])
        f[f"{prefix}.op.{kname}.xn.std"] = nstd(s.xn[lo:hi, j])
        f[f"{prefix}.op.{kname}.yn.mean"] = nmean(s.yn[lo:hi, j])
        f[f"{prefix}.op.{kname}.yn.std"] = nstd(s.yn[lo:hi, j])
    for kname in C.CORE_KP:
        f[f"{prefix}.op.{kname}.motion"] = nmean(s.kp_speed[lo:hi, C.KP[kname]])
    f[f"{prefix}.op.detect_rate"] = nfrac(val)
    f[f"{prefix}.op.hand_motion.mean"] = nmean(s.hand_motion[lo:hi])
    f[f"{prefix}.op.torso_scale.mean"] = nmean(s.torso_scale[lo:hi])
    f[f"{prefix}.op.torso_scale.std"] = nstd(s.torso_scale[lo:hi])
    f[f"{prefix}.op.shoulder_angle.mean"] = nmean(s.shoulder_angle[lo:hi])
    f[f"{prefix}.op.shoulder_angle.std"] = nstd(s.shoulder_angle[lo:hi])
    f[f"{prefix}.op.pose_score.mean"] = nmean(s.pose_score[lo:hi])
    f[f"{prefix}.op.pose_score.std"] = nstd(s.pose_score[lo:hi])
    f[f"{prefix}.op.torso_scale.slope"] = slope(s.torso_scale[lo:hi])
    nk = C.KP["neck"]
    for side, kname in (("R", "RWrist"), ("L", "LWrist")):
        # yn is measured downward in image coords, so "above the neck" is yn < 0.
        h = -(s.yn[lo:hi, C.KP[kname]] - s.yn[lo:hi, nk])
        f[f"{prefix}.op.{side}wrist_height.mean"] = nmean(h)
        f[f"{prefix}.op.{side}wrist_above_neck.frac"] = (float(np.nanmean(h > 0))
                                                         if _n(h) >= MIN_VALID else np.nan)
    f[f"{prefix}.op.Rhand_visible.frac"] = nfrac(s.hand_vis_r[lo:hi])
    f[f"{prefix}.op.Lhand_visible.frac"] = nfrac(s.hand_vis_l[lo:hi])
    return f


# ── Block C: target child, vocal activity (12) ────────────────────────────────

def block_c(s: NodeSeries, alo: int, ahi: int, prefix: str) -> dict:
    f = {}
    keys = ["valid_frac.mean", "valid_frac.std", "active.frac", "active_runs_per_min",
            "active_run_mean_s", "active_run_max_s", "silence_run_max_s",
            "activity_transition_rate", "energy.mean", "energy.std", "energy.p90",
            "energy.mean_abs_delta"]
    if not s.has_audio or ahi <= alo:
        return {f"{prefix}.eg.{k}": np.nan for k in keys}
    act = s.active[alo:ahi]
    dur_s = (ahi - alo) / C.AUDIO_FPS
    runs = runs_of(act) / C.AUDIO_FPS
    sil = runs_of(~act) / C.AUDIO_FPS
    f[f"{prefix}.eg.valid_frac.mean"] = nmean(s.valid_frac[alo:ahi])
    f[f"{prefix}.eg.valid_frac.std"] = nstd(s.valid_frac[alo:ahi])
    f[f"{prefix}.eg.active.frac"] = nfrac(act)
    f[f"{prefix}.eg.active_runs_per_min"] = float(len(runs) / dur_s * 60.0)
    f[f"{prefix}.eg.active_run_mean_s"] = float(runs.mean()) if runs.size else 0.0
    f[f"{prefix}.eg.active_run_max_s"] = float(runs.max()) if runs.size else 0.0
    f[f"{prefix}.eg.silence_run_max_s"] = float(sil.max()) if sil.size else 0.0
    f[f"{prefix}.eg.activity_transition_rate"] = (float(np.mean(act[:-1] != act[1:]))
                                                  if len(act) > 1 else np.nan)
    en = s.energy[alo:ahi]
    f[f"{prefix}.eg.energy.mean"] = nmean(en)
    f[f"{prefix}.eg.energy.std"] = nstd(en)
    f[f"{prefix}.eg.energy.p90"] = npct(en, 90)
    f[f"{prefix}.eg.energy.mean_abs_delta"] = nmean_abs_diff(en)
    return f


# ── Block D: dyadic, target <-> partner (34) ──────────────────────────────────

_D5_SCALARS = ["of2.head_trans_speed.mean", "op.hand_motion.mean", "of2.AU12_r.mean",
               "of2.gaze_angle_x.std", "of2.pose_Tz.mean", "of2.success.frac",
               "op.detect_rate", "op.RWrist.motion"]


def _core_scalars(s: NodeSeries, lo: int, hi: int) -> dict:
    """The 8 visual scalars Block D contrasts between the two nodes."""
    return {
        "of2.head_trans_speed.mean": nmean(s.head_trans_speed[lo:hi]),
        "op.hand_motion.mean": nmean(s.hand_motion[lo:hi]),
        "of2.AU12_r.mean": nmean(s.au_r[lo:hi, C.AU_R_INDEX["AU12"]]),
        "of2.gaze_angle_x.std": nstd(s.gaze_x[lo:hi]),
        "of2.pose_Tz.mean": nmean(s.pose_t[lo:hi, 2]),
        "of2.success.frac": nfrac(s.of_ok[lo:hi]),
        "op.detect_rate": nfrac(s.kp_valid[lo:hi]),
        "op.RWrist.motion": nmean(s.kp_speed[lo:hi, C.KP["RWrist"]]),
    }


def block_d(tgt: NodeSeries, par: NodeSeries, lo: int, hi: int) -> dict:
    """Dyadic coupling between the target child and the partner node -- vision only.

    Parten's categories are relational by construction (solitary/parallel vs
    associative/cooperative is a property of the coupling, not of one child), so this block is the
    one a per-child feature vector could not express. Speech-based coupling is deliberately absent:
    the microphone is shared, so any "who spoke when" feature would be an artifact.
    """
    f = {}
    # D2/D3 behavioural synchrony, and who leads.
    f["dy.r.head_motion"] = corr(tgt.head_trans_speed[lo:hi], par.head_trans_speed[lo:hi])
    f["dy.r.hand_motion"] = corr(tgt.hand_motion[lo:hi], par.hand_motion[lo:hi])
    f["dy.r.AU12"] = corr(tgt.au_r[lo:hi, C.AU_R_INDEX["AU12"]],
                          par.au_r[lo:hi, C.AU_R_INDEX["AU12"]])
    f["dy.r.gaze_angle_x"] = corr(tgt.gaze_x[lo:hi], par.gaze_x[lo:hi])
    f["dy.lag.head_motion"] = best_lag(tgt.head_trans_speed[lo:hi], par.head_trans_speed[lo:hi],
                                       C.VIDEO_FPS)
    # D4 gaze coupling -- calibration-free mutual-gaze proxy.
    gx_t, gx_p = tgt.gaze_x[lo:hi], par.gaze_x[lo:hi]
    both_ok = np.isfinite(gx_t) & np.isfinite(gx_p)
    f["dy.abs_gaze_x_diff.mean"] = nmean(np.abs(gx_t - gx_p))
    f["dy.gaze_x_opposite_sign.frac"] = (float(np.mean(np.sign(gx_t[both_ok]) !=
                                                        np.sign(gx_p[both_ok])))
                                         if both_ok.sum() >= MIN_VALID else np.nan)
    f["dy.both_face_tracked.frac"] = nfrac(tgt.of_ok[lo:hi] & par.of_ok[lo:hi])
    # D5 contrasts: |diff| and mean let one split express "both do X" vs "only one does X".
    ts, ps = _core_scalars(tgt, lo, hi), _core_scalars(par, lo, hi)
    for k in _D5_SCALARS:
        f[f"dy.absdiff.{k}"] = abs(ts[k] - ps[k])
        f[f"dy.mean.{k}"] = (ts[k] + ps[k]) / 2.0
    # D6 proximity.
    dz = np.abs(tgt.pose_t[lo:hi, 2] - par.pose_t[lo:hi, 2])
    f["dy.abs_Tz_diff.mean"] = nmean(dz)
    f["dy.abs_Tz_diff.std"] = nstd(dz)
    return f


# ── Block E: metadata (5) ─────────────────────────────────────────────────────

def read_annotation_scalar(path: str):
    """Value field of a single-line PInSoRo scalar annotation (``start;end;value;confidence``)."""
    if not os.path.exists(path):
        return np.nan
    parts = open(path).read().strip().split(";")
    try:
        return float(parts[2])
    except (IndexError, ValueError):
        return np.nan


def block_e(feature_dir: str, role: str, domain: str, partner_is_robot: bool,
            t_center: float, duration_s: float) -> dict:
    return {
        "meta.domain_cc": 1.0 if domain == "cc" else 0.0,
        "meta.age": read_annotation_scalar(os.path.join(feature_dir, f"{role}.age.annotation.csv")),
        "meta.gender": read_annotation_scalar(
            os.path.join(feature_dir, f"{role}.gender.annotation.csv")),
        "meta.session_elapsed_frac": float(t_center / duration_s) if duration_s > 0 else np.nan,
        "meta.partner_is_robot": 1.0 if partner_is_robot else 0.0,
    }


# ── Block F: env node (10) ────────────────────────────────────────────────────

def block_f(env: NodeSeries | None, lo: int, hi: int) -> dict:
    keys = ["op.detect_rate", "op.motion", "of2.success.frac", "of2.confidence.mean",
            "of2.AU12_r.mean", "of2.AU_c_any.frac", "op.pose_score.mean", "of2.gaze_angle_x.std",
            "of2.head_trans_speed.mean", "op.n_kp.mean"]
    if env is None:
        return {f"env.{k}": np.nan for k in keys}
    auc = env.au_c[lo:hi]
    return {
        "env.op.detect_rate": nfrac(env.kp_valid[lo:hi]),
        "env.op.motion": nmean(np.nanmean(env.kp_speed[lo:hi], axis=1)),
        "env.of2.success.frac": nfrac(env.of_ok[lo:hi]),
        "env.of2.confidence.mean": nmean(env.confidence[lo:hi]),
        "env.of2.AU12_r.mean": nmean(env.au_r[lo:hi, C.AU_R_INDEX["AU12"]]),
        "env.of2.AU_c_any.frac": (float(np.nanmean(auc.max(axis=1) > 0.5))
                                  if _n(auc[:, 0]) >= MIN_VALID else np.nan),
        "env.op.pose_score.mean": nmean(env.pose_score[lo:hi]),
        "env.of2.gaze_angle_x.std": nstd(env.gaze_x[lo:hi]),
        "env.of2.head_trans_speed.mean": nmean(env.head_trans_speed[lo:hi]),
        "env.op.n_kp.mean": float(np.mean(env.kp_valid[lo:hi].sum(axis=1))) if hi > lo else np.nan,
    }


# ── Block G: opaque audio channels (132) ──────────────────────────────────────
# Means for all 88 channels plus standard deviations for the 44 continuous ones (the three-level
# channels carry almost no within-window spread). Named by index, not by any acoustic descriptor --
# see the audio note in :mod:`columns`.

def block_g(s: NodeSeries, alo: int, ahi: int, prefix: str) -> dict:
    f = {}
    if not s.has_audio or ahi <= alo:
        for j in range(C.EGEMAPS_DIM):
            f[f"{prefix}.egc{j:02d}.mean"] = np.nan
        for j in range(C.EG_N_PAIR):
            f[f"{prefix}.egc{2 * j + 1:02d}.std"] = np.nan
        return f
    seg = s.eg[alo:ahi]
    ok = np.isfinite(seg).sum(axis=0)
    mu = np.where(ok >= MIN_VALID, np.nanmean(seg, axis=0), np.nan)
    sd = np.where(ok >= MIN_VALID, np.nanstd(seg, axis=0), np.nan)
    for j in range(C.EGEMAPS_DIM):
        f[f"{prefix}.egc{j:02d}.mean"] = float(mu[j])
    for j in range(C.EG_N_PAIR):
        c = 2 * j + 1
        f[f"{prefix}.egc{c:02d}.std"] = float(sd[c])
    return f


# ── Block H: multi-scale context (72) ─────────────────────────────────────────
# Justified by the run-length statistics: with mean label runs of 44-237 s, the class at time t is
# mostly set by the minute around t, not by the 10 s at t. This is how a window-statistics model
# recovers the long-range context the deep model gets from its temporal transformers.

H_SCALARS = [
    "room.eg.active.frac", "room.eg.energy.mean", "room.eg.valid_frac.mean",
    "of2.gaze_angle_x.std", "of2.gaze_angle_x.sign_change_rate", "of2.gaze_angle_y.mean",
    "of2.pose_Tz.mean", "of2.pose_Ry.mean", "of2.pose_Ry.std",
    "of2.head_trans_speed.mean", "of2.head_rot_speed.mean",
    "of2.AU06_r.mean", "of2.AU12_r.mean", "of2.AU25_r.mean", "of2.AU45_r.mean",
    "of2.success.frac", "op.detect_rate", "op.RWrist.motion", "op.LWrist.motion",
    "op.RWrist_above_neck.frac", "op.shoulder_angle.mean", "op.torso_scale.mean",
    "dy.r.head_motion", "dy.r.hand_motion",
]


def context_scalars(tgt: NodeSeries, par: NodeSeries, room: NodeSeries,
                    lo: int, hi: int, alo: int, ahi: int) -> dict:
    """The 24 scalars of :data:`H_SCALARS`, evaluated over an arbitrary frame span."""
    out = {}
    has_a = room.has_audio and ahi > alo
    out["room.eg.active.frac"] = nfrac(room.active[alo:ahi]) if has_a else np.nan
    out["room.eg.energy.mean"] = nmean(room.energy[alo:ahi]) if has_a else np.nan
    out["room.eg.valid_frac.mean"] = nmean(room.valid_frac[alo:ahi]) if has_a else np.nan
    out["of2.gaze_angle_x.std"] = nstd(tgt.gaze_x[lo:hi])
    out["of2.gaze_angle_x.sign_change_rate"] = sign_change_rate(tgt.gaze_x[lo:hi])
    out["of2.gaze_angle_y.mean"] = nmean(tgt.gaze_y[lo:hi])
    out["of2.pose_Tz.mean"] = nmean(tgt.pose_t[lo:hi, 2])
    out["of2.pose_Ry.mean"] = nmean(tgt.pose_r[lo:hi, 1])
    out["of2.pose_Ry.std"] = nstd(tgt.pose_r[lo:hi, 1])
    out["of2.head_trans_speed.mean"] = nmean(tgt.head_trans_speed[lo:hi])
    out["of2.head_rot_speed.mean"] = nmean(tgt.head_rot_speed[lo:hi])
    for a in ("AU06", "AU12", "AU25", "AU45"):
        out[f"of2.{a}_r.mean"] = nmean(tgt.au_r[lo:hi, C.AU_R_INDEX[a]])
    out["of2.success.frac"] = nfrac(tgt.of_ok[lo:hi])
    out["op.detect_rate"] = nfrac(tgt.kp_valid[lo:hi])
    out["op.RWrist.motion"] = nmean(tgt.kp_speed[lo:hi, C.KP["RWrist"]])
    out["op.LWrist.motion"] = nmean(tgt.kp_speed[lo:hi, C.KP["LWrist"]])
    h = -(tgt.yn[lo:hi, C.KP["RWrist"]] - tgt.yn[lo:hi, C.KP["neck"]])
    out["op.RWrist_above_neck.frac"] = float(np.nanmean(h > 0)) if _n(h) >= MIN_VALID else np.nan
    out["op.shoulder_angle.mean"] = nmean(tgt.shoulder_angle[lo:hi])
    out["op.torso_scale.mean"] = nmean(tgt.torso_scale[lo:hi])
    out["dy.r.head_motion"] = corr(tgt.head_trans_speed[lo:hi], par.head_trans_speed[lo:hi])
    out["dy.r.hand_motion"] = corr(tgt.hand_motion[lo:hi], par.hand_motion[lo:hi])
    return out
