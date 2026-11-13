"""Verified stream column layouts for the PInSoRo raw ``.stream`` files.

Every constant here was checked empirically against ``pinsoro/train-cc/002`` (run
``python -m pinsoro_model.columns --verify`` to re-check on any session):

* ``openface2`` (dim 714) follows the standard OpenFace 2.0 column order.
* ``openpose``  (dim 139) is COCO-18 ``(x, y, conf)`` triplets in cols 0-53, an overall pose
  detection score in col 54 (r = 0.98 with the summed keypoint confidence; ``-1`` exactly on
  frames where no body keypoint is detected), and 42 hand keypoints as ``(x, y)`` pairs (no
  confidence) in cols 55-138. Undetected values carry the ``-1`` sentinel; the hand block is
  93-99% missing.
* ``audio.egemapsv2`` (dim 88) is the eGeMAPSv02 functional set in openSMILE order, already
  standardized by the extraction pipeline (observed range ~+-8, so absolute loudness is not
  recoverable), with 7-10% of frames carrying the ``FEATURE_CLIP`` sentinel where the voiced
  functionals are undefined.
"""

from __future__ import annotations

import numpy as np

# ── OpenFace 2.0 (714 columns) ────────────────────────────────────────────────
OF2_DIM = 714
OF2_FRAME = 0
OF2_FACE_ID = 1          # all-zero dead channel
OF2_TIMESTAMP = 2
OF2_CONFIDENCE = 3
OF2_SUCCESS = 4
OF2_GAZE_VEC = slice(5, 11)      # gaze_0_x/y/z, gaze_1_x/y/z
OF2_GAZE_ANGLE_X = 11
OF2_GAZE_ANGLE_Y = 12
OF2_EYE_LMK_2D = slice(13, 125)
OF2_EYE_LMK_3D = slice(125, 293)
OF2_POSE_T = slice(293, 296)     # Tx, Ty, Tz  (mm, camera frame)
OF2_POSE_R = slice(296, 299)     # Rx, Ry, Rz  (radians)
OF2_LMK_2D = slice(299, 435)
OF2_LMK_3D = slice(435, 639)
OF2_PDM = slice(639, 679)
OF2_AU_R = slice(679, 696)       # 17 intensities, 0-5
OF2_AU_C = slice(696, 714)       # 18 presence flags, {0, 1}

AU_R_NAMES = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
              "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]
AU_C_NAMES = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
              "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU28", "AU45"]
# Action units carrying the most social/engagement signal: cheek raiser + lip corner puller
# (Duchenne smile), lips part + jaw drop (talking / mouth open), blink (attention & arousal).
SALIENT_AU_R = ["AU06", "AU12", "AU25", "AU26", "AU45"]
AU_R_INDEX = {n: i for i, n in enumerate(AU_R_NAMES)}

OF2_TRACK_CONF_MIN = 0.2         # below this the OpenFace pose/gaze estimate is not trustworthy

# Plausible head distance from the child's own camera, in mm. OpenFace can report success = 1 and
# confidence = 0.98 while emitting an impossible depth: in cc/065/yellow, frame 11967 reads
# 55,773,360,128 mm (~55,773 km), frame 11964 reads 4,883,400 mm and frame 11960 reads -14,898 mm
# (44 such frames out of 65,649 accepted). Legitimate accepted frames in that same session top out
# at 4,992 mm and train's largest 10 s window mean is 3,687 mm, so anything outside this range is
# tracker failure. A frame failing it is treated as untracked: if the 3D pose estimate collapsed,
# the gaze direction derived from it is not trustworthy either.
# streams.FEATURE_CLIP (1e8) is never applied to openface2, and would not suffice anyway -- it
# catches the 5.58e10 frame but passes the 4.9e6 and 3.0e4 ones.
OF2_POSE_TZ = 295                # the Tz column inside OF2_POSE_T
OF2_TZ_MIN_MM, OF2_TZ_MAX_MM = 100.0, 5000.0

# ── OpenPose (139 columns) ────────────────────────────────────────────────────
OP_DIM = 139
OP_N_BODY_KP = 18                # COCO-18
OP_BODY_END = 54                 # cols 0..53 are (x, y, conf) triplets
OP_POSE_SCORE_COL = 54           # overall detection score (~2x the summed keypoint confidence)
OP_HAND_START = 55               # cols 55..138 = 42 hand keypoints as (x, y) pairs
OP_SENTINEL = -1.0

KP_NAMES = ["nose", "neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
            "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]
KP = {n: i for i, n in enumerate(KP_NAMES)}
# Upper body only: seated children never expose hips/knees/ankles (observed ~100% missing).
CORE_KP = ["nose", "neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]
# Hand keypoints split evenly: first 21 = right hand, last 21 = left hand.
OP_HAND_R = slice(55, 97)
OP_HAND_L = slice(97, 139)


def op_xyc(op: np.ndarray):
    """Split an openpose array ``(T, 139)`` into ``(x, y, conf)`` of shape ``(T, 18)`` each.

    The ``-1`` sentinel is preserved; use :func:`op_valid` to mask it.
    """
    body = op[:, :OP_BODY_END].reshape(len(op), OP_N_BODY_KP, 3)
    return body[:, :, 0], body[:, :, 1], body[:, :, 2]


def op_valid(x, y, c):
    """``(T, 18)`` bool mask of detected body keypoints (all three channels above the sentinel)."""
    return (c > 0) & (x > OP_SENTINEL) & (y > OP_SENTINEL)


# ── Audio: "egemapsv2" stream (88 columns) ────────────────────────────────────
# This stream is NOT the canonical eGeMAPSv02 functional vector, despite the filename. Verified on
# 6 (session, role) pairs spanning train/val/test and cc/cr, the 88 columns are 44 interleaved
# pairs with two sharply different distributions:
#   * even columns (0, 2, ... 86): three levels only, {-2, 0, +2} (a clipped/quantised code), and
#     they carry essentially all of the sentinel missingness (5-45% of frames, varying per frame);
#   * odd columns (1, 3, ... 87): continuous, thousands of distinct values, ~0.2% sentinel.
# The pairing does not reproduce any documented eGeMAPS ordering (corr between a pair is ~0), so
# the individual channels are treated as OPAQUE numeric features -- no channel is claimed to be
# "loudness" or "F0". Two structural signals are used instead, both label-free and available at
# test time:
#   * the per-frame fraction of defined channels (functionals are undefined when the analysis
#     window holds no voiced speech), and
#   * the mean |value| over the continuous channels, an acoustic-magnitude proxy that is
#     thresholded against a session-level quantile pooled over both children (see
#     features.set_activity_threshold).
EGEMAPS_DIM = 88
EG_QUANT = slice(0, EGEMAPS_DIM, 2)    # 44 three-level channels, most of the missingness
EG_CONT = slice(1, EGEMAPS_DIM, 2)     # 44 continuous channels
EG_N_PAIR = EGEMAPS_DIM // 2

# ── Sample rates ──────────────────────────────────────────────────────────────
VIDEO_FPS = 30.0     # openface2 / openpose, and the native PInSoRo annotation grid
AUDIO_FPS = 25.0     # audio.egemapsv2


def _verify(feature_dir: str, role: str = "purple") -> None:
    """Assert the documented layouts hold for one session (see module docstring)."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .streams import read_stream, FEATURE_CLIP

    of2, sr = read_stream(os.path.join(feature_dir, f"{role}.openface2.stream"))
    assert of2.shape[1] == OF2_DIM, of2.shape
    assert sr == VIDEO_FPS, sr
    assert set(np.unique(of2[:, OF2_SUCCESS])) <= {0.0, 1.0}, "col 4 is not binary success"
    assert set(np.unique(of2[:, OF2_AU_C])) <= {0.0, 1.0}, "AU_c block is not binary"
    assert 0.0 <= of2[:, OF2_AU_R].min() and of2[:, OF2_AU_R].max() <= 5.0, "AU_r out of 0-5"
    assert np.all((of2[:, OF2_CONFIDENCE] >= 0) & (of2[:, OF2_CONFIDENCE] <= 1)), "confidence"
    assert np.allclose(of2[:, OF2_FRAME], np.arange(1, len(of2) + 1)), "col 0 is not a frame counter"

    op, sr = read_stream(os.path.join(feature_dir, f"{role}.openpose.stream"))
    assert op.shape[1] == OP_DIM, op.shape
    assert sr == VIDEO_FPS, sr
    x, y, c = op_xyc(op)
    ok = c > OP_SENTINEL
    assert c[ok].max() <= 1.2, f"conf channel far above 1: {c[ok].max()}"
    score_missing = op[:, OP_POSE_SCORE_COL] <= OP_SENTINEL
    assert not ok[score_missing].any(), "col 54 is -1 on a frame that has a detected keypoint"
    tail = op[:, OP_HAND_START:]
    assert tail[tail > OP_SENTINEL].max() > 100, "hand block is not in pixel units"

    eg, sr = read_stream(os.path.join(feature_dir, f"{role}.audio.egemapsv2.stream"))
    assert eg.shape[1] == EGEMAPS_DIM, eg.shape
    assert sr == AUDIO_FPS, sr
    bad = ~np.isfinite(eg) | (np.abs(eg) > FEATURE_CLIP)
    assert 0.0 < bad.mean() < 0.5, f"unexpected sentinel rate {bad.mean():.3f}"
    assert np.abs(eg[~bad]).max() <= 20.0, "audio stream does not look standardized"
    val = np.where(bad, np.nan, eg)
    nuniq = np.array([len(np.unique(val[np.isfinite(val[:, c]), c])) for c in range(EGEMAPS_DIM)])
    assert (nuniq[EG_QUANT] <= 5).all(), "even audio channels are not the 3-level code"
    assert (nuniq[EG_CONT] > 100).all(), "odd audio channels are not continuous"
    assert bad[:, EG_QUANT].mean() > 10 * bad[:, EG_CONT].mean(), \
        "missingness is not concentrated in the even audio channels"
    print(f"  OK {os.path.basename(feature_dir)}/{role}: "
          f"of2 {of2.shape} op {op.shape} eg {eg.shape} sentinel {bad.mean():.1%}")


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser(description="verify the documented column layouts")
    ap.add_argument("--root", default="/path/to/pinsoro")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="paths relative to --root, e.g. train-cc/002 (default: first 3 of train-cc)")
    args = ap.parse_args()

    sessions = args.sessions
    if not sessions:
        sub = os.path.join(args.root, "train-cc")
        sessions = [f"train-cc/{s}" for s in sorted(os.listdir(sub))[:3]]
    print(f"verifying {len(sessions)} session(s)")
    for rel in sessions:
        _verify(os.path.join(args.root, rel))
    print("all column layout assertions passed")


if __name__ == "__main__":
    main()
