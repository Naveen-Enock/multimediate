"""The V2 feature set: what the model sees and why. Single source of truth.

Each entry is ``(name, construct, computed_as, why, cc_only)``. ``extract_v2.py`` emits exactly these
columns in this order; ``evaluate.py`` feeds ``V2_CC`` to child-child cells and ``V2_CR`` to
child-robot cells.

Selection rules:

* **Construct** -- every feature maps to something in the coding scheme: attention, head
  orientation, hand activity, affect, observability, dyadic coupling, context.
* **Meaningful frame** -- angles rather than raw vector components; positions relative to the neck
  rather than absolute camera coordinates; seat-mirrored signals role-aligned (horizontal gaze and
  head yaw have opposite sign for purple and yellow in 27/27 cc sessions).
* **No duplicates** -- nothing with an \\|r\\| > 0.95 twin.
* **Behaviour, not identity** -- no age, gender, session position, domain flag, room audio (shared
  microphone), env camera (identical for both children) or camera-placement measure.
* **Nameable** -- no opaque channels.

Every value is computed from the per-frame ``openface2`` and ``openpose`` streams shipped in each
PInSoRo session folder. Labels are targets only, never inputs.

cr cells drop the dyadic block: the robot's face is tracked on 0.8-6.6% of frames and its body on
0.3-32%, so partner values computed from it would be tracker noise.
"""

from __future__ import annotations

V2_SUFFIX = "v2_feats.npz"

# Seat mirroring: flipping purple makes "+" the same direction for both seats. In every cr session
# the child is purple and the robot yellow (23/23).
ROLE_SIGN = {"purple": -1.0, "yellow": 1.0}

# Signals recomputed over the 60 s centred on each window.
CONTEXT_CORE = [
    "gaze_x_aligned.mean", "gaze_y.mean", "head_yaw_aligned.mean", "head_dist.mean",
    "hand_motion.mean", "AU12.mean", "AU25.std", "r.hand_motion",
]

_PER_CHILD = [
    # attention
    ("gaze_x_aligned.mean", "attention", "mean of role-aligned horizontal gaze angle",
     "Where the child looks side to side, with seat mirroring removed"),
    ("gaze_x.std", "attention", "std of horizontal gaze angle", "Scanning around vs fixating"),
    ("gaze_y.mean", "attention", "mean of vertical gaze angle", "Looking down at the table vs up"),
    ("gaze_y.std", "attention", "std of vertical gaze angle", "How much the child looks up and down"),
    ("gaze_shift_rate", "attention",
     "fraction of consecutive tracked frames where horizontal gaze changes side",
     "Switching attention back and forth"),
    # head
    ("head_yaw_aligned.mean", "head", "mean of role-aligned head yaw",
     "Which way the head is turned, seat mirroring removed"),
    ("head_yaw.std", "head", "std of head yaw", "How much the head turns"),
    ("head_pitch.mean", "head", "mean of head pitch", "Head tilted down over the task vs up"),
    ("head_dist.mean", "head", "mean head distance from the child's own camera (mm)",
     "Leaning forward vs sitting back"),
    ("head_dist.slope", "head", "least-squares slope of head distance per second",
     "Moving closer or pulling away"),
    ("head_rot_speed.mean", "head", "mean frame-to-frame change in head rotation",
     "Overall head activity"),
    # hands & posture
    ("hand_motion.mean", "hands",
     "mean frame-to-frame motion of the two wrists (neck-relative, shoulder widths)",
     "Handling toys, the core of task engagement"),
    ("RWrist.motion", "hands", "mean right-wrist frame-to-frame motion", "Right-hand activity"),
    ("LWrist.motion", "hands", "mean left-wrist frame-to-frame motion", "Left-hand activity"),
    ("Rwrist_height.mean", "hands", "mean right-wrist height above the neck (shoulder widths)",
     "Hand on the table vs raised"),
    ("Lwrist_height.mean", "hands", "mean left-wrist height above the neck (shoulder widths)",
     "Same, left"),
    ("nose_height.mean", "posture", "mean nose height above the neck (shoulder widths)",
     "Head dropped forward over the task"),
    # affect
    ("AU06.mean", "affect", "mean cheek-raiser (AU06) intensity, 0-5", "Part of a genuine smile"),
    ("AU12.mean", "affect", "mean lip-corner-puller (AU12) intensity", "Smiling"),
    ("AU12.frac_smiling", "affect", "fraction of window frames with AU12 > 1",
     "Share of time clearly smiling"),
    ("AU04.mean", "affect", "mean brow-lowerer (AU04) intensity", "Concentrating"),
    ("AU25.mean", "affect", "mean lips-part (AU25) intensity", "Mouth open"),
    ("AU25.std", "affect", "std of lips-part (AU25) intensity",
     "Mouth opening and closing (not claimed as speech)"),
    # observability
    ("face_tracked.frac", "observability", "fraction of frames the face is tracked",
     "Bent over the table or turned away loses the face"),
    ("body_detected.frac", "observability", "fraction of body keypoints detected",
     "Leaning out of view"),
]

_DYADIC = [
    ("r.head_motion", "dyadic", "Pearson r of the two children's head-movement speed",
     "Do they move together"),
    ("r.hand_motion", "dyadic", "Pearson r of the two children's hand motion",
     "Joint manipulation"),
    ("r.smile", "dyadic", "Pearson r of the two children's AU12", "Shared smiling"),
    ("r.mouth", "dyadic", "Pearson r of the two children's AU25", "Coupled mouth activity"),
    ("r.gaze_x", "dyadic", "Pearson r of the two children's horizontal gaze",
     "Shifting gaze at the same moments"),
    ("partner.hand_motion.mean", "dyadic", "hand_motion.mean computed on the partner",
     "Partner active while the target is still = onlooker; both active = parallel"),
    ("partner.smile.mean", "dyadic", "AU12.mean computed on the partner",
     "Partner's positive affect"),
    ("partner.gaze_x_aligned.mean", "dyadic", "gaze_x_aligned.mean computed on the partner",
     "Where the partner is looking"),
]

FEATURES = ([(n, c, how, why, False) for n, c, how, why in _PER_CHILD]
            + [(n, c, how, why, True) for n, c, how, why in _DYADIC])
for _core in CONTEXT_CORE:
    _cc = _core == "r.hand_motion"
    FEATURES.append((f"ctx60.{_core}", "context", f"{_core} over the 60 s centred on the window",
                     "Labels last about a minute; the surrounding minute describes the state", _cc))
    FEATURES.append((f"ctxd.{_core}", "context", f"ctx60 minus the 10 s value of {_core}",
                     "Is this moment typical of the surrounding minute", _cc))

V2_ALL = [f[0] for f in FEATURES]
V2_CC = list(V2_ALL)
V2_CR = [f[0] for f in FEATURES if not f[4]]
assert len(V2_ALL) == len(set(V2_ALL)) == 49, len(V2_ALL)
assert len(V2_CR) == 39, len(V2_CR)

# How each V2 column relates to a column of the original 442-feature tables, for the
# re-extraction spot-check: factor 1 = identical, "sign" = x role sign, -1 = negated.
# The 10 features absent here are new.
EQUIV = {
    "gaze_x_aligned.mean": ("tgt.of2.gaze_angle_x.mean", "sign"),
    "gaze_x.std": ("tgt.of2.gaze_angle_x.std", 1),
    "gaze_y.mean": ("tgt.of2.gaze_angle_y.mean", 1),
    "gaze_y.std": ("tgt.of2.gaze_angle_y.std", 1),
    "gaze_shift_rate": ("tgt.of2.gaze_angle_x.sign_change_rate", 1),
    "head_yaw_aligned.mean": ("tgt.of2.pose_Ry.mean", "sign"),
    "head_yaw.std": ("tgt.of2.pose_Ry.std", 1),
    "head_pitch.mean": ("tgt.of2.pose_Rx.mean", 1),
    "head_dist.mean": ("tgt.of2.pose_Tz.mean", 1),
    "head_dist.slope": ("tgt.of2.pose_Tz.slope", 1),
    "head_rot_speed.mean": ("tgt.of2.head_rot_speed.mean", 1),
    "hand_motion.mean": ("tgt.op.hand_motion.mean", 1),
    "RWrist.motion": ("tgt.op.RWrist.motion", 1),
    "LWrist.motion": ("tgt.op.LWrist.motion", 1),
    "Rwrist_height.mean": ("tgt.op.Rwrist_height.mean", 1),
    "Lwrist_height.mean": ("tgt.op.Lwrist_height.mean", 1),
    "nose_height.mean": ("tgt.op.nose.yn.mean", -1),
    "AU06.mean": ("tgt.of2.AU06_r.mean", 1),
    "AU12.mean": ("tgt.of2.AU12_r.mean", 1),
    "AU12.frac_smiling": ("tgt.of2.AU12_r.frac_gt_1", 1),
    "AU04.mean": ("tgt.of2.AU04_r.mean", 1),
    "AU25.mean": ("tgt.of2.AU25_r.mean", 1),
    "AU25.std": ("tgt.of2.AU25_r.std", 1),
    "face_tracked.frac": ("tgt.of2.success.frac", 1),
    "body_detected.frac": ("tgt.op.detect_rate", 1),
    "r.head_motion": ("dy.r.head_motion", 1),
    "r.hand_motion": ("dy.r.hand_motion", 1),
    "r.smile": ("dy.r.AU12", 1),
    "r.gaze_x": ("dy.r.gaze_angle_x", 1),
    "ctx60.gaze_y.mean": ("ctx60.of2.gaze_angle_y.mean", 1),
    "ctxd.gaze_y.mean": ("ctxd.of2.gaze_angle_y.mean", 1),
    "ctx60.head_yaw_aligned.mean": ("ctx60.of2.pose_Ry.mean", "sign"),
    "ctxd.head_yaw_aligned.mean": ("ctxd.of2.pose_Ry.mean", "sign"),
    "ctx60.head_dist.mean": ("ctx60.of2.pose_Tz.mean", 1),
    "ctxd.head_dist.mean": ("ctxd.of2.pose_Tz.mean", 1),
    "ctx60.AU12.mean": ("ctx60.of2.AU12_r.mean", 1),
    "ctxd.AU12.mean": ("ctxd.of2.AU12_r.mean", 1),
    "ctx60.r.hand_motion": ("ctx60.dy.r.hand_motion", 1),
    "ctxd.r.hand_motion": ("ctxd.dy.r.hand_motion", 1),
}
assert set(EQUIV) <= set(V2_ALL)
