"""Synthetic calibration session generator, used by both the pytest
accuracy-regression test (test_calibration.py) and the standalone
scripts/bench_calibration.py benchmark.

A real calibration session needs a webcam and a human in front of it, which
a test suite can't provide -- so this instead builds feature vectors from a
KNOWN, fixed ground-truth linear mapping (screen position = a fixed linear
function of the 10-dim calibration feature vector, plus realistic noise),
the same 4x4 training grid + 5-point held-out validation split
scripts/calibrate.py itself uses, and the same per-point sample count and
group-id tagging GazeCalibrator.fit() expects.

Because the ground truth is linear and GazeCalibrator is a degree-1 (linear)
Ridge model by design (see calibration.py's own docstring), a correctly
behaving calibration pipeline should recover it closely. If a future change
to feature scaling, the group-aware CV, the regularization range, or the
gaze/head-pose feature math degrades the fit, the reported validation error
on this fixed synthetic dataset will measurably increase -- that's the
"did my change help or hurt" signal this exists to give, independent of any
single real (noisy, non-reproducible) human calibration run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.gaze.calibration import build_feature_vector

# Same training grid and held-out validation points scripts/calibrate.py
# itself uses -- kept as literal, independent constants here (rather than
# importing from the script) so this fixture stays stable even if
# calibrate.py's own grid is later tuned for live-session reasons unrelated
# to this benchmark.
_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
TRAIN_GRID_FRACTIONS = [(fx, fy) for fy in _GRID_STEPS for fx in _GRID_STEPS]
VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]

SAMPLES_PER_POINT = 15

# Ground-truth generative model, in the same [mean_x, mean_y, diff_x,
# diff_y, yaw, pitch, roll, tx, ty, tz] feature layout build_feature_vector
# produces. mean_offset (the eye-in-head signal) is the dominant driver, by
# construction -- matching calibration.py's own description of it as "the
# robust, well-conditioned conjugate signal" -- with small, realistic
# secondary contributions from head pose and eye asymmetry, the same way
# real head motion legitimately perturbs the true mapping a little.
_EYE_OFFSET_RANGE = 0.30  # mean_x/mean_y span this much total across the screen
_HEAD_POSE_COEF_PER_DEG = 3.0   # px of screen position per degree of yaw/pitch/roll jitter
_TRANSLATION_COEF_PER_MM = 0.6  # px of screen position per mm of head translation jitter
_DIFF_COEF = 400.0              # px of screen position per unit of per-eye asymmetry

_RNG_MEASUREMENT_NOISE_STD = 0.01   # std dev added directly to mean_offset per sample
_HEAD_YAW_PITCH_STD_DEG = 3.0
_HEAD_ROLL_STD_DEG = 2.0
_HEAD_TRANSLATION_XY_STD_MM = 5.0
_HEAD_TRANSLATION_Z_STD_MM = 10.0
_DIFF_OFFSET_STD = 0.02
_BASELINE_TRANSLATION_MM = (0.0, 0.0, 400.0)  # matches head_pose.py's _DEFAULT_TVEC

# A real eye-in-head-offset -> screen-position relationship isn't perfectly
# linear (eyeball/lens geometry curves it a little, especially toward the
# edges), and GazeCalibrator is deliberately only degree-1 (see its own
# docstring: a safer bet than degree=2 on this little calibration data, "a
# bounded, somewhat less accurate linear fit"). Without *some* irreducible,
# unlearnable error, this synthetic benchmark would let a linear model fit
# it to ~0px and the test would be unable to tell a healthy calibration
# pipeline from a subtly broken one -- everything would just read ~perfect.
# This cubic term (small, odd-symmetric, so it doesn't bias the mean) and a
# touch of independent target noise (real-world ground-truth imprecision
# no feature could ever explain) give the benchmark a believable, nonzero
# floor to measure real regressions against.
_NONLINEARITY_COEF = 8.0
_TARGET_NOISE_STD_PX = 4.0


@dataclass
class SyntheticSession:
    screen_w: int
    screen_h: int
    train_features: np.ndarray
    train_targets: np.ndarray
    train_groups: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray


def _ground_truth_screen_position(
    mean_offset: np.ndarray, diff_offset: np.ndarray, head_pose_deg: np.ndarray, translation_mm: np.ndarray,
    screen_w: int, screen_h: int, rng: np.random.Generator,
) -> np.ndarray:
    cx, cy = screen_w / 2.0, screen_h / 2.0
    sx = screen_w / _EYE_OFFSET_RANGE
    sy = screen_h / _EYE_OFFSET_RANGE
    translation_jitter = translation_mm - np.array(_BASELINE_TRANSLATION_MM)
    x = (
        cx + sx * mean_offset[0]
        + _NONLINEARITY_COEF * screen_w * mean_offset[0] ** 3
        + _DIFF_COEF * diff_offset[0]
        + _HEAD_POSE_COEF_PER_DEG * head_pose_deg[0]  # yaw
        + _TRANSLATION_COEF_PER_MM * translation_jitter[0]
        + rng.normal(scale=_TARGET_NOISE_STD_PX)
    )
    y = (
        cy + sy * mean_offset[1]
        + _NONLINEARITY_COEF * screen_h * mean_offset[1] ** 3
        + _DIFF_COEF * diff_offset[1]
        + _HEAD_POSE_COEF_PER_DEG * head_pose_deg[1]  # pitch
        + _TRANSLATION_COEF_PER_MM * translation_jitter[1]
        + rng.normal(scale=_TARGET_NOISE_STD_PX)
    )
    return np.array([x, y])


def _sample_point(fx: float, fy: float, screen_w: int, screen_h: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic frame's (feature, target) for calibration target
    fraction (fx, fy) -- target is the *true* fixed grid point (what the
    user was asked to look at); feature is a noisy sample consistent with
    that target under the ground-truth model above."""
    base_mean_offset = np.array([(fx - 0.5) * _EYE_OFFSET_RANGE, (fy - 0.5) * _EYE_OFFSET_RANGE])
    mean_offset = base_mean_offset + rng.normal(scale=_RNG_MEASUREMENT_NOISE_STD, size=2)
    diff_offset = rng.normal(scale=_DIFF_OFFSET_STD, size=2)
    head_pose_deg = np.array(
        [rng.normal(scale=_HEAD_YAW_PITCH_STD_DEG), rng.normal(scale=_HEAD_YAW_PITCH_STD_DEG), rng.normal(scale=_HEAD_ROLL_STD_DEG)]
    )
    translation_mm = np.array(_BASELINE_TRANSLATION_MM) + np.array(
        [
            rng.normal(scale=_HEAD_TRANSLATION_XY_STD_MM),
            rng.normal(scale=_HEAD_TRANSLATION_XY_STD_MM),
            rng.normal(scale=_HEAD_TRANSLATION_Z_STD_MM),
        ]
    )
    feature = build_feature_vector(np.concatenate([mean_offset, diff_offset]), tuple(head_pose_deg), tuple(translation_mm))
    target = _ground_truth_screen_position(mean_offset, diff_offset, head_pose_deg, translation_mm, screen_w, screen_h, rng)
    return feature, target


def generate_synthetic_session(seed: int = 0, screen_w: int = 1920, screen_h: int = 1080) -> SyntheticSession:
    rng = np.random.default_rng(seed)

    train_features, train_targets, train_groups = [], [], []
    for group_id, (fx, fy) in enumerate(TRAIN_GRID_FRACTIONS):
        for _ in range(SAMPLES_PER_POINT):
            feature, target = _sample_point(fx, fy, screen_w, screen_h, rng)
            train_features.append(feature)
            train_targets.append(target)
            train_groups.append(group_id)

    val_features, val_targets = [], []
    for fx, fy in VALIDATION_FRACTIONS:
        for _ in range(SAMPLES_PER_POINT):
            feature, target = _sample_point(fx, fy, screen_w, screen_h, rng)
            val_features.append(feature)
            val_targets.append(target)

    return SyntheticSession(
        screen_w=screen_w,
        screen_h=screen_h,
        train_features=np.array(train_features),
        train_targets=np.array(train_targets),
        train_groups=np.array(train_groups),
        val_features=np.array(val_features),
        val_targets=np.array(val_targets),
    )


def _generate_validation_samples(
    rng: np.random.Generator, screen_w: int, screen_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Factored out of generate_synthetic_session() so the pursuit-protocol
    generator below can produce an IDENTICALLY-DISTRIBUTED validation set
    (same fixed probe points, same noise model) without duplicating the
    original function or depending on its internal RNG draw order."""
    val_features, val_targets = [], []
    for fx, fy in VALIDATION_FRACTIONS:
        for _ in range(SAMPLES_PER_POINT):
            feature, target = _sample_point(fx, fy, screen_w, screen_h, rng)
            val_features.append(feature)
            val_targets.append(target)
    return np.array(val_features), np.array(val_targets)


# ---------------------------------------------------------------------------
# Continuous cursor-pursuit calibration protocol (synthetic)
#
# Models a DIFFERENT calibration *data-capture* protocol: instead of the
# still-point grid above (a sequence of discrete, fixed dots the user
# fixates on with a settle/dwell period before each is captured), the
# cursor moves continuously along a path and the user's eyes track it,
# with frames sampled at a steady rate throughout.
#
# The two protocols share the exact same VALIDATION_FRACTIONS validation
# set (see generate_pursuit_session / generate_protocol_comparison_session
# below) -- this narrowly answers "does changing how the TRAINING data was
# captured change accuracy on a fixed set of probe points," not some
# differently-measured notion of pursuit accuracy.
#
# The key modeled difference beyond "many more, continuously-varying target
# positions instead of 16 fixed ones": smooth pursuit eye movements do not
# track a moving target perfectly. There is a well-documented response
# latency (roughly 100-150ms) before the eye begins moving, and the eye's
# tracking velocity typically falls a little short of the target's --
# pursuit "gain" below 1, more so for faster or more erratic motion -- so at
# any instant, gaze is lagging behind and undershooting where the cursor
# actually is. A still-point protocol with a brief settle/dwell time avoids
# almost all of this (the eye has caught up and stopped by the time a
# sample is taken); a continuously-moving-target protocol that simply logs
# "cursor position = calibration label" for every captured frame bakes
# this lag/undershoot in as label noise/bias the model can never fit away.
# That is modeled explicitly below via PURSUIT_LATENCY_S / PURSUIT_GAIN,
# rather than silently assumed away -- adjust them (or pass --latency /
# --gain to scripts/compare_calibration_protocols.py) if you think your
# actual tracking is better or worse than this assumption, since the
# conclusion is sensitive to them.
PURSUIT_LATENCY_S = 0.12
PURSUIT_GAIN = 0.90
PURSUIT_DURATION_S = 16.0
PURSUIT_SAMPLE_HZ = 15.0
# Total pursuit training samples (16.0 * 15.0 = 240) deliberately matches
# the still-point grid's 16 points * SAMPLES_PER_POINT(15) = 240, so a
# comparison isn't confounded by one protocol simply having more data.
PURSUIT_GROUP_WINDOW_S = 1.0  # contiguous time window treated as one group for group-aware CV

# A Lissajous sweep -- the kind of continuous path real pursuit-based
# calibration tools use -- so the trajectory covers the screen without
# retracing the same line. A 3:2 frequency ratio, with each axis set to
# complete a whole number of cycles over PURSUIT_DURATION_S, forms a
# closed figure that sweeps the full rectangle evenly within one session;
# an earlier, non-matching frequency pair left y short of a full cycle
# (ending mid-sweep, under-covering the screen center) and made pursuit
# look needlessly worse for a reason that had nothing to do with pursuit
# lag/gain -- the thing this benchmark is actually trying to isolate.
_LISSAJOUS_FREQ_X_HZ = 3.0 / PURSUIT_DURATION_S
_LISSAJOUS_FREQ_Y_HZ = 2.0 / PURSUIT_DURATION_S
_LISSAJOUS_AMPLITUDE = 0.42  # fraction of half-screen the sweep covers


def _pursuit_cursor_fraction(t: float) -> tuple[float, float]:
    """True on-screen cursor position (as a screen fraction) at time t."""
    fx = 0.5 + _LISSAJOUS_AMPLITUDE * np.sin(2 * np.pi * _LISSAJOUS_FREQ_X_HZ * t)
    fy = 0.5 + _LISSAJOUS_AMPLITUDE * np.sin(2 * np.pi * _LISSAJOUS_FREQ_Y_HZ * t + np.pi / 2)
    return fx, fy


def _sample_pursuit_point(
    t: float, screen_w: int, screen_h: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic frame's (feature, target) at time t of a continuous
    pursuit sweep.

    feature is built from where the eye actually, physically is: a
    latency-delayed, gain-reduced ("undershot") tracking of the moving
    cursor, using the same per-frame measurement/head-pose/translation
    noise model as the still-point protocol.

    target is what the real system would record as the training label:
    the cursor's own (un-lagged) displayed position at time t, run through
    the same ground-truth screen-position function used for still points.
    The gap between where the feature says the eye is and what the label
    claims is the pursuit-specific mislabeling error this protocol has that
    still-point calibration (with adequate dwell time) does not.
    """
    cursor_fx, cursor_fy = _pursuit_cursor_fraction(t)
    delayed_fx, delayed_fy = _pursuit_cursor_fraction(max(t - PURSUIT_LATENCY_S, 0.0))
    tracked_fx = 0.5 + PURSUIT_GAIN * (delayed_fx - 0.5)
    tracked_fy = 0.5 + PURSUIT_GAIN * (delayed_fy - 0.5)

    base_mean_offset = np.array(
        [(tracked_fx - 0.5) * _EYE_OFFSET_RANGE, (tracked_fy - 0.5) * _EYE_OFFSET_RANGE]
    )
    mean_offset = base_mean_offset + rng.normal(scale=_RNG_MEASUREMENT_NOISE_STD, size=2)
    diff_offset = rng.normal(scale=_DIFF_OFFSET_STD, size=2)
    head_pose_deg = np.array(
        [
            rng.normal(scale=_HEAD_YAW_PITCH_STD_DEG),
            rng.normal(scale=_HEAD_YAW_PITCH_STD_DEG),
            rng.normal(scale=_HEAD_ROLL_STD_DEG),
        ]
    )
    translation_mm = np.array(_BASELINE_TRANSLATION_MM) + np.array(
        [
            rng.normal(scale=_HEAD_TRANSLATION_XY_STD_MM),
            rng.normal(scale=_HEAD_TRANSLATION_XY_STD_MM),
            rng.normal(scale=_HEAD_TRANSLATION_Z_STD_MM),
        ]
    )
    feature = build_feature_vector(
        np.concatenate([mean_offset, diff_offset]), tuple(head_pose_deg), tuple(translation_mm)
    )

    cursor_mean_offset = np.array(
        [(cursor_fx - 0.5) * _EYE_OFFSET_RANGE, (cursor_fy - 0.5) * _EYE_OFFSET_RANGE]
    )
    target = _ground_truth_screen_position(
        cursor_mean_offset, diff_offset, head_pose_deg, translation_mm, screen_w, screen_h, rng
    )
    return feature, target


def generate_pursuit_session(seed: int = 0, screen_w: int = 1920, screen_h: int = 1080) -> SyntheticSession:
    """Drop-in alternative to generate_synthetic_session() using the
    continuous cursor-pursuit protocol instead of the discrete still-point
    grid for TRAINING data (validation uses the identical still-point
    VALIDATION_FRACTIONS set) -- so scripts/bench_calibration.py's existing
    "did my change to calibration.py help or hurt" workflow
    (--synthetic --protocol pursuit) works unchanged against pursuit-shaped
    data too."""
    rng = np.random.default_rng(seed)

    n_samples = int(round(PURSUIT_DURATION_S * PURSUIT_SAMPLE_HZ))
    train_features, train_targets, train_groups = [], [], []
    for i in range(n_samples):
        t = i / PURSUIT_SAMPLE_HZ
        feature, target = _sample_pursuit_point(t, screen_w, screen_h, rng)
        train_features.append(feature)
        train_targets.append(target)
        train_groups.append(int(t // PURSUIT_GROUP_WINDOW_S))

    val_features, val_targets = _generate_validation_samples(rng, screen_w, screen_h)

    return SyntheticSession(
        screen_w=screen_w,
        screen_h=screen_h,
        train_features=np.array(train_features),
        train_targets=np.array(train_targets),
        train_groups=np.array(train_groups),
        val_features=val_features,
        val_targets=val_targets,
    )


@dataclass
class ProtocolComparisonSession:
    """Still-point vs. cursor-pursuit TRAINING data, paired with ONE shared
    validation set so both can be fairly compared -- see
    generate_protocol_comparison_session() and
    scripts/compare_calibration_protocols.py."""

    screen_w: int
    screen_h: int
    still_train_features: np.ndarray
    still_train_targets: np.ndarray
    still_train_groups: np.ndarray
    pursuit_train_features: np.ndarray
    pursuit_train_targets: np.ndarray
    pursuit_train_groups: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray


def generate_protocol_comparison_session(
    seed: int = 0, screen_w: int = 1920, screen_h: int = 1080
) -> ProtocolComparisonSession:
    """Builds still-point and cursor-pursuit training sets from independent
    RNG streams (via SeedSequence.spawn, not just two different loops
    sharing one generator) so adding/removing samples from one protocol can
    never shift the other's random draws, plus a THIRD independent stream
    for a validation set shared, bit-for-bit, by both -- the fair, paired
    comparison scripts/compare_calibration_protocols.py reports on."""
    seed_seq = np.random.SeedSequence(seed)
    still_rng, pursuit_rng, val_rng = (np.random.default_rng(s) for s in seed_seq.spawn(3))

    still_features, still_targets, still_groups = [], [], []
    for group_id, (fx, fy) in enumerate(TRAIN_GRID_FRACTIONS):
        for _ in range(SAMPLES_PER_POINT):
            feature, target = _sample_point(fx, fy, screen_w, screen_h, still_rng)
            still_features.append(feature)
            still_targets.append(target)
            still_groups.append(group_id)

    n_samples = int(round(PURSUIT_DURATION_S * PURSUIT_SAMPLE_HZ))
    pursuit_features, pursuit_targets, pursuit_groups = [], [], []
    for i in range(n_samples):
        t = i / PURSUIT_SAMPLE_HZ
        feature, target = _sample_pursuit_point(t, screen_w, screen_h, pursuit_rng)
        pursuit_features.append(feature)
        pursuit_targets.append(target)
        pursuit_groups.append(int(t // PURSUIT_GROUP_WINDOW_S))

    val_features, val_targets = _generate_validation_samples(val_rng, screen_w, screen_h)

    return ProtocolComparisonSession(
        screen_w=screen_w,
        screen_h=screen_h,
        still_train_features=np.array(still_features),
        still_train_targets=np.array(still_targets),
        still_train_groups=np.array(still_groups),
        pursuit_train_features=np.array(pursuit_features),
        pursuit_train_targets=np.array(pursuit_targets),
        pursuit_train_groups=np.array(pursuit_groups),
        val_features=val_features,
        val_targets=val_targets,
    )
