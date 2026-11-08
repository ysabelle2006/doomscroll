# Generates synthetic calibration sessions (still-point, pursuit, and both) for testing.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.gaze.calibration import build_feature_vector

# Grid points used for training and held-out points used for validation.
_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
TRAIN_GRID_FRACTIONS = [(fx, fy) for fy in _GRID_STEPS for fx in _GRID_STEPS]
VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]

# Number of noisy samples generated per calibration point.
SAMPLES_PER_POINT = 15

# Coefficients and noise scales controlling the synthetic gaze-to-screen mapping.
_EYE_OFFSET_RANGE = 0.30
_HEAD_POSE_COEF_PER_DEG = 3.0
_TRANSLATION_COEF_PER_MM = 0.6
_DIFF_COEF = 400.0

_RNG_MEASUREMENT_NOISE_STD = 0.01
_HEAD_YAW_PITCH_STD_DEG = 3.0
_HEAD_ROLL_STD_DEG = 2.0
_HEAD_TRANSLATION_XY_STD_MM = 5.0
_HEAD_TRANSLATION_Z_STD_MM = 10.0
_DIFF_OFFSET_STD = 0.02
_BASELINE_TRANSLATION_MM = (0.0, 0.0, 400.0)

# Nonlinearity strength and target position noise for the synthetic mapping.
_NONLINEARITY_COEF = 8.0
_TARGET_NOISE_STD_PX = 4.0


# Synthetic calibration/validation data for a single still-point session.
@dataclass
class SyntheticSession:
    screen_w: int
    screen_h: int
    train_features: np.ndarray
    train_targets: np.ndarray
    train_groups: np.ndarray
    val_features: np.ndarray
    val_targets: np.ndarray


# Computes the synthetic ground-truth screen position for given gaze/pose/translation.
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
        + _HEAD_POSE_COEF_PER_DEG * head_pose_deg[0]
        + _TRANSLATION_COEF_PER_MM * translation_jitter[0]
        + rng.normal(scale=_TARGET_NOISE_STD_PX)
    )
    y = (
        cy + sy * mean_offset[1]
        + _NONLINEARITY_COEF * screen_h * mean_offset[1] ** 3
        + _DIFF_COEF * diff_offset[1]
        + _HEAD_POSE_COEF_PER_DEG * head_pose_deg[1]
        + _TRANSLATION_COEF_PER_MM * translation_jitter[1]
        + rng.normal(scale=_TARGET_NOISE_STD_PX)
    )
    return np.array([x, y])


# Samples one noisy feature/target pair for a calibration point at (fx, fy).
def _sample_point(fx: float, fy: float, screen_w: int, screen_h: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
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


# Generates a full synthetic still-point session: grid training samples plus validation samples.
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


# Generates validation samples at the fixed validation fractions.
def _generate_validation_samples(
    rng: np.random.Generator, screen_w: int, screen_h: int
) -> tuple[np.ndarray, np.ndarray]:
    val_features, val_targets = [], []
    for fx, fy in VALIDATION_FRACTIONS:
        for _ in range(SAMPLES_PER_POINT):
            feature, target = _sample_point(fx, fy, screen_w, screen_h, rng)
            val_features.append(feature)
            val_targets.append(target)
    return np.array(val_features), np.array(val_targets)


# Timing/gain constants for the simulated smooth-pursuit tracking protocol.
PURSUIT_LATENCY_S = 0.12
PURSUIT_GAIN = 0.90
PURSUIT_DURATION_S = 16.0
PURSUIT_SAMPLE_HZ = 15.0
PURSUIT_GROUP_WINDOW_S = 1.0

# Lissajous curve parameters for the simulated pursuit cursor path.
_LISSAJOUS_FREQ_X_HZ = 3.0 / PURSUIT_DURATION_S
_LISSAJOUS_FREQ_Y_HZ = 2.0 / PURSUIT_DURATION_S
_LISSAJOUS_AMPLITUDE = 0.42


# Returns the pursuit target's screen fraction at time t along a Lissajous curve.
def _pursuit_cursor_fraction(t: float) -> tuple[float, float]:
    fx = 0.5 + _LISSAJOUS_AMPLITUDE * np.sin(2 * np.pi * _LISSAJOUS_FREQ_X_HZ * t)
    fy = 0.5 + _LISSAJOUS_AMPLITUDE * np.sin(2 * np.pi * _LISSAJOUS_FREQ_Y_HZ * t + np.pi / 2)
    return fx, fy


# Samples one noisy feature/target pair for a pursuit session at time t, with tracking lag/gain.
def _sample_pursuit_point(
    t: float, screen_w: int, screen_h: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
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


# Generates a full synthetic pursuit-protocol session with time-windowed groups.
def generate_pursuit_session(seed: int = 0, screen_w: int = 1920, screen_h: int = 1080) -> SyntheticSession:
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


# Synthetic data comparing still-point and pursuit training protocols on shared validation data.
@dataclass
class ProtocolComparisonSession:

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


# Generates a session with both still-point and pursuit training data sharing one validation set.
def generate_protocol_comparison_session(
    seed: int = 0, screen_w: int = 1920, screen_h: int = 1080
) -> ProtocolComparisonSession:
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
