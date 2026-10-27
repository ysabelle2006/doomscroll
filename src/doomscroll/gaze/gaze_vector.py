"""Eye-in-head gaze feature: iris position relative to each eye's own
corners, independent of head pose by construction. Combined with head pose
as separate regressor inputs in calibration.py, per the plan's approach to
head-motion compensation (the regression learns the correction rather than
subtracting it out geometrically)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.gaze.pupil_refine import refine_iris_center
from doomscroll.landmarks.mediapipe_backend import LEFT_IRIS, RIGHT_IRIS

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_IRIS_CENTER = LEFT_IRIS[0]
RIGHT_IRIS_CENTER = RIGHT_IRIS[0]

# Approximate top/bottom eyelid midpoints, for a simple eye-aspect-ratio
# blink check -- a closed/blinking eye gives an unreliable iris position
# that should never be used as a calibration sample or a live gaze update.
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

DEFAULT_MIN_EYE_ASPECT_RATIO = 0.08


@dataclass
class GazeFeature:
    vector: np.ndarray  # shape (4,): [mean_x, mean_y, diff_x, diff_y]


def _eye_in_head_offset_from_px(outer: np.ndarray, inner: np.ndarray, iris: np.ndarray) -> np.ndarray:
    eye_center = (outer + inner) / 2.0
    eye_width = np.linalg.norm(outer - inner)
    if eye_width < 1e-6:
        return np.zeros(2)
    return (iris - eye_center) / eye_width


def compute_gaze_feature(points_px: np.ndarray, frame_gray: np.ndarray | None = None) -> GazeFeature:
    """Both eyes move together (conjugate eye movement), so their raw
    per-eye offsets are highly correlated -- feeding both directly into a
    Ridge fit is a collinearity problem: the model can split credit between
    two near-redundant predictors unstably, sometimes landing on coefficients
    that nearly cancel, leaving a small residual that can flip sign for
    feature combinations that don't exactly match calibration (natural
    per-frame asymmetry between the eyes). mean_offset is the robust,
    well-conditioned conjugate signal; diff_offset isolates the much smaller
    residual (vergence/asymmetry) so it can't fight the dominant signal for
    credit -- same dimensionality (4), much better conditioned to regress on.

    frame_gray: optional grayscale frame. When given, each eye's iris
    position is first passed through pupil_refine.refine_iris_center() -- a
    classical image-processing check/refinement layered on top of
    MediaPipe's landmark, which falls back to the landmark unchanged
    whenever the refinement doesn't pass its own confidence check. Pass
    None (the default) to use the landmark position directly, e.g. for
    callers without easy access to the raw frame.
    """
    left_outer, left_inner, left_iris = (
        points_px[LEFT_EYE_OUTER],
        points_px[LEFT_EYE_INNER],
        points_px[LEFT_IRIS_CENTER],
    )
    right_outer, right_inner, right_iris = (
        points_px[RIGHT_EYE_OUTER],
        points_px[RIGHT_EYE_INNER],
        points_px[RIGHT_IRIS_CENTER],
    )

    if frame_gray is not None:
        left_iris = refine_iris_center(frame_gray, left_iris, left_outer, left_inner)
        right_iris = refine_iris_center(frame_gray, right_iris, right_outer, right_inner)

    left = _eye_in_head_offset_from_px(left_outer, left_inner, left_iris)
    right = _eye_in_head_offset_from_px(right_outer, right_inner, right_iris)
    mean_offset = (left + right) / 2.0
    diff_offset = left - right
    return GazeFeature(vector=np.concatenate([mean_offset, diff_offset]))


def _eye_aspect_ratio(points_px: np.ndarray, top_idx: int, bottom_idx: int, outer_idx: int, inner_idx: int) -> float:
    vertical = np.linalg.norm(points_px[top_idx] - points_px[bottom_idx])
    horizontal = np.linalg.norm(points_px[outer_idx] - points_px[inner_idx])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def is_gaze_usable(points_px: np.ndarray, min_ear: float = DEFAULT_MIN_EYE_ASPECT_RATIO) -> bool:
    """False while either eye is closed/blinking, so callers can skip
    calibration samples and live updates during a blink rather than feeding
    in an unreliable iris position."""
    left_ear = _eye_aspect_ratio(points_px, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_ear = _eye_aspect_ratio(points_px, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    return left_ear >= min_ear and right_ear >= min_ear
