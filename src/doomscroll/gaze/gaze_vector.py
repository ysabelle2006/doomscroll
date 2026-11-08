# Computes a handcrafted gaze feature vector (iris offset within each eye)
# from face landmarks, and a blink/closed-eye usability check.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.gaze.pupil_refine import refine_iris_center
from doomscroll.landmarks.mediapipe_backend import LEFT_IRIS, RIGHT_IRIS

# MediaPipe landmark indices for eye corners, iris centers, and eyelids.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
LEFT_IRIS_CENTER = LEFT_IRIS[0]
RIGHT_IRIS_CENTER = RIGHT_IRIS[0]

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

DEFAULT_MIN_EYE_ASPECT_RATIO = 0.08


# Holds the computed gaze feature vector for one frame.
@dataclass
class GazeFeature:
    vector: np.ndarray


# Returns the iris position relative to the eye corners, normalized by eye width.
def _eye_in_head_offset_from_px(outer: np.ndarray, inner: np.ndarray, iris: np.ndarray) -> np.ndarray:
    eye_center = (outer + inner) / 2.0
    eye_width = np.linalg.norm(outer - inner)
    if eye_width < 1e-6:
        return np.zeros(2)
    return (iris - eye_center) / eye_width


# Computes the gaze feature vector (mean and left/right-difference iris offsets) for one frame.
def compute_gaze_feature(points_px: np.ndarray, frame_gray: np.ndarray | None = None) -> GazeFeature:
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


# Computes one eye's aspect ratio (vertical / horizontal eyelid distance).
def _eye_aspect_ratio(points_px: np.ndarray, top_idx: int, bottom_idx: int, outer_idx: int, inner_idx: int) -> float:
    vertical = np.linalg.norm(points_px[top_idx] - points_px[bottom_idx])
    horizontal = np.linalg.norm(points_px[outer_idx] - points_px[inner_idx])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


# Returns True if both eyes are open enough (not blinking) to trust this frame's gaze reading.
def is_gaze_usable(points_px: np.ndarray, min_ear: float = DEFAULT_MIN_EYE_ASPECT_RATIO) -> bool:
    left_ear = _eye_aspect_ratio(points_px, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right_ear = _eye_aspect_ratio(points_px, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    return left_ear >= min_ear and right_ear >= min_ear
