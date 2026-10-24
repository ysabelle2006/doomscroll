"""Shared pytest fixtures for the eye-tracking/calibration test suite.

The project's face-landmark pipeline normally comes from MediaPipe running
on real webcam frames, which isn't something a test suite can depend on.
Everything here instead builds synthetic landmark arrays with known,
controllable geometry -- just the handful of points each module under test
actually reads -- so the math in gaze_vector.py / head_pose.py can be
exercised deterministically, with a known expected answer, with no camera
and no MediaPipe model file involved.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.gaze_vector import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
    RIGHT_IRIS_CENTER,
)

N_LANDMARKS = 478  # MediaPipe's full face mesh + iris point count

# Fixed, plausible eye geometry for a roughly centered frontal face in a
# 640x480 frame. Arbitrary but fixed -- only the *relationships* the code
# actually depends on (eye_center, eye_width, EAR) matter, not real-world
# anatomical accuracy.
LEFT_EYE_OUTER_PX = np.array([230.0, 200.0])
LEFT_EYE_INNER_PX = np.array([270.0, 200.0])
RIGHT_EYE_OUTER_PX = np.array([410.0, 200.0])
RIGHT_EYE_INNER_PX = np.array([370.0, 200.0])
LEFT_EYE_WIDTH = float(np.linalg.norm(LEFT_EYE_OUTER_PX - LEFT_EYE_INNER_PX))  # 40.0
RIGHT_EYE_WIDTH = float(np.linalg.norm(RIGHT_EYE_OUTER_PX - RIGHT_EYE_INNER_PX))  # 40.0


def make_eye_landmarks(
    mean_offset: tuple[float, float] = (0.0, 0.0),
    diff_offset: tuple[float, float] = (0.0, 0.0),
    eye_openness: float = 1.0,
) -> np.ndarray:
    """Builds a (478, 2) pixel-coordinate landmark array with eye corners,
    eyelids and iris centers placed so that compute_gaze_feature() on it is
    *known* to recover exactly `mean_offset`/`diff_offset` (within float
    tolerance), and is_gaze_usable() reflects `eye_openness` (1.0 = normal
    open eye, 0.0 = fully closed/blinking). All other landmark indices are
    left at the origin -- nothing under test reads them.

    This inverts gaze_vector.compute_gaze_feature()'s own math: each eye's
    offset is (iris - eye_center) / eye_width, and
    mean_offset=(left+right)/2, diff_offset=left-right. Solving for the
    per-eye offsets: left = mean + diff/2, right = mean - diff/2.
    """
    mean_offset = np.asarray(mean_offset, dtype=np.float64)
    diff_offset = np.asarray(diff_offset, dtype=np.float64)
    left_offset = mean_offset + diff_offset / 2.0
    right_offset = mean_offset - diff_offset / 2.0

    left_eye_center = (LEFT_EYE_OUTER_PX + LEFT_EYE_INNER_PX) / 2.0
    right_eye_center = (RIGHT_EYE_OUTER_PX + RIGHT_EYE_INNER_PX) / 2.0
    left_iris_px = left_eye_center + left_offset * LEFT_EYE_WIDTH
    right_iris_px = right_eye_center + right_offset * RIGHT_EYE_WIDTH

    # Vertical eyelid gap of 12px at eye_width=40 gives EAR=0.3 when fully
    # open (comfortably above DEFAULT_MIN_EYE_ASPECT_RATIO=0.12); scaling it
    # by eye_openness sweeps smoothly down to EAR=0 (clearly closed) at 0.0.
    half_gap_left = 6.0 * eye_openness
    half_gap_right = 6.0 * eye_openness

    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
    pts[LEFT_EYE_OUTER] = LEFT_EYE_OUTER_PX
    pts[LEFT_EYE_INNER] = LEFT_EYE_INNER_PX
    pts[RIGHT_EYE_OUTER] = RIGHT_EYE_OUTER_PX
    pts[RIGHT_EYE_INNER] = RIGHT_EYE_INNER_PX
    pts[LEFT_IRIS_CENTER] = left_iris_px
    pts[RIGHT_IRIS_CENTER] = right_iris_px
    pts[LEFT_EYE_TOP] = left_eye_center - np.array([0.0, half_gap_left])
    pts[LEFT_EYE_BOTTOM] = left_eye_center + np.array([0.0, half_gap_left])
    pts[RIGHT_EYE_TOP] = right_eye_center - np.array([0.0, half_gap_right])
    pts[RIGHT_EYE_BOTTOM] = right_eye_center + np.array([0.0, half_gap_right])
    return pts


@pytest.fixture
def eye_landmarks_factory():
    """Returns make_eye_landmarks so tests can build several variants."""
    return make_eye_landmarks
