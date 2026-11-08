# Shared fixtures and synthetic landmark builders for gaze-related tests.
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

# Number of landmarks and pixel positions used to build synthetic eye geometry.
N_LANDMARKS = 478

LEFT_EYE_OUTER_PX = np.array([230.0, 200.0])
LEFT_EYE_INNER_PX = np.array([270.0, 200.0])
RIGHT_EYE_OUTER_PX = np.array([410.0, 200.0])
RIGHT_EYE_INNER_PX = np.array([370.0, 200.0])
LEFT_EYE_WIDTH = float(np.linalg.norm(LEFT_EYE_OUTER_PX - LEFT_EYE_INNER_PX))
RIGHT_EYE_WIDTH = float(np.linalg.norm(RIGHT_EYE_OUTER_PX - RIGHT_EYE_INNER_PX))


# Builds synthetic facial landmarks for two eyes from mean/diff offsets and openness.
def make_eye_landmarks(
    mean_offset: tuple[float, float] = (0.0, 0.0),
    diff_offset: tuple[float, float] = (0.0, 0.0),
    eye_openness: float = 1.0,
) -> np.ndarray:
    mean_offset = np.asarray(mean_offset, dtype=np.float64)
    diff_offset = np.asarray(diff_offset, dtype=np.float64)
    left_offset = mean_offset + diff_offset / 2.0
    right_offset = mean_offset - diff_offset / 2.0

    left_eye_center = (LEFT_EYE_OUTER_PX + LEFT_EYE_INNER_PX) / 2.0
    right_eye_center = (RIGHT_EYE_OUTER_PX + RIGHT_EYE_INNER_PX) / 2.0
    left_iris_px = left_eye_center + left_offset * LEFT_EYE_WIDTH
    right_iris_px = right_eye_center + right_offset * RIGHT_EYE_WIDTH

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


# Fixture exposing make_eye_landmarks to tests.
@pytest.fixture
def eye_landmarks_factory():
    return make_eye_landmarks
