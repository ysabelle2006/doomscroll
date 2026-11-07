# Tests for HeadPoseEstimator's PnP-based pose recovery and pitch unwrapping.
from __future__ import annotations

import cv2
import numpy as np
import pytest

from doomscroll.gaze.head_pose import (
    HeadPoseEstimator,
    _LANDMARK_INDICES,
    _MODEL_POINTS_3D,
    _unwrap_pitch,
    _camera_matrix,
)

# Image size and landmark count used across these tests.
IMAGE_W, IMAGE_H = 640, 480
N_LANDMARKS = 478


# Projects a synthetic 3D head pose into 2D landmark pixel positions.
def _project_pose_to_landmarks(yaw_deg: float, pitch_deg: float, roll_deg: float, tvec: np.ndarray) -> np.ndarray:
    y, p, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(y), -np.sin(y)], [0, np.sin(y), np.cos(y)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    rotation_matrix = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    camera_matrix = _camera_matrix(IMAGE_W, IMAGE_H)
    dist_coeffs = np.zeros((4, 1))
    image_points, _ = cv2.projectPoints(_MODEL_POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = image_points.reshape(-1, 2)

    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
    for idx, px in zip(_LANDMARK_INDICES, image_points):
        pts[idx] = px
    return pts


# Checks normal pitch values pass through unchanged.
def test_unwrap_pitch_leaves_normal_values_unchanged():
    assert _unwrap_pitch(0.0) == 0.0
    assert _unwrap_pitch(45.0) == 45.0
    assert _unwrap_pitch(-45.0) == -45.0


# Checks pitch wraps correctly near +180 degrees.
def test_unwrap_pitch_wraps_near_positive_180():
    assert _unwrap_pitch(170.0) == pytest.approx(10.0)
    assert _unwrap_pitch(100.0) == pytest.approx(80.0)


# Checks pitch wraps correctly near -180 degrees.
def test_unwrap_pitch_wraps_near_negative_180():
    assert _unwrap_pitch(-170.0) == pytest.approx(-10.0)
    assert _unwrap_pitch(-100.0) == pytest.approx(-80.0)


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [
        (0.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, -10.0, 0.0),
        (10.0, -8.0, 5.0),
    ],
)
# Checks the estimator recovers a known synthetic yaw/pitch/roll/position.
def test_estimate_recovers_known_pose(yaw, pitch, roll):
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(yaw, pitch, roll, tvec)

    estimator = HeadPoseEstimator()
    result = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)

    assert result is not None
    assert result.yaw_deg == pytest.approx(yaw, abs=1.0)
    assert result.pitch_deg == pytest.approx(pitch, abs=1.0)
    assert result.roll_deg == pytest.approx(roll, abs=1.0)
    np.testing.assert_allclose(result.translation_vector.flatten(), tvec.flatten(), atol=5.0)


# Checks a second estimate seeded by the previous one is consistent.
def test_estimate_uses_previous_pose_as_seed():
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(5.0, 5.0, 0.0, tvec)
    estimator = HeadPoseEstimator()

    first = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    second = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)

    assert first is not None and second is not None
    assert second.yaw_deg == pytest.approx(first.yaw_deg, abs=0.5)
    assert second.pitch_deg == pytest.approx(first.pitch_deg, abs=0.5)


# Checks an extreme roll angle is rejected as an invalid estimate.
def test_estimate_rejects_extreme_roll():
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(0.0, 0.0, 120.0, tvec)
    estimator = HeadPoseEstimator()
    result = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    assert result is None


# Checks reset() clears the estimator's seeded previous pose.
def test_estimate_reset_clears_seed():
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(5.0, 5.0, 0.0, tvec)
    estimator = HeadPoseEstimator()
    estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    estimator.reset()
    assert estimator._prev_rvec is None
    assert estimator._prev_tvec is None
