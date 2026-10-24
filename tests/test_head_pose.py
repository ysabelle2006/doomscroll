"""Tests for doomscroll.gaze.head_pose: the solvePnP-based yaw/pitch/roll +
translation estimate that feeds into calibration alongside the eye-in-head
gaze feature.

Rather than synthetic-but-arbitrary landmarks, this builds landmarks by
*projecting* head_pose.py's own 3D face model through a chosen, known
camera pose (cv2.projectPoints) -- an exact analytic round-trip: feed in a
known yaw/pitch/roll/translation, and HeadPoseEstimator.estimate() should
recover (close to) that same pose back out. This exercises the real
solvePnP + Euler-angle decomposition + pitch-unwrap pipeline, not a
hand-wavy approximation of it.
"""
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

IMAGE_W, IMAGE_H = 640, 480
N_LANDMARKS = 478


def _project_pose_to_landmarks(yaw_deg: float, pitch_deg: float, roll_deg: float, tvec: np.ndarray) -> np.ndarray:
    """Builds a (478, 2) landmark array whose 6 head-pose landmarks are the
    exact projection of _MODEL_POINTS_3D under the given pose -- the
    ground truth HeadPoseEstimator.estimate() should recover."""
    # OpenCV's rotation vector is axis-angle, not Euler angles directly;
    # build the rotation matrix from yaw/pitch/roll via the same
    # convention cv2.decomposeProjectionMatrix uses (XYZ order) by
    # composing elementary rotations and letting cv2.Rodrigues convert
    # back to a rotation vector for projectPoints.
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


def test_unwrap_pitch_leaves_normal_values_unchanged():
    assert _unwrap_pitch(0.0) == 0.0
    assert _unwrap_pitch(45.0) == 45.0
    assert _unwrap_pitch(-45.0) == -45.0


def test_unwrap_pitch_wraps_near_positive_180():
    # decomposeProjectionMatrix can report a frontal-ish face as pitch near
    # +/-180 depending on sign convention; _unwrap_pitch folds that back to
    # the equivalent near-zero angle a frontal face should read as.
    assert _unwrap_pitch(170.0) == pytest.approx(10.0)
    assert _unwrap_pitch(100.0) == pytest.approx(80.0)


def test_unwrap_pitch_wraps_near_negative_180():
    assert _unwrap_pitch(-170.0) == pytest.approx(-10.0)
    assert _unwrap_pitch(-100.0) == pytest.approx(-80.0)


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [
        (0.0, 0.0, 0.0),      # straight-on
        (15.0, 0.0, 0.0),     # turned right
        (-15.0, 0.0, 0.0),    # turned left
        (0.0, 10.0, 0.0),     # chin up/down a little (small -- see calibrate.py's own caution about losing face)
        (0.0, -10.0, 0.0),
        (10.0, -8.0, 5.0),    # a realistic combined incidental pose
    ],
)
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


def test_estimate_uses_previous_pose_as_seed():
    """Feeding two frames of the same pose shouldn't flip to a different
    (degenerate) solvePnP solution the second time -- the estimator should
    settle on a consistent answer once seeded."""
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(5.0, 5.0, 0.0, tvec)
    estimator = HeadPoseEstimator()

    first = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    second = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)

    assert first is not None and second is not None
    assert second.yaw_deg == pytest.approx(first.yaw_deg, abs=0.5)
    assert second.pitch_deg == pytest.approx(first.pitch_deg, abs=0.5)


def test_estimate_rejects_extreme_roll():
    """Roll > 90 degrees is treated as a bad/degenerate solve (see the
    sanity check in estimate()) and should return None, not a nonsense
    pose."""
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(0.0, 0.0, 120.0, tvec)
    estimator = HeadPoseEstimator()
    result = estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    assert result is None


def test_estimate_reset_clears_seed():
    tvec = np.array([[0.0], [0.0], [400.0]])
    landmarks_px = _project_pose_to_landmarks(5.0, 5.0, 0.0, tvec)
    estimator = HeadPoseEstimator()
    estimator.estimate(landmarks_px, IMAGE_W, IMAGE_H)
    estimator.reset()
    assert estimator._prev_rvec is None
    assert estimator._prev_tvec is None
