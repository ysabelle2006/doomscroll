# Tests for compute_gaze_feature and is_gaze_usable.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from tests.conftest import LEFT_EYE_OUTER_PX, LEFT_EYE_INNER_PX, make_eye_landmarks


# Checks centered eyes give a zero-offset feature vector.
def test_compute_gaze_feature_centered_gives_zero_offset():
    pts = make_eye_landmarks(mean_offset=(0.0, 0.0), diff_offset=(0.0, 0.0))
    feature = compute_gaze_feature(pts)
    np.testing.assert_allclose(feature.vector, [0.0, 0.0, 0.0, 0.0], atol=1e-9)


# Checks known mean/diff offsets are recovered exactly in the feature vector.
@pytest.mark.parametrize(
    "mean_offset,diff_offset",
    [
        ((0.1, 0.0), (0.0, 0.0)),
        ((0.0, -0.15), (0.0, 0.0)),
        ((0.05, 0.05), (0.02, -0.01)),
        ((-0.12, 0.08), (0.0, 0.03)),
    ],
)
def test_compute_gaze_feature_recovers_known_offsets(mean_offset, diff_offset):
    pts = make_eye_landmarks(mean_offset=mean_offset, diff_offset=diff_offset)
    feature = compute_gaze_feature(pts)
    expected = [*mean_offset, *diff_offset]
    np.testing.assert_allclose(feature.vector, expected, atol=1e-9)


# Checks a degenerate eye width still returns a finite feature vector.
def test_compute_gaze_feature_degenerate_eye_width_returns_zero():
    pts = make_eye_landmarks()
    pts[33] = pts[133] = np.array([100.0, 100.0])
    feature = compute_gaze_feature(pts)
    assert np.all(np.isfinite(feature.vector))


# Checks passing frame_gray=None matches omitting the argument entirely.
def test_compute_gaze_feature_frame_gray_none_matches_landmark_only_behavior():
    pts = make_eye_landmarks(mean_offset=(0.05, -0.03), diff_offset=(0.01, 0.0))
    without_arg = compute_gaze_feature(pts)
    with_none = compute_gaze_feature(pts, frame_gray=None)
    np.testing.assert_allclose(with_none.vector, without_arg.vector)


# Checks supplying frame_gray enables pupil refinement and changes the result.
def test_compute_gaze_feature_with_frame_gray_applies_pupil_refinement():
    import cv2

    pts = make_eye_landmarks(mean_offset=(0.0, 0.0), diff_offset=(0.0, 0.0))
    left_eye_center = (LEFT_EYE_OUTER_PX + LEFT_EYE_INNER_PX) / 2.0
    gray = np.full((480, 640), 180, dtype=np.uint8)
    blob_center = left_eye_center + np.array([3.0, 0.0])
    cv2.circle(gray, tuple(blob_center.astype(int)), 6, 40, -1)

    without_refine = compute_gaze_feature(pts, frame_gray=None)
    with_refine = compute_gaze_feature(pts, frame_gray=gray)

    assert not np.allclose(with_refine.vector, without_refine.vector)


# Checks gaze is usable when both eyes are open.
def test_is_gaze_usable_true_when_eyes_open():
    pts = make_eye_landmarks(eye_openness=1.0)
    assert is_gaze_usable(pts) is True


# Checks gaze is unusable when both eyes are closed.
def test_is_gaze_usable_false_when_eyes_closed():
    pts = make_eye_landmarks(eye_openness=0.0)
    assert is_gaze_usable(pts) is False


# Checks gaze is unusable when only one eye is closed.
def test_is_gaze_usable_false_when_only_one_eye_closed():
    pts = make_eye_landmarks(eye_openness=1.0)
    half_gap = 6.0 * 0.0
    left_eye_center = (pts[33] + pts[133]) / 2.0
    pts[159] = left_eye_center - np.array([0.0, half_gap])
    pts[145] = left_eye_center + np.array([0.0, half_gap])
    assert is_gaze_usable(pts) is False


# Checks is_gaze_usable respects a custom EAR threshold.
def test_is_gaze_usable_respects_custom_threshold():
    pts = make_eye_landmarks(eye_openness=1.0)
    assert is_gaze_usable(pts, min_ear=0.5) is False
    assert is_gaze_usable(pts, min_ear=0.1) is True
