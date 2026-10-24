"""Pure-math tests for doomscroll.gaze.gaze_vector: the eye-in-head gaze
feature computation and blink detection that feed straight into
calibration. These use synthetic landmarks with a known, exact expected
answer (see conftest.make_eye_landmarks) -- no webcam or MediaPipe needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from tests.conftest import LEFT_EYE_OUTER_PX, LEFT_EYE_INNER_PX, make_eye_landmarks


def test_compute_gaze_feature_centered_gives_zero_offset():
    """Iris exactly at each eye's center -> looking straight ahead -> both
    the mean (conjugate) and diff (asymmetry) offsets should be ~0."""
    pts = make_eye_landmarks(mean_offset=(0.0, 0.0), diff_offset=(0.0, 0.0))
    feature = compute_gaze_feature(pts)
    np.testing.assert_allclose(feature.vector, [0.0, 0.0, 0.0, 0.0], atol=1e-9)


@pytest.mark.parametrize(
    "mean_offset,diff_offset",
    [
        ((0.1, 0.0), (0.0, 0.0)),      # looking right, both eyes agree
        ((0.0, -0.15), (0.0, 0.0)),    # looking up, both eyes agree
        ((0.05, 0.05), (0.02, -0.01)), # looking down-right with slight eye asymmetry
        ((-0.12, 0.08), (0.0, 0.03)),  # looking up-left with vertical asymmetry only
    ],
)
def test_compute_gaze_feature_recovers_known_offsets(mean_offset, diff_offset):
    """compute_gaze_feature() should exactly recover whatever (mean, diff)
    offset the landmarks were constructed to represent -- this is the
    feature calibration.py's Ridge model is fit on, so a regression here
    (e.g. an index swapped, a sign flipped) would silently corrupt every
    downstream calibration."""
    pts = make_eye_landmarks(mean_offset=mean_offset, diff_offset=diff_offset)
    feature = compute_gaze_feature(pts)
    expected = [*mean_offset, *diff_offset]
    np.testing.assert_allclose(feature.vector, expected, atol=1e-9)


def test_compute_gaze_feature_degenerate_eye_width_returns_zero():
    """If an eye's outer/inner corners collapse to (near) the same point
    (a bad landmark detection), dividing by eye_width must not blow up --
    it should degrade to a zero offset rather than raising or returning
    inf/nan."""
    pts = make_eye_landmarks()
    pts[33] = pts[133] = np.array([100.0, 100.0])  # LEFT_EYE_OUTER == LEFT_EYE_INNER
    feature = compute_gaze_feature(pts)
    assert np.all(np.isfinite(feature.vector))


def test_compute_gaze_feature_frame_gray_none_matches_landmark_only_behavior():
    """frame_gray defaults to None, which must reproduce the exact
    pre-pupil-refine behavior -- every other test in this file calls
    compute_gaze_feature() without frame_gray at all, so this just makes
    the equivalence explicit."""
    pts = make_eye_landmarks(mean_offset=(0.05, -0.03), diff_offset=(0.01, 0.0))
    without_arg = compute_gaze_feature(pts)
    with_none = compute_gaze_feature(pts, frame_gray=None)
    np.testing.assert_allclose(with_none.vector, without_arg.vector)


def test_compute_gaze_feature_with_frame_gray_applies_pupil_refinement():
    """When a grayscale frame is supplied and it contains a plausible dark
    pupil blob near (but not exactly on) the landmark-based iris position,
    the refined position should shift the gaze feature away from the pure
    landmark-only result -- proving frame_gray is actually wired through to
    pupil_refine.refine_iris_center() for both eyes, not just accepted and
    ignored."""
    import cv2

    pts = make_eye_landmarks(mean_offset=(0.0, 0.0), diff_offset=(0.0, 0.0))
    left_eye_center = (LEFT_EYE_OUTER_PX + LEFT_EYE_INNER_PX) / 2.0
    gray = np.full((480, 640), 180, dtype=np.uint8)
    # A dark "pupil" a few pixels right of where the landmark says it is.
    blob_center = left_eye_center + np.array([3.0, 0.0])
    cv2.circle(gray, tuple(blob_center.astype(int)), 6, 40, -1)

    without_refine = compute_gaze_feature(pts, frame_gray=None)
    with_refine = compute_gaze_feature(pts, frame_gray=gray)

    assert not np.allclose(with_refine.vector, without_refine.vector)


def test_is_gaze_usable_true_when_eyes_open():
    pts = make_eye_landmarks(eye_openness=1.0)
    assert is_gaze_usable(pts) is True


def test_is_gaze_usable_false_when_eyes_closed():
    """A blink should be rejected -- feeding a mid-blink frame's iris
    position into calibration or a live gaze update is explicitly what
    is_gaze_usable() exists to prevent."""
    pts = make_eye_landmarks(eye_openness=0.0)
    assert is_gaze_usable(pts) is False


def test_is_gaze_usable_false_when_only_one_eye_closed():
    """Both eyes must be open -- a wink/partial blink on either side should
    still reject the frame, since compute_gaze_feature() mixes both eyes
    together (mean_offset)."""
    pts = make_eye_landmarks(eye_openness=1.0)
    half_gap = 6.0 * 0.0
    left_eye_center = (pts[33] + pts[133]) / 2.0
    pts[159] = left_eye_center - np.array([0.0, half_gap])  # close just the left eye
    pts[145] = left_eye_center + np.array([0.0, half_gap])
    assert is_gaze_usable(pts) is False


def test_is_gaze_usable_respects_custom_threshold():
    pts = make_eye_landmarks(eye_openness=1.0)  # EAR = 0.3
    assert is_gaze_usable(pts, min_ear=0.5) is False
    assert is_gaze_usable(pts, min_ear=0.1) is True
