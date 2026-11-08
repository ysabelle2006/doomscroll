# Tests for refining a MediaPipe iris estimate against a dark-pupil blob in the frame.
from __future__ import annotations

import cv2
import numpy as np
import pytest

from doomscroll.gaze.pupil_refine import refine_iris_center

# Fixed eye-corner points and frame size used across these tests.
EYE_OUTER = np.array([20.0, 40.0])
EYE_INNER = np.array([70.0, 40.0])
FRAME_SHAPE = (80, 120)


# Builds a grayscale frame with a dark circular blob at the given center.
def _frame_with_dark_blob(center: tuple[float, float], radius: int = 6, background: int = 180, blob: int = 40) -> np.ndarray:
    gray = np.full(FRAME_SHAPE, background, dtype=np.uint8)
    cv2.circle(gray, (int(center[0]), int(center[1])), radius, blob, -1)
    return gray


# Checks refinement moves a nearby estimate closer to a true dark blob.
def test_refinement_nudges_toward_a_nearby_dark_blob():
    true_center = np.array([46.0, 41.0])
    gray = _frame_with_dark_blob(tuple(true_center))
    mp_iris = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    dist_before = np.linalg.norm(mp_iris - true_center)
    dist_after = np.linalg.norm(refined - true_center)
    assert dist_after < dist_before
    assert not np.allclose(refined, mp_iris)
    assert not np.allclose(refined, true_center)


# Checks a frame with no dark blob leaves the landmark estimate unchanged.
def test_no_dark_blob_falls_back_to_landmark_unchanged():
    gray = np.full(FRAME_SHAPE, 180, dtype=np.uint8)
    mp_iris = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    np.testing.assert_array_equal(refined, mp_iris)


# Checks a landmark estimate far from any dark blob is left unchanged.
def test_large_disagreement_falls_back_to_landmark_unchanged():
    gray = _frame_with_dark_blob((46.0, 41.0))
    far_mp_iris = np.array([10.0, 10.0])

    refined = refine_iris_center(gray, far_mp_iris, EYE_OUTER, EYE_INNER)

    np.testing.assert_array_equal(refined, far_mp_iris)


# Checks coincident eye corners (degenerate width) fall back to the landmark unchanged.
def test_degenerate_eye_width_falls_back_to_landmark():
    gray = _frame_with_dark_blob((46.0, 41.0))
    mp_iris = np.array([44.0, 40.0])
    coincident = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, coincident, coincident)

    np.testing.assert_array_equal(refined, mp_iris)


# Checks an iris estimate near the frame edge doesn't crash the refinement.
@pytest.mark.parametrize("corner", [(0.0, 0.0), (119.0, 0.0), (0.0, 79.0), (119.0, 79.0)])
def test_iris_near_frame_edge_does_not_crash(corner):
    gray = np.full(FRAME_SHAPE, 180, dtype=np.uint8)
    mp_iris = np.array(corner)

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    assert refined.shape == (2,)
