"""Tests for doomscroll.gaze.pupil_refine.refine_iris_center: the optional
classical-image-processing nudge applied on top of MediaPipe's landmark-level
iris position (see that module's docstring for the full rationale).

Builds small synthetic grayscale "eye crops" with a known dark blob at a
known location, rather than using real webcam frames, so the refine/fallback
behavior can be checked deterministically.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from doomscroll.gaze.pupil_refine import refine_iris_center

EYE_OUTER = np.array([20.0, 40.0])
EYE_INNER = np.array([70.0, 40.0])  # eye_width = 50.0
FRAME_SHAPE = (80, 120)


def _frame_with_dark_blob(center: tuple[float, float], radius: int = 6, background: int = 180, blob: int = 40) -> np.ndarray:
    gray = np.full(FRAME_SHAPE, background, dtype=np.uint8)
    cv2.circle(gray, (int(center[0]), int(center[1])), radius, blob, -1)
    return gray


def test_refinement_nudges_toward_a_nearby_dark_blob():
    """When MediaPipe's landmark is close to (but not exactly on) a clear
    dark pupil blob, refinement should move the estimate measurably closer
    to the blob's true center -- but only partway, per BLEND_WEIGHT."""
    true_center = np.array([46.0, 41.0])
    gray = _frame_with_dark_blob(tuple(true_center))
    mp_iris = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    dist_before = np.linalg.norm(mp_iris - true_center)
    dist_after = np.linalg.norm(refined - true_center)
    assert dist_after < dist_before
    # Blended, not fully replaced -- shouldn't land exactly on either point.
    assert not np.allclose(refined, mp_iris)
    assert not np.allclose(refined, true_center)


def test_no_dark_blob_falls_back_to_landmark_unchanged():
    """A uniform crop (no plausible pupil blob at all) must return the
    MediaPipe landmark position completely unchanged, not some arbitrary
    refined guess."""
    gray = np.full(FRAME_SHAPE, 180, dtype=np.uint8)
    mp_iris = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    np.testing.assert_array_equal(refined, mp_iris)


def test_large_disagreement_falls_back_to_landmark_unchanged():
    """If the nearest dark blob is implausibly far from where MediaPipe
    says the iris is (beyond MAX_DISAGREEMENT_EYE_WIDTHS), that's more
    likely an eyelash/shadow artifact than the real pupil -- refinement
    should distrust it and keep the landmark position as-is."""
    gray = _frame_with_dark_blob((46.0, 41.0))
    far_mp_iris = np.array([10.0, 10.0])  # far from the blob, near the eye corner

    refined = refine_iris_center(gray, far_mp_iris, EYE_OUTER, EYE_INNER)

    np.testing.assert_array_equal(refined, far_mp_iris)


def test_degenerate_eye_width_falls_back_to_landmark():
    """Coincident/degenerate eye corners (eye_width ~ 0) must not crash --
    just skip refinement entirely."""
    gray = _frame_with_dark_blob((46.0, 41.0))
    mp_iris = np.array([44.0, 40.0])
    coincident = np.array([44.0, 40.0])

    refined = refine_iris_center(gray, mp_iris, coincident, coincident)

    np.testing.assert_array_equal(refined, mp_iris)


@pytest.mark.parametrize("corner", [(0.0, 0.0), (119.0, 0.0), (0.0, 79.0), (119.0, 79.0)])
def test_iris_near_frame_edge_does_not_crash(corner):
    """The crop window can extend past the frame boundary when the iris
    estimate is near an edge -- must clip gracefully, never index out of
    bounds or crash."""
    gray = np.full(FRAME_SHAPE, 180, dtype=np.uint8)
    mp_iris = np.array(corner)

    refined = refine_iris_center(gray, mp_iris, EYE_OUTER, EYE_INNER)

    assert refined.shape == (2,)
