"""Tests for doomscroll.gaze.feature_backend: the gaze-vector-pipeline
dispatch used by scripts/calibrate.py and scripts/phase2_gaze_scroll.py.

HandcraftedGazeExtractor is the only backend and is pure/fast, so it's
tested directly.
"""
from __future__ import annotations

import numpy as np

from doomscroll.gaze.feature_backend import (
    HandcraftedGazeExtractor,
    make_gaze_extractor,
)
from doomscroll.gaze.gaze_vector import compute_gaze_feature


def _four_point_eyes() -> np.ndarray:
    """Minimal 478-point-shaped array with just the eye/iris landmarks
    gaze_vector.py actually reads populated, matching how
    test_gaze_vector.py builds its own fixtures."""
    points = np.zeros((478, 2), dtype=np.float64)
    # left eye: outer=33, inner=133, iris center=468
    points[33] = [100.0, 100.0]
    points[133] = [120.0, 100.0]
    points[468] = [112.0, 100.0]
    # right eye: outer=263, inner=362, iris center=473
    points[263] = [220.0, 100.0]
    points[362] = [200.0, 100.0]
    points[473] = [208.0, 100.0]
    return points


def test_handcrafted_extractor_name():
    assert HandcraftedGazeExtractor().name == "handcrafted"


def test_handcrafted_extractor_matches_compute_gaze_feature_directly():
    points_px = _four_point_eyes()
    frame_gray = np.zeros((480, 640), dtype=np.uint8)
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    extractor = HandcraftedGazeExtractor()
    vector = extractor.extract(frame_bgr, frame_gray, points_px)

    expected = compute_gaze_feature(points_px, frame_gray=frame_gray).vector
    np.testing.assert_allclose(vector, expected)
    assert vector.shape == (4,)


def test_make_gaze_extractor_returns_handcrafted_instance():
    extractor = make_gaze_extractor()
    assert isinstance(extractor, HandcraftedGazeExtractor)
