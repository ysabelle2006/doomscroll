"""Tests for doomscroll.gaze.feature_backend: the --features dispatch that
lets scripts/calibrate.py and scripts/phase2_gaze_scroll.py swap between the
hand-crafted eye-vector and L2CS-Net's appearance-based estimate at one
call site, without either script needing to know which is active.

HandcraftedGazeExtractor is pure/fast and tested directly. L2CSGazeExtractor
needs torch/torchvision/safetensors (the `l2cs` extra) -- not installed in
this sandbox -- so it's tested only for the error path: make_gaze_extractor
must fail with a clear, actionable RuntimeError (pointing at `uv sync
--extra l2cs`) rather than a bare ImportError/ModuleNotFoundError, when
those aren't installed. Its actual model-loading/inference behavior needs
checking on a real machine with the extra installed -- see
test_l2cs_backend.py's module docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.feature_backend import (
    FEATURE_BACKEND_NAMES,
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


def test_feature_backend_names_lists_both_backends():
    assert set(FEATURE_BACKEND_NAMES) == {"handcrafted", "l2cs"}


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


def test_make_gaze_extractor_handcrafted_returns_handcrafted_instance():
    extractor = make_gaze_extractor("handcrafted")
    assert isinstance(extractor, HandcraftedGazeExtractor)


def test_make_gaze_extractor_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown --features backend"):
        make_gaze_extractor("not-a-real-backend")


def test_make_gaze_extractor_l2cs_without_torch_installed_raises_actionable_error():
    """In any environment without the `l2cs` extra installed (this sandbox
    included), selecting l2cs must fail loudly with a message that tells
    you how to fix it, not a bare ImportError from deep inside
    l2cs_backend.py."""
    try:
        import torch  # noqa: F401

        pytest.skip("torch is installed in this environment -- this test only applies when it's absent")
    except ImportError:
        pass

    with pytest.raises(RuntimeError, match="uv sync --extra l2cs"):
        make_gaze_extractor("l2cs")
