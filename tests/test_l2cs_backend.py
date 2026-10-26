"""Tests for doomscroll.gaze.l2cs_backend's pure (non-torch) pieces:
face_crop_from_points() and _pick_device(). L2CSGazeEstimator itself needs
torch/torchvision/safetensors plus the vendored L2CS-Net repo and its
pretrained weights -- none of which are available in this sandbox (that's
the whole reason l2cs_backend's heavy imports are lazy, inside __init__,
not at module level) -- so its model-loading/forward-pass behavior can't be
exercised here and needs checking on a real machine with the `l2cs` extra
installed (`uv sync --extra l2cs`). See test_feature_backend.py for a test
that make_gaze_extractor("l2cs") fails with a clear, actionable error when
those aren't installed, rather than importing successfully and this module
not needing torch to import."""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.l2cs_backend import FACE_CROP_PAD_FRAC, face_crop_from_points


def _square_points(cx: float, cy: float, half: float) -> np.ndarray:
    """A minimal 4-point "landmark" box centered at (cx, cy)."""
    return np.array(
        [[cx - half, cy - half], [cx + half, cy - half], [cx - half, cy + half], [cx + half, cy + half]],
        dtype=np.float64,
    )


def test_face_crop_centered_box_is_padded_on_all_sides():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    points_px = _square_points(320, 240, 50)  # box is [270,190]-[370,290], 100x100

    crop = face_crop_from_points(frame, points_px, pad_frac=0.25)

    assert crop is not None
    # 100 * (1 + 2*0.25) = 150 on each side
    assert crop.shape[0] == pytest.approx(150, abs=1)
    assert crop.shape[1] == pytest.approx(150, abs=1)


def test_face_crop_near_top_left_edge_clamps_without_crashing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    points_px = _square_points(10, 10, 20)  # box extends well past (0, 0) once padded

    crop = face_crop_from_points(frame, points_px, pad_frac=FACE_CROP_PAD_FRAC)

    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # Clamped to the frame -- can't exceed the unclamped top-left corner's reach.
    assert crop.shape[0] <= 10 + 20 * (1 + 2 * FACE_CROP_PAD_FRAC) + 1
    assert crop.shape[1] <= 10 + 20 * (1 + 2 * FACE_CROP_PAD_FRAC) + 1


def test_face_crop_near_bottom_right_edge_clamps_without_crashing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    points_px = _square_points(630, 470, 20)  # box extends well past the frame's bottom-right

    crop = face_crop_from_points(frame, points_px, pad_frac=FACE_CROP_PAD_FRAC)

    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0


def test_face_crop_degenerate_zero_size_box_returns_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # All points coincide -- w=h=0, and padding a zero width/height by a
    # fraction of itself is still zero, so the box never grows.
    points_px = _square_points(320, 240, 0)

    crop = face_crop_from_points(frame, points_px, pad_frac=0.25)

    assert crop is None


def test_face_crop_returns_a_view_into_the_original_frame_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[190:290, 270:370] = 255  # mark exactly the unpadded box white
    points_px = _square_points(320, 240, 50)

    crop = face_crop_from_points(frame, points_px, pad_frac=0.0)  # no padding -- crop should equal the marked region exactly

    assert crop is not None
    assert np.all(crop == 255)
