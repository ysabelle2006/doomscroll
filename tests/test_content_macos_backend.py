# Tests for mapping Vision OCR bounding boxes to screen pixel coordinates.
from __future__ import annotations

from doomscroll.content.macos_backend import _vision_bbox_to_screen_px
from doomscroll.windowing.base import WindowBounds


# Builds a WindowBounds for testing with sensible defaults.
def _bounds(x=100, y=50, width=800, height=600) -> WindowBounds:
    return WindowBounds(x=x, y=y, width=width, height=height, app_name="test")


# Checks a full-frame box maps to the entire window.
def test_full_frame_box_maps_to_entire_window():
    bounds = _bounds()
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 1.0, 1.0), bounds)
    assert (x, y, w, h) == (bounds.x, bounds.y, bounds.width, bounds.height)


# Checks a top-left Vision box (y near 1.0) maps near the top of the window.
def test_top_left_vision_box_maps_to_top_of_window():
    bounds = _bounds()
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.9, 0.1, 0.1), bounds)
    assert x == bounds.x
    assert abs(y - bounds.y) <= 1
    assert w == bounds.width * 0.1
    assert h == bounds.height * 0.1


# Checks a bottom-left Vision box (y near 0.0) maps near the bottom of the window.
def test_bottom_left_vision_box_maps_to_bottom_of_window():
    bounds = _bounds()
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 0.1, 0.1), bounds)
    assert y == bounds.y + bounds.height * 0.9


# Checks a centered box maps to the window's vertical center regardless of flip.
def test_center_box_maps_to_center_regardless_of_flip():
    bounds = _bounds()
    _, y, _, h = _vision_bbox_to_screen_px((0.4, 0.45, 0.2, 0.1), bounds)
    assert abs((y + h / 2) - (bounds.y + bounds.height / 2)) < 1e-6


# Checks the mapping is deterministic across repeated calls.
def test_scale_invariant_to_capture_resolution():
    bounds = _bounds()
    result_a = _vision_bbox_to_screen_px((0.2, 0.3, 0.4, 0.1), bounds)
    result_b = _vision_bbox_to_screen_px((0.2, 0.3, 0.4, 0.1), bounds)
    assert result_a == result_b


# Checks an offset window's bounds are respected in the mapped result.
def test_offset_window_bounds_are_respected():
    bounds = _bounds(x=500, y=200, width=400, height=300)
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 1.0, 1.0), bounds)
    assert (x, y, w, h) == (500, 200, 400, 300)
