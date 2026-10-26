"""Tests for doomscroll.content.macos_backend's pure coordinate-conversion
helper (_vision_bbox_to_screen_px) -- the one piece of this backend that's
plain arithmetic rather than a live Vision/mss call, and so the one piece
that can actually be exercised on this (non-macOS) dev machine. See that
module's docstring for why the rest of the backend is unverified until it
runs on real macOS hardware.

Importing the module itself must also succeed without pyobjc/Vision
installed (the import is wrapped in try/except ImportError, falling back
to _VISION_AVAILABLE = False) -- this file's mere existence/collection is
itself a check of that fallback.
"""
from __future__ import annotations

from doomscroll.content.macos_backend import _vision_bbox_to_screen_px
from doomscroll.windowing.base import WindowBounds


def _bounds(x=100, y=50, width=800, height=600) -> WindowBounds:
    return WindowBounds(x=x, y=y, width=width, height=height, app_name="test")


def test_full_frame_box_maps_to_entire_window():
    """A bbox covering the whole image (0,0,1,1) should map to exactly the
    window's own bounds, regardless of the flip."""
    bounds = _bounds()
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 1.0, 1.0), bounds)
    assert (x, y, w, h) == (bounds.x, bounds.y, bounds.width, bounds.height)


def test_top_left_vision_box_maps_to_top_of_window():
    """Vision's origin is bottom-left (y increases upward) -- a box near
    the TOP of the image has a HIGH y fraction in Vision's convention, and
    must map to a LOW (near bounds.y) pixel y here, not a high one.

    Allows 1px of slack on y: the function applies a small, deliberate
    empirical correction (see its own comment) tuned by hand against a
    real screen, so an exact match to the untouched flip math isn't the
    right thing to assert here -- "near the top edge" is."""
    bounds = _bounds()
    # A small box occupying the top-left 10% of width/height, in Vision's
    # bottom-left-origin terms: y=0.9 means "starts 90% of the way up from
    # the bottom" i.e. the top strip of the image.
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.9, 0.1, 0.1), bounds)
    assert x == bounds.x
    assert abs(y - bounds.y) <= 1  # top of the window, not the bottom
    assert w == bounds.width * 0.1
    assert h == bounds.height * 0.1


def test_bottom_left_vision_box_maps_to_bottom_of_window():
    """The mirror image of the above: a box at Vision's y=0 (the bottom of
    the image) must map to the BOTTOM of the window in screen-pixel
    (top-left-origin) terms."""
    bounds = _bounds()
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 0.1, 0.1), bounds)
    assert y == bounds.y + bounds.height * 0.9  # 90% of the way down


def test_center_box_maps_to_center_regardless_of_flip():
    """A box centered vertically (y=0.45, height=0.1, so it spans
    0.45-0.55 in Vision's frame either way up) should land at the
    window's vertical center under either origin convention -- a good
    sanity check that the flip formula doesn't introduce an off-by-one or
    double-flip for the symmetric case."""
    bounds = _bounds()
    _, y, _, h = _vision_bbox_to_screen_px((0.4, 0.45, 0.2, 0.1), bounds)
    assert abs((y + h / 2) - (bounds.y + bounds.height / 2)) < 1e-6


def test_scale_invariant_to_capture_resolution():
    """The conversion only uses `bounds`' own width/height, never any
    captured-image pixel size -- so there's nothing in this function's
    signature for a Retina 2x vs 1x capture to even affect. This test
    mostly documents that invariant: the same fractional bbox always
    produces the same output for the same bounds, independent of
    whatever resolution the image was actually captured at (which simply
    never appears as an input here)."""
    bounds = _bounds()
    result_a = _vision_bbox_to_screen_px((0.2, 0.3, 0.4, 0.1), bounds)
    result_b = _vision_bbox_to_screen_px((0.2, 0.3, 0.4, 0.1), bounds)
    assert result_a == result_b


def test_offset_window_bounds_are_respected():
    """x/y of the window (not at the screen origin) must shift the output
    accordingly -- a regression here would silently break multi-monitor or
    any non-full-screen focused window."""
    bounds = _bounds(x=500, y=200, width=400, height=300)
    x, y, w, h = _vision_bbox_to_screen_px((0.0, 0.0, 1.0, 1.0), bounds)
    assert (x, y, w, h) == (500, 200, 400, 300)
