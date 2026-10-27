"""Maps a screen-space gaze point to fractional coordinates within the
focused window's bounds, so behavior is correct whether that window is
full-screen or occupies only part of the display (requirement 6)."""
from __future__ import annotations

import numpy as np

from doomscroll.windowing.base import WindowBounds


def to_window_relative_fraction(screen_point: np.ndarray, bounds: WindowBounds) -> np.ndarray:
    """Returns (fx, fy) in [0, 1] relative to the window, clamped -- a gaze
    point slightly outside the window bounds (calibration/tracking noise)
    still maps to the nearest edge rather than an out-of-range value.

    Raises ValueError for a degenerate (zero-size) window rather than
    silently dividing by zero -- callers should treat that the same as "no
    usable window bounds" (e.g. a transient OS focus report with no real
    window) rather than feeding inf/NaN into downstream smoothing state."""
    if bounds.width <= 0 or bounds.height <= 0:
        raise ValueError(f"Degenerate window bounds, cannot map gaze point: {bounds!r}")
    fx = (screen_point[0] - bounds.x) / bounds.width
    fy = (screen_point[1] - bounds.y) / bounds.height
    return np.clip(np.array([fx, fy]), 0.0, 1.0)
