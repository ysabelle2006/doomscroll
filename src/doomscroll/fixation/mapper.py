# Maps a screen-space gaze point into a window-relative fraction.
from __future__ import annotations

import numpy as np

from doomscroll.windowing.base import WindowBounds


# Converts a screen point to a 0-1 fraction of the window's width/height.
def to_window_relative_fraction(screen_point: np.ndarray, bounds: WindowBounds) -> np.ndarray:
    if bounds.width <= 0 or bounds.height <= 0:
        raise ValueError(f"Degenerate window bounds, cannot map gaze point: {bounds!r}")
    fx = (screen_point[0] - bounds.x) / bounds.width
    fy = (screen_point[1] - bounds.y) / bounds.height
    return np.clip(np.array([fx, fy]), 0.0, 1.0)
