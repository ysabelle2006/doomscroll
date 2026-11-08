# Builds per-frame training labels (cursor position, on-screen + moving mask).

from __future__ import annotations

import numpy as np


# Interpolates cursor position at the given query timestamps.
def cursor_at(query_t_ms: np.ndarray, t_ms: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query_t_ms, t_ms, xy[:, 0]), np.interp(query_t_ms, t_ms, xy[:, 1])])


# True for frames where the cursor traveled at least min_travel_px within a window.
def activity_mask(t_ms: np.ndarray, xy: np.ndarray, window_ms: float = 600.0, min_travel_px: float = 40.0) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])
    lo = np.searchsorted(t_ms, t_ms - window_ms / 2, side="left")
    hi = np.clip(np.searchsorted(t_ms, t_ms + window_ms / 2, side="right") - 1, 0, len(t_ms) - 1)
    return (path[hi] - path[lo]) >= min_travel_px


# Labels each frame with a (lagged) cursor fraction and whether it's usable (on-screen + moving).
def make_labels(
    t_ms: np.ndarray,
    cursor_xy: np.ndarray,
    screen_wh: tuple[int, int],
    lag_ms: float = 0.0,
    window_ms: float = 600.0,
    min_travel_px: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = cursor_at(t_ms + lag_ms, t_ms, cursor_xy)
    y_frac = shifted / np.array(screen_wh, dtype=np.float64)
    on_screen = np.all((y_frac >= 0.0) & (y_frac <= 1.0), axis=1)
    moving = activity_mask(t_ms, cursor_xy, window_ms, min_travel_px)
    return y_frac, on_screen & moving
