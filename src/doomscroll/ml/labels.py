"""Turning the raw cursor log into training labels (pure numpy, no camera).

Two honest problems with "cursor position == where the person looks":

1. LAG. The eyes land on a target ~100-250 ms before the hand gets the cursor
   there, and the camera frame is itself slightly old. So the label for frame
   at time t is best taken from the cursor at t + lag. `lag_ms` is a
   hyper-parameter: train_model.py sweeps it automatically and keeps the one
   with the lowest held-out error (stored for the CNN's train_cnn.py to
   reuse -- see docs/ML_PIPELINE.md).

2. IDLE CURSOR. A cursor sitting still says nothing about where the eyes are
   (the person may be reading elsewhere). We keep only frames where the cursor
   has actually been moving around that moment, because that's when people
   track it with their eyes.
"""
from __future__ import annotations

import numpy as np


def cursor_at(query_t_ms: np.ndarray, t_ms: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Linearly interpolate cursor (N,2) at arbitrary times. Times outside the
    recording are clamped to the first/last sample."""
    return np.column_stack([np.interp(query_t_ms, t_ms, xy[:, 0]), np.interp(query_t_ms, t_ms, xy[:, 1])])


def activity_mask(t_ms: np.ndarray, xy: np.ndarray, window_ms: float = 600.0, min_travel_px: float = 40.0) -> np.ndarray:
    """True where the cursor travelled >= min_travel_px (path length) inside a
    window of window_ms centred on that frame."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])  # path[i] = distance travelled up to sample i
    lo = np.searchsorted(t_ms, t_ms - window_ms / 2, side="left")
    hi = np.clip(np.searchsorted(t_ms, t_ms + window_ms / 2, side="right") - 1, 0, len(t_ms) - 1)
    return (path[hi] - path[lo]) >= min_travel_px


def make_labels(
    t_ms: np.ndarray,
    cursor_xy: np.ndarray,
    screen_wh: tuple[int, int],
    lag_ms: float = 0.0,
    window_ms: float = 600.0,
    min_travel_px: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_frac, keep): per-frame label as a FRACTION of the screen
    (0..1, so models transfer between monitors) and a bool mask of frames
    whose label is trustworthy (cursor on-screen and recently moving)."""
    shifted = cursor_at(t_ms + lag_ms, t_ms, cursor_xy)
    y_frac = shifted / np.array(screen_wh, dtype=np.float64)
    on_screen = np.all((y_frac >= 0.0) & (y_frac <= 1.0), axis=1)
    moving = activity_mask(t_ms, cursor_xy, window_ms, min_travel_px)
    return y_frac, on_screen & moving
