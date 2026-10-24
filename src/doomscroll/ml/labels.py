"""Turning the raw cursor log into training labels (pure numpy, no camera).

"Cursor position == where the person looks" is only approximately true, and
much less so while playing a game than while following a dot. Every filter
here exists to throw away frames where that assumption is probably false:

1. LAG. The eyes land on a target ~100-250 ms before the hand gets the cursor
   there, and the camera frame is itself slightly old. So the label for the
   frame at time t is best taken from the cursor at t + lag_ms. Sweep it with
   train_model.py --sweep-lag.

2. IDLE CURSOR. A cursor sitting still says nothing about where the eyes are.
   Keep only frames where the cursor travelled >= min_travel_px nearby in time.

3. TOO FAST. Flicking the mouse across the screen (aiming in a shooter) is
   faster than the eyes follow. Drop frames whose recent cursor speed exceeds
   max_speed_px_s.

4. PAUSED. The recorder's pause hotkey marks menus/loading/chat as inactive.

5. CLICKS (optional, mode="clicks"/"either"). People look at what they click,
   so frames shortly BEFORE a button press have the most trustworthy labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABEL_MODES = ("moving", "clicks", "either")
CLICK_BEFORE_MS = 800.0  # eyes are on the target for a while before the press...
CLICK_AFTER_MS = 150.0  # ...and leave soon after it


@dataclass(frozen=True)
class LabelSettings:
    """All the knobs of label building in one place (CLI flags map 1:1 to these).

    mode: "moving" = cursor recently moving; "clicks" = near a button press;
          "either" = both kinds of frames are kept."""

    lag_ms: float = 0.0
    window_ms: float = 600.0
    min_travel_px: float = 40.0
    max_speed_px_s: float = 3000.0
    mode: str = "moving"

    def __post_init__(self):
        if self.mode not in LABEL_MODES:
            raise ValueError(f"mode must be one of {LABEL_MODES}, got {self.mode!r}")


def cursor_at(query_t_ms: np.ndarray, t_ms: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Linearly interpolate cursor (N,2) at arbitrary times. Times outside the
    recording are clamped to the first/last sample."""
    return np.column_stack([np.interp(query_t_ms, t_ms, xy[:, 0]), np.interp(query_t_ms, t_ms, xy[:, 1])])


def _travel_in_window(t_ms: np.ndarray, xy: np.ndarray, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """For each frame: (path length in px, elapsed seconds) of the cursor over a
    window of window_ms centred on it."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(seg)])  # path[i] = distance travelled up to sample i
    lo = np.searchsorted(t_ms, t_ms - window_ms / 2, side="left")
    hi = np.clip(np.searchsorted(t_ms, t_ms + window_ms / 2, side="right") - 1, 0, len(t_ms) - 1)
    return path[hi] - path[lo], (t_ms[hi] - t_ms[lo]) / 1000.0


def activity_mask(t_ms: np.ndarray, xy: np.ndarray, window_ms: float = 600.0, min_travel_px: float = 40.0) -> np.ndarray:
    """True where the cursor travelled >= min_travel_px inside the window."""
    travel, _ = _travel_in_window(t_ms, xy, window_ms)
    return travel >= min_travel_px


def speed_mask(t_ms: np.ndarray, xy: np.ndarray, window_ms: float, max_speed_px_s: float) -> np.ndarray:
    """True where the average cursor speed over the window is <= max_speed_px_s."""
    travel, seconds = _travel_in_window(t_ms, xy, window_ms)
    return travel <= max_speed_px_s * np.maximum(seconds, 1e-3)


def click_times(t_ms: np.ndarray, pressed: np.ndarray) -> np.ndarray:
    """Times (ms) of button PRESS edges (up -> down). A held button counts once."""
    p = pressed.astype(bool)
    edges = p & ~np.concatenate([[False], p[:-1]])
    return t_ms[edges]


def near_click_mask(
    t_ms: np.ndarray, pressed: np.ndarray, before_ms: float = CLICK_BEFORE_MS, after_ms: float = CLICK_AFTER_MS
) -> np.ndarray:
    """True for frames within [click - before_ms, click + after_ms] of any press."""
    mask = np.zeros(len(t_ms), dtype=bool)
    for c in click_times(t_ms, pressed):
        mask |= (t_ms >= c - before_ms) & (t_ms <= c + after_ms)
    return mask


def make_labels(
    t_ms: np.ndarray,
    cursor_xy: np.ndarray,
    screen_wh: tuple[int, int],
    settings: LabelSettings = LabelSettings(),
    pressed: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_frac, keep): per-frame label as a FRACTION of the screen
    (0..1, so models transfer between monitors) and a bool mask of frames
    whose label is trustworthy. `pressed`/`active` are per-frame 0/1 arrays
    from the recorder; omitted = no clicks recorded / always active."""
    s = settings
    shifted = cursor_at(t_ms + s.lag_ms, t_ms, cursor_xy)
    y_frac = shifted / np.array(screen_wh, dtype=np.float64)
    keep = np.all((y_frac >= 0.0) & (y_frac <= 1.0), axis=1)  # cursor on-screen

    moving = activity_mask(t_ms, cursor_xy, s.window_ms, s.min_travel_px)
    clicking = near_click_mask(t_ms, pressed) if pressed is not None else np.zeros(len(t_ms), dtype=bool)
    if s.mode == "moving":
        keep &= moving
    elif s.mode == "clicks":
        keep &= clicking
    else:
        keep &= moving | clicking

    keep &= speed_mask(t_ms, cursor_xy, s.window_ms, s.max_speed_px_s)
    if active is not None:
        keep &= active.astype(bool)
    return y_frac, keep


def screen_coverage(y_frac: np.ndarray, grid: int = 5) -> np.ndarray:
    """(grid, grid) count of labeled frames per screen cell (row = y, col = x).
    Empty cells = screen regions the model has never seen, so it can't predict there."""
    idx = np.clip((y_frac * grid).astype(int), 0, grid - 1)
    counts = np.zeros((grid, grid), dtype=int)
    np.add.at(counts, (idx[:, 1], idx[:, 0]), 1)
    return counts


def format_coverage(counts: np.ndarray) -> str:
    total = max(int(counts.sum()), 1)
    return "\n".join(" ".join(f"{100 * c / total:5.1f}%" for c in row) for row in counts)


def cursor_quality_warnings(t_ms: np.ndarray, cursor_xy: np.ndarray, screen_wh: tuple[int, int]) -> list[str]:
    """Cheap sanity checks on a raw cursor log, to catch a game that captured
    or hid the mouse (cursor frozen at one spot, or pinned to the centre)."""
    if len(t_ms) < 30:
        return ["very few frames recorded"]
    warnings = []
    _, counts = np.unique(np.round(cursor_xy).astype(np.int64), axis=0, return_counts=True)
    if counts.max() / len(t_ms) > 0.5:
        warnings.append(
            f"cursor sat on ONE position for {counts.max() / len(t_ms):.0%} of the session -- "
            "game capturing/locking the mouse? Those frames carry no gaze label."
        )
    moving = activity_mask(t_ms, cursor_xy)
    if moving.mean() < 0.25:
        warnings.append(f"cursor only moving during {moving.mean():.0%} of the session")
    return warnings
