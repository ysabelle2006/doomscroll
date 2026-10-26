"""Small reusable debug-overlay drawing helpers, used by the live validation
scripts and extended in later phases (content region boxes, FSM state).

The OCR-boxes-plus-gaze-dot view lives in screen_overlay.py instead of
here, as a live, click-through, transparent window drawn directly over
the real content -- rather than a cv2 window drawn onto a captured
screenshot, which is what used to be here."""
from __future__ import annotations

import cv2
import numpy as np


def draw_status_lines(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 30),
    line_height: int = 26,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_iris_points(frame: np.ndarray, points_px: np.ndarray, iris_indices: list[int]) -> None:
    """Marks the detected iris landmark points directly on the face -- shows
    where the raw eye tracking sees the pupil, independent of calibration."""
    for idx in iris_indices:
        x, y = points_px[idx].astype(int)
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def draw_gaze_minimap(
    frame: np.ndarray,
    fx: float,
    fy: float,
    panel_w: int = 180,
    panel_h: int = 120,
    margin: int = 10,
    dwelling: bool = False,
) -> None:
    """Small minimap in the frame's top-right corner representing the
    focused window, with a dot at the fractional gaze position (fx, fy)
    within it -- the calibrated "tracking point", for visually confirming
    where the system thinks you're looking relative to the window."""
    h, w = frame.shape[:2]
    x0 = w - panel_w - margin
    y0 = margin

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (200, 200, 200), 1)

    dot_x = int(x0 + np.clip(fx, 0.0, 1.0) * panel_w)
    dot_y = int(y0 + np.clip(fy, 0.0, 1.0) * panel_h)
    dot_color = (0, 165, 255) if dwelling else (0, 0, 255)
    cv2.circle(frame, (dot_x, dot_y), 6, dot_color, -1)
    cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), 1)
