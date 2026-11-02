"""Records the webcam, and the cursor position at every webcam frame.

HOW SYNC WORKS (the whole trick): everything happens in ONE process, ONE loop,
on ONE clock. For every frame the camera hands us we immediately

    1. stamp it with t = perf_counter() - t0,
    2. read the cursor position,
    3. write the frame to webcam.mp4 and (frame_idx, t, cursor) to frames.csv.

So "frame i of the video" and "row i of the CSV" are the same instant by
construction -- there are no separate recordings to line up afterwards, and
no assumption about fps. (The mp4's own fps header is only nominal: webcams
drop/stretch frames, so ALWAYS use t_ms from the CSV, never frame_idx / fps.)

Remaining error is camera latency (the frame was exposed ~30-100 ms before we
received it) and human latency (the eye reaches a target before the hand
moves the cursor there). Both are handled as ONE tunable offset when building
the dataset (`label_lag_ms`, see ml/labels.py), not here.

The recorder does NOT run MediaPipe: keeping the loop light gives a higher,
steadier fps, and means group-mates collecting data don't need the ML setup.
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from typing import Callable, Protocol

import cv2
import numpy as np

from doomscroll.recording.screen import ScreenRecorder
from doomscroll.recording.session import FRAME_LOG_COLUMNS, SessionMeta, SessionPaths

NOMINAL_VIDEO_FPS = 30  # container header only, see module docstring
FLUSH_EVERY_N_FRAMES = 60  # a crash loses at most ~2 s of the CSV


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class CursorSource(Protocol):
    def position(self) -> tuple[float, float]: ...


class PynputCursor:
    """Global mouse position (works while another app is focused).
    macOS: needs Accessibility permission for the terminal/IDE running this."""

    def __init__(self):
        from pynput.mouse import Controller

        self._mouse = Controller()

    def position(self) -> tuple[float, float]:
        x, y = self._mouse.position
        return float(x), float(y)


class Recorder:
    def __init__(
        self,
        camera: FrameSource,
        cursor: CursorSource,
        paths: SessionPaths,
        meta: SessionMeta,
        record_screen: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        on_frame: Callable[[int, float, np.ndarray], bool] | None = None,
    ):
        """on_frame(idx, t_ms, frame) is called per frame (e.g. for a preview
        window); return True to request stop."""
        self._camera, self._cursor = camera, cursor
        self._paths, self._meta = paths, meta
        self._record_screen = record_screen
        self._clock = clock
        self._on_frame = on_frame

    def run(self, duration_s: float, should_stop: Callable[[], bool] = lambda: False) -> SessionMeta:
        paths, meta = self._paths, self._meta
        paths.create()
        meta.started_at = datetime.now().isoformat(timespec="seconds")

        t0 = self._clock()
        screen = None
        if self._record_screen:
            screen = ScreenRecorder(paths.screen_video, paths.screen_csv, self._clock, t0)
            screen.start()

        writer: cv2.VideoWriter | None = None
        n_frames = 0
        last_t_ms = 0.0
        try:
            with open(paths.frames_csv, "w", newline="") as f:
                log = csv.writer(f)
                log.writerow(FRAME_LOG_COLUMNS)
                while not should_stop():
                    ok, frame = self._camera.read()
                    t_ms = (self._clock() - t0) * 1000.0  # stamp as soon as the frame arrives
                    cx, cy = self._cursor.position()
                    if not ok or frame is None:
                        continue  # a failed read is simply not a frame: no row, no video frame
                    if t_ms > duration_s * 1000.0:
                        break

                    if writer is None:
                        h, w = frame.shape[:2]
                        meta.frame_w, meta.frame_h = w, h
                        writer = cv2.VideoWriter(
                            str(paths.webcam_video), cv2.VideoWriter_fourcc(*"mp4v"), NOMINAL_VIDEO_FPS, (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video writer for {paths.webcam_video}")
                    writer.write(frame)
                    log.writerow([n_frames, f"{t_ms:.3f}", f"{cx:.1f}", f"{cy:.1f}"])
                    last_t_ms = t_ms
                    n_frames += 1
                    if n_frames % FLUSH_EVERY_N_FRAMES == 0:
                        f.flush()
                    if self._on_frame is not None and self._on_frame(n_frames - 1, t_ms, frame):
                        break
        finally:
            if writer is not None:
                writer.release()
            if screen is not None:
                screen.stop()
            meta.n_frames = n_frames
            meta.duration_s = last_t_ms / 1000.0
            meta.measured_fps = n_frames / meta.duration_s if meta.duration_s > 0 else 0.0
            meta.has_screen_video = screen is not None and screen.error is None and paths.screen_video.exists()
            if screen is not None and screen.error is not None:
                meta.extra["screen_error"] = repr(screen.error)
            paths.write_meta(meta)  # written even on Ctrl+C / crash, so partial sessions stay usable
        return meta
