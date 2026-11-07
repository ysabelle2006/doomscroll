# Records webcam frames (and optionally screen) with cursor position to disk.
from __future__ import annotations

import csv
import time
from datetime import datetime
from typing import Callable, Protocol

import cv2
import numpy as np

from doomscroll.recording.screen import ScreenRecorder
from doomscroll.recording.session import FRAME_LOG_COLUMNS, SessionMeta, SessionPaths

# Nominal output video fps and CSV flush interval.
NOMINAL_VIDEO_FPS = 30
FLUSH_EVERY_N_FRAMES = 60


# Protocol for a frame-producing camera source.
class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


# Protocol for a cursor-position source.
class CursorSource(Protocol):
    def position(self) -> tuple[float, float]: ...


# CursorSource backed by pynput's mouse controller.
class PynputCursor:

    # Creates the pynput mouse controller.
    def __init__(self):
        from pynput.mouse import Controller

        self._mouse = Controller()

    # Returns the current mouse position.
    def position(self) -> tuple[float, float]:
        x, y = self._mouse.position
        return float(x), float(y)


# Captures webcam frames (and optional screen) to video, logging timestamps/cursor to CSV.
class Recorder:
    # Stores the camera/cursor sources, output paths and session metadata.
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
        self._camera, self._cursor = camera, cursor
        self._paths, self._meta = paths, meta
        self._record_screen = record_screen
        self._clock = clock
        self._on_frame = on_frame

    # Runs the capture loop until duration_s elapses or should_stop() is true.
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
                    t_ms = (self._clock() - t0) * 1000.0
                    cx, cy = self._cursor.position()
                    if not ok or frame is None:
                        continue
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
            paths.write_meta(meta)
        return meta
