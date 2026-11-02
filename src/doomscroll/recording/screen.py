"""Optional screen recorder (background thread, uses `mss`).

NOT needed for training the gaze model -- the cursor log is the label. It is
useful for eyeballing a session afterwards ("was the participant actually
doing the task here?") and for later content-aware work. Frames are stamped
with the same clock as the webcam (see Recorder), so syncing is a lookup:
`nearest_index(screen_t_ms, webcam_t_ms)`.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doomscroll.recording.session import SCREEN_LOG_COLUMNS

SCREEN_FPS = 10
SCREEN_OUT_WIDTH = 960  # downscaled: full-res screen video gets huge for no benefit


def nearest_index(query_t_ms: float, sorted_t_ms: np.ndarray) -> int:
    """Index of the timestamp in sorted_t_ms closest to query_t_ms."""
    i = int(np.searchsorted(sorted_t_ms, query_t_ms))
    if i == 0:
        return 0
    if i == len(sorted_t_ms):
        return len(sorted_t_ms) - 1
    return i if sorted_t_ms[i] - query_t_ms < query_t_ms - sorted_t_ms[i - 1] else i - 1


class ScreenRecorder:
    def __init__(self, video_path: Path, csv_path: Path, clock: Callable[[], float], t0: float, fps: int = SCREEN_FPS):
        self._video_path, self._csv_path = video_path, csv_path
        self._clock, self._t0, self._fps = clock, t0, fps
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: Exception | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            import mss  # imported here so webcam-only recording doesn't need it

            with mss.mss() as grabber, open(self._csv_path, "w", newline="") as f:
                log = csv.writer(f)
                log.writerow(SCREEN_LOG_COLUMNS)
                monitor = grabber.monitors[1]  # primary display
                writer = None
                idx = 0
                period = 1.0 / self._fps
                next_tick = self._clock()
                while not self._stop.is_set():
                    shot = np.asarray(grabber.grab(monitor))[:, :, :3]  # BGRA -> BGR
                    t_ms = (self._clock() - self._t0) * 1000.0
                    h, w = shot.shape[:2]
                    out_w, out_h = SCREEN_OUT_WIDTH, int(h * SCREEN_OUT_WIDTH / w)
                    small = cv2.resize(shot, (out_w, out_h), interpolation=cv2.INTER_AREA)
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (out_w, out_h)
                        )
                    writer.write(small)
                    log.writerow([idx, f"{t_ms:.3f}"])
                    idx += 1
                    next_tick += period
                    time.sleep(max(0.0, next_tick - self._clock()))
                if writer is not None:
                    writer.release()
        except Exception as e:  # surfaced by Recorder after the run, never kills webcam recording
            self.error = e
