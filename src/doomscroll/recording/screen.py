# Background thread that periodically captures the screen to a video file.
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doomscroll.recording.session import SCREEN_LOG_COLUMNS

# Capture rate and output frame width for the screen recording.
SCREEN_FPS = 10
SCREEN_OUT_WIDTH = 960


# Finds the index of the closest timestamp in a sorted array.
def nearest_index(query_t_ms: float, sorted_t_ms: np.ndarray) -> int:
    i = int(np.searchsorted(sorted_t_ms, query_t_ms))
    if i == 0:
        return 0
    if i == len(sorted_t_ms):
        return len(sorted_t_ms) - 1
    return i if sorted_t_ms[i] - query_t_ms < query_t_ms - sorted_t_ms[i - 1] else i - 1


# Grabs screenshots on a timer, writing frames to video and timestamps to CSV.
class ScreenRecorder:
    # Stores paths/clock/fps and prepares the background capture thread.
    def __init__(self, video_path: Path, csv_path: Path, clock: Callable[[], float], t0: float, fps: int = SCREEN_FPS):
        self._video_path, self._csv_path = video_path, csv_path
        self._clock, self._t0, self._fps = clock, t0, fps
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error: Exception | None = None

    # Starts the capture thread.
    def start(self) -> None:
        self._thread.start()

    # Signals the capture thread to stop and waits for it to finish.
    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # Capture loop: grabs, resizes and writes a frame each tick, logging its timestamp.
    def _run(self) -> None:
        try:
            import mss

            with mss.mss() as grabber, open(self._csv_path, "w", newline="") as f:
                log = csv.writer(f)
                log.writerow(SCREEN_LOG_COLUMNS)
                monitor = grabber.monitors[1]
                writer = None
                idx = 0
                period = 1.0 / self._fps
                next_tick = self._clock()
                while not self._stop.is_set():
                    shot = np.asarray(grabber.grab(monitor))[:, :, :3]
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
        except Exception as e:
            self.error = e
