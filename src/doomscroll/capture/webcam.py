"""Webcam frame source, with a file-path source for offline test fixtures."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080


class WebcamSource:
    """Yields BGR frames from a live camera or a recorded video file.

    A file source is accepted (rather than only a device index) so the same
    pipeline can be replayed against recorded fixtures in tests, per the
    project's testing strategy.
    """

    def __init__(
        self,
        source: int | str | Path = 0,
        capture_width: int = DEFAULT_CAPTURE_WIDTH,
        capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    ):
        self._source = source
        self._capture_width = capture_width
        self._capture_height = capture_height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if isinstance(self._source, int) and sys.platform == "win32":
            # CAP_DSHOW avoids the slow default MSMF backend init on Windows.
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        elif isinstance(self._source, int):
            # Pass the device index as-is (e.g. via AVFoundation on macOS).
            # Stringifying an int here turns 0 into the *filename* "0",
            # which OpenCV then fails to open as a video file.
            self._cap = cv2.VideoCapture(self._source)
        else:
            self._cap = cv2.VideoCapture(str(self._source))
            
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam source: {self._source!r}")
        
        if isinstance(self._source, int):
            # The OpenCV default (often 640x480) leaves the iris spanning
            # only a handful of pixels, capping tracking precision regardless
            # of downstream calibration quality. Request a high-res capture;
            # if the device doesn't support it, it silently keeps whatever
            # the closest supported mode is.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            raise RuntimeError("WebcamSource.open() must be called before read()")
        return self._cap.read()

    def frame_timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
