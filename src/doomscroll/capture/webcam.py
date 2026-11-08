# Cross-platform webcam capture source using OpenCV.
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Default capture resolution requested from the camera.
DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080


# Opens and reads frames from a webcam using the platform's OpenCV backend.
class WebcamSource:

    # Stores the camera source and desired capture resolution.
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

    # Opens the camera with the platform-appropriate backend and sets resolution.
    def open(self) -> None:
        if isinstance(self._source, int):
            if sys.platform == "win32":
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            elif sys.platform == "darwin":
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(str(self._source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam source: {self._source!r}")

        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)

    # Reads the next frame from the camera.
    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            raise RuntimeError("WebcamSource.open() must be called before read()")
        return self._cap.read()

    # Returns the current wall-clock time in milliseconds.
    def frame_timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    # Releases the camera.
    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Opens the camera on context-manager entry.
    def __enter__(self) -> "WebcamSource":
        self.open()
        return self

    # Closes the camera on context-manager exit.
    def __exit__(self, *exc) -> None:
        self.close()
