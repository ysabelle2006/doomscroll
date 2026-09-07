"""Webcam capture module that handles video input from a live camera or a saved file."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Standard HD resolution defaults for tracking
DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080


class WebcamSource:
    """Manages opening, reading, and releasing webcam or video file streams."""

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
        """Initializes and configures the camera stream."""
        # Use DirectShow on Windows to avoid slow camera startup
        if isinstance(self._source, int):
            if sys.platform == "win32":
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            elif sys.platform == "darwin":
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_V4L2)
        else:
            # If a file path is provided instead of a camera index
            self._cap = cv2.VideoCapture(str(self._source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam source: {self._source!r}")

        # Request higher resolution if using a real webcam for better eye tracking
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Reads the next frame from the camera. Returns (success_flag, image_array)."""
        if self._cap is None:
            raise RuntimeError("WebcamSource.open() must be called before read()")
        return self._cap.read()

    def frame_timestamp_ms(self) -> int:
        """Returns the current system time in milliseconds."""
        return int(time.time() * 1000)

    def close(self) -> None:
        """Releases the camera hardware."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamSource":
        """Allows usage with Python 'with' blocks for automatic cleanup."""
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
