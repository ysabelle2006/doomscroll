"""Common interface for face-landmark backends, so mediapipe can be swapped
for a fallback (dlib, OpenCV LBF) later without touching downstream code."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceLandmarkResult:
    """Landmark points in normalized [0,1] image coordinates (x, y, z)."""

    points: np.ndarray  # shape (N, 3)
    image_width: int
    image_height: int

    def points_px(self) -> np.ndarray:
        """Landmark points in pixel coordinates (x, y), shape (N, 2)."""
        px = self.points[:, :2].copy()
        px[:, 0] *= self.image_width
        px[:, 1] *= self.image_height
        return px


class LandmarkBackend(ABC):
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarkResult | None:
        """Return landmarks for the largest/first detected face, or None."""

    def close(self) -> None:
        pass
