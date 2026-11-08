# Abstract interface for facial landmark detection backends.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


# Normalized landmark points from one detected face.
@dataclass
class FaceLandmarkResult:

    points: np.ndarray
    image_width: int
    image_height: int

    # Converts normalized points to pixel coordinates.
    def points_px(self) -> np.ndarray:
        px = self.points[:, :2].copy()
        px[:, 0] *= self.image_width
        px[:, 1] *= self.image_height
        return px


# Platform/library backends implement this to detect face landmarks per frame.
class LandmarkBackend(ABC):
    # Detects face landmarks in a BGR frame, or returns None if no face found.
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarkResult | None:
        pass

    # Releases any backend resources. No-op by default.
    def close(self) -> None:
        pass
