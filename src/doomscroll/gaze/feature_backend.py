# Interface for extracting a per-frame gaze feature vector from face landmarks.

from __future__ import annotations

from typing import Protocol

import numpy as np

from doomscroll.gaze.gaze_vector import compute_gaze_feature


# Interface implemented by any gaze-feature extractor.
class GazeVectorExtractor(Protocol):

    name: str

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None: ...


# Extracts a gaze feature vector from handcrafted eye-geometry features.
class HandcraftedGazeExtractor:

    name = "handcrafted"

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
        return compute_gaze_feature(points_px, frame_gray=frame_gray).vector


# Returns the extractor used by the math-model pipeline.
def make_gaze_extractor() -> GazeVectorExtractor:
    return HandcraftedGazeExtractor()
