"""Builds the gaze-vector portion of build_feature_vector(): the
hand-crafted eye-in-head offset (gaze_vector.py). Both scripts/calibrate.py
and scripts/phase2_gaze_scroll.py go through make_gaze_extractor() so the
one per-frame call site in each doesn't need a direct
compute_gaze_feature() call of its own.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from doomscroll.gaze.gaze_vector import compute_gaze_feature


class GazeVectorExtractor(Protocol):
    """Common interface a gaze-vector backend satisfies -- the leading
    gaze-vector portion of build_feature_vector()'s input (4-dim for the
    hand-crafted backend). Returns None for "no usable gaze vector this
    frame" (e.g. a failed face crop), matching how compute_gaze_feature()
    callers already treat a missing/unusable read."""

    name: str

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None: ...


class HandcraftedGazeExtractor:
    """Thin wrapper around compute_gaze_feature(), conforming to the
    shared .extract() interface."""

    name = "handcrafted"

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
        return compute_gaze_feature(points_px, frame_gray=frame_gray).vector


def make_gaze_extractor() -> GazeVectorExtractor:
    return HandcraftedGazeExtractor()
