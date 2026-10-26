"""Picks which gaze-vector pipeline feeds build_feature_vector(): the
original hand-crafted eye-in-head offset (gaze_vector.py) or the L2CS-Net
appearance-based estimate (l2cs_backend.py) -- see calibration.py's
FEATURE_DIMS for the resulting feature widths (10 vs 8); an offline
comparison (see the README's Gaze feature backends section) motivated
adding the second option. Both scripts/calibrate.py and scripts/phase2_gaze_scroll.py go
through make_gaze_extractor() so the one per-frame call site in each
(previously a direct compute_gaze_feature() call) doesn't need to know or
care which backend is active -- it just calls .extract(...).

Importing THIS module is always cheap and torch-free. Only
make_gaze_extractor("l2cs") actually imports l2cs_backend (and, inside
that, torch/torchvision/safetensors) -- lazily, at call time, not at this
module's import time -- so a default --features handcrafted run never
needs torch installed at all.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from doomscroll.gaze.gaze_vector import compute_gaze_feature

FEATURE_BACKEND_NAMES = ("handcrafted", "l2cs")


class GazeVectorExtractor(Protocol):
    """Common interface both backends satisfy -- the leading gaze-vector
    portion of build_feature_vector()'s input (4-dim for handcrafted, 2-dim
    for l2cs). Returns None for "no usable gaze vector this frame" (e.g. a
    failed face crop), matching how compute_gaze_feature() callers already
    treat a missing/unusable read."""

    name: str

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None: ...


class HandcraftedGazeExtractor:
    """Thin wrapper around the original compute_gaze_feature() -- unchanged
    behavior, just conforming to the shared .extract() interface so
    calibrate.py/phase2_gaze_scroll.py can treat both backends uniformly."""

    name = "handcrafted"

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
        return compute_gaze_feature(points_px, frame_gray=frame_gray).vector


class L2CSGazeExtractor:
    """Wraps l2cs_backend.L2CSGazeEstimator. Constructing this loads the
    model (see that class's docstring) -- expect a brief pause the first
    time --features l2cs is selected, not per frame."""

    name = "l2cs"

    def __init__(self) -> None:
        from doomscroll.gaze.l2cs_backend import L2CSGazeEstimator  # lazy: torch import lives here

        self._estimator = L2CSGazeEstimator()
        print(f"L2CS-Net loaded on device: {self._estimator.device}")

    def extract(self, frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
        return self._estimator.extract_from_frame(frame_bgr, points_px)


def make_gaze_extractor(name: str) -> GazeVectorExtractor:
    if name == "handcrafted":
        return HandcraftedGazeExtractor()
    if name == "l2cs":
        return L2CSGazeExtractor()
    raise ValueError(f"Unknown --features backend {name!r} -- expected one of {FEATURE_BACKEND_NAMES}")
