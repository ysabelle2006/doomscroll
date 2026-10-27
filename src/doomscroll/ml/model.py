"""The trained gaze model as the rest of the app uses it.

    model = TrainedGazeModel.load("models/gaze_model.joblib")
    x_px, y_px = model.predict_px(extract_features(...), (screen_w, screen_h))

It predicts screen FRACTIONS (0..1) so one model serves any monitor.

Optional per-user AffineCorrection: a handful of calibration dots (~15 s)
fit a tiny "nudge" (scale+shift) on top of the shared model, instead of the
old 80-point calibration having to learn the whole eye->screen mapping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from platformdirs import user_data_dir

from doomscroll.ml.features import FEATURE_NAMES

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "gaze_model.joblib"
DEFAULT_CORRECTION_PATH = Path(user_data_dir("doomscroll")) / "calibration" / "affine_correction.json"
CLAMP_FRAC = (-0.25, 1.25)  # allow slightly off-screen predictions, never wild ones


@dataclass
class AffineCorrection:
    """frac_out = A @ frac_in + b, stored as a 3x2 matrix W: [x, y, 1] @ W."""

    W: np.ndarray

    @classmethod
    def identity(cls) -> "AffineCorrection":
        return cls(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    @classmethod
    def fit(cls, predicted: np.ndarray, truth: np.ndarray, shrink: float = 1.0) -> "AffineCorrection":
        """Ridge toward the identity: with few points the fit stays close to
        'no correction' instead of overfitting; with more points data wins.
        predicted, truth: (n, 2) fractions. shrink: pseudo-observations of identity."""
        P = np.column_stack([predicted, np.ones(len(predicted))])
        W0 = cls.identity().W
        W = np.linalg.solve(P.T @ P + shrink * np.eye(3), P.T @ truth + shrink * W0)
        return cls(W)

    def apply(self, frac: np.ndarray) -> np.ndarray:
        return np.append(frac, 1.0) @ self.W

    def save(self, path: Path = DEFAULT_CORRECTION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"W": self.W.tolist()}))

    @classmethod
    def load(cls, path: Path = DEFAULT_CORRECTION_PATH) -> "AffineCorrection":
        return cls(np.array(json.loads(Path(path).read_text())["W"]))


class TrainedGazeModel:
    def __init__(self, estimator, metadata: dict | None = None, correction: AffineCorrection | None = None):
        self.estimator = estimator  # sklearn regressor: (N, FEATURE_DIM) -> (N, 2) fractions
        self.metadata = metadata or {}
        self.correction = correction

    def predict_frac(self, feature: np.ndarray) -> np.ndarray:
        frac = self.estimator.predict(feature.reshape(1, -1))[0]
        if self.correction is not None:
            frac = self.correction.apply(frac)
        return np.clip(frac, *CLAMP_FRAC)

    def predict_px(self, feature: np.ndarray, screen_wh: tuple[int, int]) -> np.ndarray:
        return self.predict_frac(feature) * np.array(screen_wh, dtype=np.float64)

    def predict_frac_frame(self, points_px: np.ndarray, pose, gray: np.ndarray | None) -> np.ndarray:
        """Same call shape as CnnGazeModel, so scripts can use either (see load_gaze_model)."""
        from doomscroll.ml.features import extract_features

        return self.predict_frac(extract_features(points_px, pose, gray))

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": self.metadata, "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH, correction_path: Path | str | None = None) -> "TrainedGazeModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run scripts/train_model.py first.")
        data = joblib.load(path)
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Model was trained with different features than the current code; retrain.")
        correction = None
        if correction_path is not None and Path(correction_path).exists():
            correction = AffineCorrection.load(Path(correction_path))
        return cls(data["estimator"], data["metadata"], correction)


def fit_user_correction(
    predicted: np.ndarray, truth: np.ndarray, screen_wh: tuple[int, int], shrink: float = 1.0
) -> tuple[AffineCorrection, float, float]:
    """Fit the per-user correction from quick-calibration dots and report
    (correction, error_before_px, leave_one_out_error_after_px).
    The LOO number is the honest one: each dot is predicted by a correction
    fitted WITHOUT it. If it isn't lower than 'before', don't use the correction."""
    wh = np.array(screen_wh, dtype=np.float64)
    before = float(np.linalg.norm((predicted - truth) * wh, axis=1).mean())
    loo = []
    for i in range(len(predicted)):
        keep = np.arange(len(predicted)) != i
        c = AffineCorrection.fit(predicted[keep], truth[keep], shrink)
        loo.append(np.linalg.norm((c.apply(predicted[i]) - truth[i]) * wh))
    return AffineCorrection.fit(predicted, truth, shrink), before, float(np.mean(loo))


def correction_path_for(model_path: Path | str) -> Path:
    """Each model gets its own per-user correction (a nudge fitted for one model is wrong for another)."""
    if Path(model_path).suffix == ".pt":
        return DEFAULT_CORRECTION_PATH.with_name("affine_correction_cnn.json")
    return DEFAULT_CORRECTION_PATH


def load_gaze_model(path: Path | str, with_correction: bool = True):
    """.joblib -> TrainedGazeModel (sklearn on landmark features); .pt -> CnnGazeModel (needs torch).
    Both offer predict_frac_frame(points_px, pose, gray) and .metadata / .correction."""
    path = Path(path)
    corr = correction_path_for(path) if with_correction else None
    if path.suffix == ".pt":
        from doomscroll.ml.cnn import CnnGazeModel  # torch only needed for the CNN

        return CnnGazeModel.load(path, correction_path=corr)
    return TrainedGazeModel.load(path, correction_path=corr)
