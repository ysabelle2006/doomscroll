# Loads/saves trained gaze models (ridge/CNN) and per-user affine corrections.

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
CLAMP_FRAC = (-0.25, 1.25)


# A small per-user affine fit applied on top of a model's raw prediction.
@dataclass
class AffineCorrection:

    W: np.ndarray

    # Returns a no-op (pass-through) correction.
    @classmethod
    def identity(cls) -> "AffineCorrection":
        return cls(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    # Fits a ridge-regularized affine map from predicted to true screen fractions.
    @classmethod
    def fit(cls, predicted: np.ndarray, truth: np.ndarray, shrink: float = 1.0) -> "AffineCorrection":
        P = np.column_stack([predicted, np.ones(len(predicted))])
        W0 = cls.identity().W
        W = np.linalg.solve(P.T @ P + shrink * np.eye(3), P.T @ truth + shrink * W0)
        return cls(W)

    # Applies the correction to one predicted screen fraction.
    def apply(self, frac: np.ndarray) -> np.ndarray:
        return np.append(frac, 1.0) @ self.W

    # Saves the correction to disk.
    def save(self, path: Path = DEFAULT_CORRECTION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"W": self.W.tolist()}))

    # Loads a saved correction from disk.
    @classmethod
    def load(cls, path: Path = DEFAULT_CORRECTION_PATH) -> "AffineCorrection":
        return cls(np.array(json.loads(Path(path).read_text())["W"]))


# A trained ridge/hgb gaze model plus its optional per-user correction.
class TrainedGazeModel:
    def __init__(self, estimator, metadata: dict | None = None, correction: AffineCorrection | None = None):
        self.estimator = estimator
        self.metadata = metadata or {}
        self.correction = correction

    # Predicts a screen-fraction point from one feature vector.
    def predict_frac(self, feature: np.ndarray) -> np.ndarray:
        frac = self.estimator.predict(feature.reshape(1, -1))[0]
        if self.correction is not None:
            frac = self.correction.apply(frac)
        return np.clip(frac, *CLAMP_FRAC)

    # Predicts a screen-pixel point from one feature vector.
    def predict_px(self, feature: np.ndarray, screen_wh: tuple[int, int]) -> np.ndarray:
        return self.predict_frac(feature) * np.array(screen_wh, dtype=np.float64)

    # Extracts features from a frame, then predicts a screen-fraction point.
    def predict_frac_frame(self, points_px: np.ndarray, pose, gray: np.ndarray | None) -> np.ndarray:
        from doomscroll.ml.features import extract_features

        return self.predict_frac(extract_features(points_px, pose, gray))

    # Saves the model to disk.
    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self.estimator, "metadata": self.metadata, "feature_names": FEATURE_NAMES}, path)

    # Loads a saved model, plus a correction if one exists at correction_path.
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


# Fits a user correction and reports its leave-one-out validation error vs. uncorrected.
def fit_user_correction(
    predicted: np.ndarray, truth: np.ndarray, screen_wh: tuple[int, int], shrink: float = 1.0
) -> tuple[AffineCorrection, float, float]:
    wh = np.array(screen_wh, dtype=np.float64)
    before = float(np.linalg.norm((predicted - truth) * wh, axis=1).mean())
    loo = []
    for i in range(len(predicted)):
        keep = np.arange(len(predicted)) != i
        c = AffineCorrection.fit(predicted[keep], truth[keep], shrink)
        loo.append(np.linalg.norm((c.apply(predicted[i]) - truth[i]) * wh))
    return AffineCorrection.fit(predicted, truth, shrink), before, float(np.mean(loo))


# Returns the per-user correction file path for a given model path.
def correction_path_for(model_path: Path | str) -> Path:
    if Path(model_path).suffix == ".pt":
        return DEFAULT_CORRECTION_PATH.with_name("affine_correction_cnn.json")
    return DEFAULT_CORRECTION_PATH


# Loads a trained model (ridge/hgb or CNN, by file extension) with its correction.
def load_gaze_model(path: Path | str, with_correction: bool = True):
    path = Path(path)
    corr = correction_path_for(path) if with_correction else None
    if path.suffix == ".pt":
        from doomscroll.ml.cnn import CnnGazeModel

        return CnnGazeModel.load(path, correction_path=corr)
    return TrainedGazeModel.load(path, correction_path=corr)
