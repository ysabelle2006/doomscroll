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
from doomscroll.ml.ood import OODDetector

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
    def __init__(
        self,
        estimator,
        metadata: dict | None = None,
        correction: AffineCorrection | None = None,
        ood_detector: OODDetector | None = None,
    ):
        self.estimator = estimator  # sklearn regressor: (N, FEATURE_DIM) -> (N, 2) fractions
        self.metadata = metadata or {}
        self.correction = correction
        self.ood_detector = ood_detector  # see ml/ood.py; None = no confidence gating available

    def predict_frac_stages(self, feature: np.ndarray) -> dict:
        """-> {"raw": .., "corrected": .., "clamped": .., "ood_score": .. or None}.
        One place that does the raw->correction->clamp pipeline, so predict_frac()
        and any caller that needs the intermediate values (logging, the confidence
        gate) can't drift apart."""
        raw = self.estimator.predict(feature.reshape(1, -1))[0]
        corrected = self.correction.apply(raw) if self.correction is not None else raw
        clamped = np.clip(corrected, *CLAMP_FRAC)
        score = float(self.ood_detector.score(feature.reshape(1, -1))[0]) if self.ood_detector is not None else None
        return {"raw": raw, "corrected": corrected, "clamped": clamped, "ood_score": score}

    def predict_frac(self, feature: np.ndarray) -> np.ndarray:
        return self.predict_frac_stages(feature)["clamped"]

    def predict_px(self, feature: np.ndarray, screen_wh: tuple[int, int]) -> np.ndarray:
        return self.predict_frac(feature) * np.array(screen_wh, dtype=np.float64)

    def ood_score(self, feature: np.ndarray) -> float | None:
        if self.ood_detector is None:
            return None
        return float(self.ood_detector.score(feature.reshape(1, -1))[0])

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self.estimator, "metadata": self.metadata, "feature_names": FEATURE_NAMES,
             "ood_detector": self.ood_detector},
            path,
        )

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
        return cls(data["estimator"], data["metadata"], correction, data.get("ood_detector"))


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
