"""RidgeCV calibration fit directly on many frames from the continuous
anchor+raster trajectory in `scripts/quick_calibrate.py --full`, rather
than a handful of static dots.

Findings behind this design (see trajectory.py and quick_calibrate.py for
the rest of the write-up):
  - more dots alone does not help the existing 9-dot affine correction.
  - a piecewise/region-specific correction overfits at quick-calibration
    sample sizes and hurts.
  - 30-60s of continuous, good-coverage collection matches or beats static-
    dot calibration -- coverage matters more than point count.
  - RidgeCV fit directly on ~1000 collected frames is the best result so
    far, beating the affine-correction-on-top-of-the-base-model approach.

This predicts screen FRACTION directly from the same per-frame feature
vector the trained base model uses (doomscroll.ml.features.extract_features,
FEATURE_DIM=11: gaze + eyelid ratio + head rotation + head translation), so
it is a standalone replacement for TrainedGazeModel+AffineCorrection
(ml/model.py) rather than a small correction layered on top of it.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from platformdirs import user_data_dir
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

CALIBRATION_DIR = Path(user_data_dir("doomscroll")) / "calibration"
DEFAULT_FULL_CALIBRATION_PATH = CALIBRATION_DIR / "full_calibration.joblib"

MIN_FULL_CALIBRATION_SAMPLES = 200

# Same log-spaced search range as calibration.py's GazeCalibrator, for the
# same reason documented there: heavy regularization on a model this small
# compresses its dynamic range (predictions pull in short of the real
# screen edges) well before it stops helping validation error.
DEFAULT_ALPHAS = tuple(np.logspace(-3, 2, 12))


class FullCalibrator:
    """Linear RidgeCV, x and y fit independently sharing one StandardScaler
    (same reasoning as calibration.py's GazeCalibrator: sklearn's
    alpha_per_target only applies to the default per-sample LOO path, not a
    custom group-aware `cv`, and group-awareness is what matters here too --
    see fit()).

    Deliberately no polynomial expansion and no local residual-correction
    stage: those were useful in calibrate.py/calibration.py because that
    routine collects many genuinely independent head-angle point clusters.
    This routine instead collects ~1000 continuous, highly-correlated frames
    along one path -- closer to quick_calibrate.py's rejected
    piecewise-correction experiment than to calibrate.py's regime -- so a
    plain linear fit is the one that's actually supported by the data.
    """

    def __init__(self, alphas: tuple[float, ...] = DEFAULT_ALPHAS):
        self._alphas = alphas
        self._scaler = StandardScaler()
        self._ridge_x: RidgeCV | None = None
        self._ridge_y: RidgeCV | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def selected_alpha(self) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("FullCalibrator.selected_alpha accessed before fit()/load()")
        return (float(self._ridge_x.alpha_), float(self._ridge_y.alpha_))

    def fit(self, features: np.ndarray, targets_frac: np.ndarray, groups: np.ndarray | None = None) -> None:
        """features: (N, FEATURE_DIM) per-frame feature vectors (see
        ml/features.extract_features). targets_frac: (N, 2) screen (x, y) as
        fractions of screen size. groups: (N,) time-block id per sample --
        strongly recommended (see RidgeCV's cv docs and calibration.py's
        GazeCalibrator.fit for why plain per-sample LOO under-regularizes on
        near-duplicate consecutive frames); without it, falls back to
        RidgeCV's default per-sample LOO."""
        if features.shape[0] < MIN_FULL_CALIBRATION_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_FULL_CALIBRATION_SAMPLES} samples for a stable full-calibration "
                f"fit, got {features.shape[0]}"
            )
        X = self._scaler.fit_transform(features)

        cv = None
        if groups is not None:
            from sklearn.model_selection import GroupKFold

            n_groups = len(np.unique(groups))
            cv = list(GroupKFold(n_splits=min(5, n_groups)).split(X, targets_frac[:, 0], groups=groups))

        self._ridge_x = RidgeCV(alphas=self._alphas, cv=cv)
        self._ridge_y = RidgeCV(alphas=self._alphas, cv=cv)
        self._ridge_x.fit(X, targets_frac[:, 0])
        self._ridge_y.fit(X, targets_frac[:, 1])
        self._fitted = True

    def predict(self, feature: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FullCalibrator.predict() called before fit()/load()")
        X = self._scaler.transform(feature.reshape(1, -1))
        return np.array([self._ridge_x.predict(X)[0], self._ridge_y.predict(X)[0]])

    def predict_px(self, feature: np.ndarray, screen_wh: tuple[int, int]) -> np.ndarray:
        return self.predict(feature) * np.array(screen_wh, dtype=np.float64)

    def validate(self, features: np.ndarray, targets_frac: np.ndarray) -> float:
        """Mean Euclidean error as a FRACTION of the screen diagonal
        (resolution-independent -- see ml/training.py's normalized_errors)."""
        preds = np.array([self.predict(f) for f in features])
        errors = np.linalg.norm(preds - targets_frac, axis=1)
        return float(np.mean(errors)) / np.sqrt(2.0)

    def save(self, path: Path | str = DEFAULT_FULL_CALIBRATION_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "ridge_x": self._ridge_x, "ridge_y": self._ridge_y}, path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_FULL_CALIBRATION_PATH) -> "FullCalibrator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No full-calibration profile at {path}. Run `python scripts/quick_calibrate.py --full` first."
            )
        data = joblib.load(path)
        instance = cls()
        instance._scaler = data["scaler"]
        instance._ridge_x = data["ridge_x"]
        instance._ridge_y = data["ridge_y"]
        instance._fitted = True
        return instance
