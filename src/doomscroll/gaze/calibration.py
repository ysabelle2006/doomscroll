# Fits a per-user gaze-point predictor: polynomial RidgeCV plus a local
# nearest-neighbor residual correction, saved/loaded from disk.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from platformdirs import user_data_dir
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CALIBRATION_DIR = Path(user_data_dir("gazescroll")) / "calibration"
DEFAULT_PROFILE_PATH = CALIBRATION_DIR / "default.joblib"

FEATURE_DIM = 10

MIN_CALIBRATION_SAMPLES = 30
MAX_LEAVE_ONE_GROUP_OUT_GROUPS = 20

DEFAULT_ALPHAS = tuple(np.logspace(-3, 2, 12))

LOCAL_CORRECTION_K = 8
LOCAL_CORRECTION_BANDWIDTH = 1.5
LOCAL_CORRECTION_MAX_WEIGHT = 0.5


# Concatenates gaze, head-pose, and head-translation features into one vector.
def build_feature_vector(
    gaze_vector: np.ndarray,
    head_pose_deg: tuple[float, float, float],
    head_translation_mm: tuple[float, float, float],
) -> np.ndarray:
    return np.concatenate(
        [
            gaze_vector,
            np.array(head_pose_deg, dtype=np.float64),
            np.array(head_translation_mm, dtype=np.float64),
        ]
    )


# Predicts a screen point from a gaze feature vector: global Ridge fit + local correction.
class GazeCalibrator:

    def __init__(
        self,
        alphas: tuple[float, ...] = DEFAULT_ALPHAS,
        poly_degree: int = 1,
        local_correction_k: int = LOCAL_CORRECTION_K,
        local_correction_bandwidth: float = LOCAL_CORRECTION_BANDWIDTH,
        local_correction_max_weight: float = LOCAL_CORRECTION_MAX_WEIGHT,
    ):
        self._alphas = alphas
        self._preprocessor = Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=True)),
            ]
        )
        self._ridge_x: RidgeCV | None = None
        self._ridge_y: RidgeCV | None = None
        self._fitted = False

        self._local_k = local_correction_k
        self._local_bandwidth = local_correction_bandwidth
        self._local_max_weight = local_correction_max_weight
        self._train_X: np.ndarray | None = None
        self._train_residuals: np.ndarray | None = None

    # True once fit() or load() has set up the model.
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # The RidgeCV-selected regularization strength for x and y.
    @property
    def selected_alpha(self) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("GazeCalibrator.selected_alpha accessed before fit()/load()")
        return (float(self._ridge_x.alpha_), float(self._ridge_y.alpha_))

    # Fits the polynomial Ridge regressors and records training residuals for local correction.
    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray | None = None) -> None:
        if features.shape[0] < MIN_CALIBRATION_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_CALIBRATION_SAMPLES} samples for a stable polynomial "
                f"Ridge fit, got {features.shape[0]}"
            )

        X = self._preprocessor.fit_transform(features)

        cv = None
        if groups is not None:
            n_groups = len(np.unique(groups))
            splitter = (
                LeaveOneGroupOut()
                if n_groups <= MAX_LEAVE_ONE_GROUP_OUT_GROUPS
                else GroupKFold(n_splits=5)
            )
            cv = list(splitter.split(X, targets[:, 0], groups=groups))

        self._ridge_x = RidgeCV(alphas=self._alphas, cv=cv)
        self._ridge_y = RidgeCV(alphas=self._alphas, cv=cv)
        self._ridge_x.fit(X, targets[:, 0])
        self._ridge_y.fit(X, targets[:, 1])
        self._fitted = True

        global_preds = np.column_stack([self._ridge_x.predict(X), self._ridge_y.predict(X)])
        self._train_X = X
        self._train_residuals = targets - global_preds

    # Returns a small correction based on the nearest training points' residuals.
    def _local_correction(self, X_query: np.ndarray) -> np.ndarray:
        if self._train_X is None or len(self._train_X) < self._local_k:
            return np.zeros(2)

        dists = np.linalg.norm(self._train_X - X_query, axis=1)
        nearest_idx = np.argsort(dists)[: self._local_k]
        nearest_dists = dists[nearest_idx]

        weights = np.exp(-(nearest_dists**2) / (2 * self._local_bandwidth**2))
        weight_sum = float(weights.sum())
        if weight_sum < 1e-8:
            return np.zeros(2)

        weighted_residual = (weights[:, None] * self._train_residuals[nearest_idx]).sum(axis=0) / weight_sum
        confidence = float(np.clip(weights.max(), 0.0, 1.0)) * self._local_max_weight
        return confidence * weighted_residual

    # Predicts a screen point for one feature vector.
    def predict(self, feature: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GazeCalibrator.predict() called before fit()/load()")
        X = self._preprocessor.transform(feature.reshape(1, -1))
        global_pred = np.array([self._ridge_x.predict(X)[0], self._ridge_y.predict(X)[0]])
        return global_pred + self._local_correction(X)

    # Returns the mean pixel error of predictions against held-out targets.
    def validate(self, features: np.ndarray, targets: np.ndarray) -> float:
        predictions = np.array([self.predict(f) for f in features])
        errors = np.linalg.norm(predictions - targets, axis=1)
        return float(np.mean(errors))

    # Saves the fitted model to disk.
    def save(self, path: Path | str = DEFAULT_PROFILE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self._preprocessor,
                "ridge_x": self._ridge_x,
                "ridge_y": self._ridge_y,
                "local_k": self._local_k,
                "local_bandwidth": self._local_bandwidth,
                "local_max_weight": self._local_max_weight,
                "train_X": self._train_X,
                "train_residuals": self._train_residuals,
            },
            path,
        )

    # Loads a previously saved model from disk.
    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE_PATH) -> "GazeCalibrator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No calibration profile at {path}. Run scripts/calibrate.py first.")
        data = joblib.load(path)
        instance = cls(
            local_correction_k=data.get("local_k", LOCAL_CORRECTION_K),
            local_correction_bandwidth=data.get("local_bandwidth", LOCAL_CORRECTION_BANDWIDTH),
            local_correction_max_weight=data.get("local_max_weight", LOCAL_CORRECTION_MAX_WEIGHT),
        )
        instance._preprocessor = data["preprocessor"]
        instance._ridge_x = data["ridge_x"]
        instance._ridge_y = data["ridge_y"]
        instance._train_X = data.get("train_X")
        instance._train_residuals = data.get("train_residuals")
        instance._fitted = True
        return instance
