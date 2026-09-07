"""Screen-space gaze mapping: Ridge regression from
[eye-in-head vector (4), head rotation (3), head translation (3)] ->
(screen_x, screen_y).

Including head pose -- both rotation (yaw/pitch/roll) and translation
(the solvePnP position vector, i.e. where the head sits relative to the
camera) -- as regressor inputs, rather than geometrically subtracting them
out, is what lets the model absorb natural head-pose variation captured
during calibration, so small incidental head motion at runtime doesn't
require recalibration -- see the project plan's calibration design
rationale. Translation matters here as much as rotation: the eye-in-head
offset for a fixed gaze target changes as the head moves nearer/farther or
side-to-side relative to the screen, and that shift was previously
invisible to the model entirely (only rotation was fed in). See
calibrate.py's head-pose variation pass for how meaningful variation in
both is captured during calibration rather than left at near-zero.
"""
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

FEATURE_DIM = 10  # 4 (eye-in-head) + 3 (yaw, pitch, roll) + 3 (head translation x,y,z mm)
MIN_CALIBRATION_SAMPLES = 30
MAX_LEAVE_ONE_GROUP_OUT_GROUPS = 20  # switch to GroupKFold beyond this many points

# Log-spaced search range handed to RidgeCV. Kept much smaller now that the
# default model is linear (11 params incl. bias, not the 36+-term polynomial
# this range was originally sized for): heavy regularization shrinks the
# fitted mapping's slope/gain, which technically minimizes validation MSE on
# noisy calibration data but compresses the model's dynamic range in
# practice -- observed as needing to look far past the real screen edge
# (e.g. down at the trackpad) before the mapped point reaches the actual
# edge. A lower ceiling keeps RidgeCV from reaching for regularization that
# strong on a model this small.
DEFAULT_ALPHAS = tuple(np.logspace(-3, 2, 12))


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


class GazeCalibrator:
    """Degree-1 (linear) by default. Degree-2 polynomial expansion was tried
    first (it captures the real nonlinearity in eye-rotation-to-screen
    mapping, and in principle improves accuracy near the edges), but in
    practice a 36-term quadratic model fit on only ~16 independent
    calibration points is prone to extrapolating wildly for any live input
    that isn't close to one of those 16 clusters -- observed as the
    predicted point pinning to a screen corner on most frames, regardless of
    regularization strategy (RidgeCV, group-aware CV). A bounded, somewhat
    less accurate linear fit is the safer choice for this small-sample-per-
    session regime -- this is also what lightweight webcam gaze trackers
    like WebGazer.js use for the same reason. Revisit degree=2 if calibration
    ever collects substantially more independent points, or move to an
    appearance-based model instead of chasing more polynomial capacity here.

    StandardScaler runs first: the raw feature vector mixes a unitless
    gaze-offset signal (~+-0.1 range) with head-rotation degrees (~+-30-50
    range) and head-translation millimeters (~hundreds), and Ridge's L2
    penalty treats coefficient magnitude uniformly -- without scaling, the
    small-scale (but most informative) gaze signal needs large coefficients
    and gets disproportionately penalized relative to head pose, which is
    backwards from what the model should learn.

    Regularization strength is chosen by cross-validation over DEFAULT_ALPHAS
    rather than a hand-picked fixed alpha -- but critically, that CV must be
    *group-aware* (fit() takes a `groups` array, one id per calibration
    point). Plain per-sample leave-one-out is invalid here: the ~15-20 frames
    captured per calibration point are highly correlated near-duplicates, so
    ordinary LOO leaves near-identical frames from the same point in the
    training fold, making an overfit model look like it generalizes
    perfectly. That systematically selects too little regularization, and
    the resulting degree-2 polynomial model then extrapolates wildly (huge
    x/y values, clamped to a screen corner after fraction-clipping) for any
    live input that isn't essentially an exact calibration point. Leaving out
    one whole calibration point per fold forces the model to generalize to
    a genuinely new head/eye position, matching how it's actually used.

    x and y are fit as two independent RidgeCV models sharing one
    preprocessing pipeline (rather than one multi-output RidgeCV with
    alpha_per_target) because sklearn's alpha_per_target optimization only
    applies to the default fast leave-one-out path, not to an explicit
    custom `cv` -- and a custom group-aware cv is exactly the point here.
    """

    def __init__(self, alphas: tuple[float, ...] = DEFAULT_ALPHAS, poly_degree: int = 1):
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

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def selected_alpha(self) -> tuple[float, float]:
        """The alpha RidgeCV chose via cross-validation, per output axis
        (x, y) -- useful to sanity-check calibration quality. A value at or
        near the top of DEFAULT_ALPHAS' range is a sign the fit is still
        struggling to generalize even with heavy regularization."""
        if not self._fitted:
            raise RuntimeError("GazeCalibrator.selected_alpha accessed before fit()/load()")
        return (float(self._ridge_x.alpha_), float(self._ridge_y.alpha_))

    def fit(self, features: np.ndarray, targets: np.ndarray, groups: np.ndarray | None = None) -> None:
        """features: (N, FEATURE_DIM), targets: (N, 2) screen (x, y) px.
        groups: (N,) calibration-point id per sample -- strongly recommended
        (see class docstring); without it, falls back to RidgeCV's default
        per-sample LOO, which is known to under-regularize on this data."""
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

    def predict(self, feature: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GazeCalibrator.predict() called before fit()/load()")
        X = self._preprocessor.transform(feature.reshape(1, -1))
        return np.array([self._ridge_x.predict(X)[0], self._ridge_y.predict(X)[0]])

    def validate(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Mean Euclidean pixel error on held-out (features, targets)."""
        X = self._preprocessor.transform(features)
        predictions = np.column_stack([self._ridge_x.predict(X), self._ridge_y.predict(X)])
        errors = np.linalg.norm(predictions - targets, axis=1)
        return float(np.mean(errors))

    def save(self, path: Path | str = DEFAULT_PROFILE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"preprocessor": self._preprocessor, "ridge_x": self._ridge_x, "ridge_y": self._ridge_y},
            path,
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE_PATH) -> "GazeCalibrator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No calibration profile at {path}. Run scripts/calibrate.py first.")
        data = joblib.load(path)
        instance = cls()
        instance._preprocessor = data["preprocessor"]
        instance._ridge_x = data["ridge_x"]
        instance._ridge_y = data["ridge_y"]
        instance._fitted = True
        return instance
