"""Model selection and fitting for the gaze model.

Evaluation is the part that is easy to get wrong, so read this once:
consecutive webcam frames are near-duplicates. A random train/test split puts
neighbours of every test frame in the training set and reports a
fantastically optimistic error. We therefore always hold out WHOLE groups:
  - >= 3 sessions: leave-sessions-out (the honest "new session/person" number)
  - fewer: contiguous time blocks of BLOCK_S seconds within the sessions
Error is reported in pixels (fraction error x that session's screen size).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from doomscroll.ml.dataset import LabeledFrames
from doomscroll.ml.features import YAW_INDEX

BLOCK_S = 20.0
MIN_SESSIONS_FOR_SESSION_CV = 3
YAW_BINS_DEG = (-90, -25, -10, 10, 25, 90)

# name -> factory. Add a candidate = add a line. "ridge" is the closest
# stand-in for the old linear calibration, i.e. the baseline to beat.
CANDIDATES = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0),
    ),
    "hgb": lambda: MultiOutputRegressor(
        HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=250, random_state=0)
    ),
}


@dataclass
class Pooled:
    X: np.ndarray
    y: np.ndarray  # screen fractions
    screen_wh: np.ndarray  # (N, 2) per-sample screen size, for px errors
    groups: np.ndarray
    cv_kind: str


def pool(sessions: list[LabeledFrames]) -> Pooled:
    if not sessions:
        raise ValueError("No datasets found. Run scripts/build_dataset.py first.")
    sessions = [s for s in sessions if len(s.X) > 0]
    if not sessions:
        raise ValueError("Every frame was filtered out (no face found, or cursor never moved).")
    X = np.vstack([s.X for s in sessions])
    y = np.vstack([s.y_frac for s in sessions])
    wh = np.vstack([np.tile(s.screen_wh, (len(s.X), 1)) for s in sessions]).astype(np.float64)
    if len(sessions) >= MIN_SESSIONS_FOR_SESSION_CV:
        groups = np.concatenate([[s.session_id] * len(s.X) for s in sessions])
        kind = "leave-sessions-out"
    else:
        groups = np.concatenate([[f"{s.session_id}#{int(t // (BLOCK_S * 1000))}" for t in s.t_ms] for s in sessions])
        kind = f"{BLOCK_S:.0f}s time blocks (only {len(sessions)} session(s); collect more for a stricter test)"
    return Pooled(X, y, wh, groups, kind)


def pixel_errors(pred_frac: np.ndarray, true_frac: np.ndarray, screen_wh: np.ndarray) -> np.ndarray:
    return np.linalg.norm((pred_frac - true_frac) * screen_wh, axis=1)


def diagonal_px(screen_wh: np.ndarray) -> np.ndarray:
    """Per-sample screen diagonal, in pixels -- the normalizer for cross-participant/

    cross-resolution comparable error. Different participants recording on different
    screen resolutions (e.g. 3200x2000 vs 1512x982) make raw pixel error incomparable
    between them; dividing by each sample's own diagonal fixes that."""
    return np.linalg.norm(screen_wh, axis=1)


def normalized_errors(pred_frac: np.ndarray, true_frac: np.ndarray, screen_wh: np.ndarray) -> np.ndarray:
    """Error as a fraction of that sample's screen diagonal -- resolution-independent,
    comparable across participants/screens. Multiply by 100 for a percentage."""
    return pixel_errors(pred_frac, true_frac, screen_wh) / diagonal_px(screen_wh)


def out_of_fold_predictions(factory, data: Pooled, n_splits: int = 5) -> np.ndarray:
    n_groups = len(np.unique(data.groups))
    if n_groups < 2:
        raise ValueError("Need at least 2 groups (sessions or 20 s blocks) to cross-validate; record more data.")
    pred = np.zeros_like(data.y)
    for train, test in GroupKFold(n_splits=min(n_splits, n_groups)).split(data.X, data.y, data.groups):
        model = factory()
        model.fit(data.X[train], data.y[train])
        pred[test] = model.predict(data.X[test])
    return pred


def error_by_yaw(err_px: np.ndarray, err_norm: np.ndarray, yaw_deg: np.ndarray) -> list[dict]:
    """Mean error per head-yaw bin, in BOTH units: the direct measure of the
    head-angle problem. err_norm (fraction of screen diagonal) is the
    resolution-independent figure -- prefer it when comparing across
    participants/screens; err_px is kept for backwards comparison."""
    rows = []
    for lo, hi in zip(YAW_BINS_DEG[:-1], YAW_BINS_DEG[1:]):
        m = (yaw_deg >= lo) & (yaw_deg < hi)
        if m.any():
            rows.append({
                "label": f"yaw [{lo:+d},{hi:+d})", "n": int(m.sum()),
                "mean_px": float(err_px[m].mean()), "mean_pct": float(err_norm[m].mean()) * 100,
            })
    return rows


def evaluate_candidates(data: Pooled, names: list[str] | None = None) -> dict[str, dict]:
    """-> {name: {mean/median/p90 in both _px and _pct (= % of screen diagonal,
    the resolution-independent/primary figure -- see normalized_errors), "by_yaw"}}
    from out-of-fold predictions."""
    report = {}
    for name in names or list(CANDIDATES):
        pred = out_of_fold_predictions(CANDIDATES[name], data)
        err_px = pixel_errors(pred, data.y, data.screen_wh)
        err_norm = err_px / diagonal_px(data.screen_wh)  # same err_px, no second model pass
        report[name] = {
            "mean_px": float(err_px.mean()), "median_px": float(np.median(err_px)),
            "p90_px": float(np.percentile(err_px, 90)),
            "mean_pct": float(err_norm.mean()) * 100, "median_pct": float(np.median(err_norm)) * 100,
            "p90_pct": float(np.percentile(err_norm, 90)) * 100,
            "by_yaw": error_by_yaw(err_px, err_norm, data.X[:, YAW_INDEX]),
        }
    return report


def fit_final(name: str, data: Pooled):
    model = CANDIDATES[name]()
    model.fit(data.X, data.y)
    return model
