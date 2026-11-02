"""Model selection and fitting for the gaze model.

Evaluation is the part that is easy to get wrong, so read this once:
consecutive webcam frames are near-duplicates. A random train/test split puts
neighbours of every test frame in the training set and reports a
fantastically optimistic error. We therefore always hold out WHOLE groups:
  - group_by="person": every session of one participant at a time (the honest
    "works on someone it has never seen" number; needs >= 2 participants)
  - >= 3 sessions: leave-sessions-out ("works on a new session", but the same
    people can be in train and test)
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
    participant: np.ndarray | None = None  # (N,) who each row belongs to, for the per-person report


def pool(sessions: list[LabeledFrames], group_by: str = "session") -> Pooled:
    """group_by: "session" (default; leave-sessions-out or time blocks) or "person"."""
    if not sessions:
        raise ValueError("No datasets found. Run scripts/build_dataset.py first.")
    sessions = [s for s in sessions if len(s.X) > 0]
    if not sessions:
        raise ValueError("Every frame was filtered out (no face found, or cursor never moved).")
    X = np.vstack([s.X for s in sessions])
    y = np.vstack([s.y_frac for s in sessions])
    wh = np.vstack([np.tile(s.screen_wh, (len(s.X), 1)) for s in sessions]).astype(np.float64)
    who = np.concatenate([[s.participant or s.session_id] * len(s.X) for s in sessions])
    if group_by == "person":
        people = sorted(set(who))
        if len(people) < 2:
            raise ValueError(f"--cv person needs at least 2 participants; found {people}. "
                             "Check the 'participant' field in each session's meta.json.")
        return Pooled(X, y, wh, who, f"leave-person-out ({len(people)} people: {', '.join(people)})", who)
    if group_by != "session":
        raise ValueError(f"group_by must be 'session' or 'person', not {group_by!r}")
    if len(sessions) >= MIN_SESSIONS_FOR_SESSION_CV:
        groups = np.concatenate([[s.session_id] * len(s.X) for s in sessions])
        kind = "leave-sessions-out"
    else:
        groups = np.concatenate([[f"{s.session_id}#{int(t // (BLOCK_S * 1000))}" for t in s.t_ms] for s in sessions])
        kind = f"{BLOCK_S:.0f}s time blocks (only {len(sessions)} session(s); collect more for a stricter test)"
    return Pooled(X, y, wh, groups, kind, who)


def pixel_errors(pred_frac: np.ndarray, true_frac: np.ndarray, screen_wh: np.ndarray) -> np.ndarray:
    return np.linalg.norm((pred_frac - true_frac) * screen_wh, axis=1)


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


def error_by_yaw(errors: np.ndarray, yaw_deg: np.ndarray) -> list[tuple[str, int, float]]:
    """Mean px error per head-yaw bin: the direct measure of the head-angle problem."""
    rows = []
    for lo, hi in zip(YAW_BINS_DEG[:-1], YAW_BINS_DEG[1:]):
        m = (yaw_deg >= lo) & (yaw_deg < hi)
        if m.any():
            rows.append((f"yaw [{lo:+d},{hi:+d})", int(m.sum()), float(errors[m].mean())))
    return rows


def error_by_group(errors: np.ndarray, labels: np.ndarray) -> list[tuple[str, int, float]]:
    """Mean px error per label (e.g. per participant)."""
    return [(str(g), int((labels == g).sum()), float(errors[labels == g].mean())) for g in sorted(set(labels))]


def evaluate_candidates(data: Pooled, names: list[str] | None = None) -> dict[str, dict]:
    """-> {name: {"mean_px", "median_px", "p90_px", "by_yaw", "by_person"}} from out-of-fold predictions."""
    report = {}
    for name in names or list(CANDIDATES):
        pred = out_of_fold_predictions(CANDIDATES[name], data)
        err = pixel_errors(pred, data.y, data.screen_wh)
        report[name] = {
            "mean_px": float(err.mean()), "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90)),
            "by_yaw": error_by_yaw(err, data.X[:, YAW_INDEX]),
            "by_person": error_by_group(err, data.participant) if data.participant is not None else [],
        }
    return report


def fit_final(name: str, data: Pooled):
    model = CANDIDATES[name]()
    model.fit(data.X, data.y)
    return model
