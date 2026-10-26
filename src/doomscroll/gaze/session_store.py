"""Persistent store of RAW calibration samples across sessions.

Motivation and the tradeoff this manages:

More calibration data genuinely helps -- RidgeCV's group-aware alpha
selection is more reliable with more independent calibration points, and
more points mean better local coverage near wherever you actually look at
runtime.

But pooling sessions is NOT unconditionally good. Your physical setup drifts
between sittings: seating distance, lighting, laptop lid angle, camera
position. Old samples encode the *old* geometry, so pooling a session
captured under different conditions teaches the model contradictory
mappings for the same screen position, and the global linear fit averages
them into something worse than either session alone.

This store therefore:
  - keeps each session separate and timestamped, never silently merged
  - pools only the most recent MAX_POOLED_SESSIONS by default
  - keeps group ids globally unique across sessions, so calibration.py's
    group-aware CV still correctly treats each (session, point) pair as one
    correlated cluster rather than leaking near-duplicate frames across
    train/validation folds
  - makes wiping history a one-liner, because after any real setup change
    (new desk, new lighting, different chair height) starting fresh is the
    right move, not accumulating through it

Judge pooling empirically, not on faith: calibrate.py prints validation
error for both the session-only and pooled fit when run with --pool, so you
can compare them directly rather than assuming pooling helps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from doomscroll.gaze.calibration import CALIBRATION_DIR

SESSIONS_DIR = CALIBRATION_DIR / "sessions"

# Pool at most this many recent sessions. Kept deliberately small: the value
# of more data falls off quickly, while the risk of pooling across a setup
# change grows with every older session included.
MAX_POOLED_SESSIONS = 3


@dataclass
class CalibrationSession:
    timestamp: float
    features: np.ndarray  # (N, FEATURE_DIM) -- width depends on feature_backend, see calibration.py
    targets: np.ndarray  # (N, 2) screen px
    groups: np.ndarray  # (N,) calibration-point id, unique within this session
    screen_size: tuple[int, int]
    feature_backend: str = "handcrafted"  # which gaze-feature pipeline captured `features` -- see calibration.py

    @property
    def label(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


def save_session(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    screen_size: tuple[int, int],
    feature_backend: str = "handcrafted",
) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    # Second-resolution filenames alone can collide (two runs finishing in
    # the same second silently overwriting each other), so disambiguate with
    # a counter rather than trusting the clock to be unique.
    base = int(timestamp)
    path = SESSIONS_DIR / f"session_{base}.joblib"
    suffix = 1
    while path.exists():
        path = SESSIONS_DIR / f"session_{base}_{suffix}.joblib"
        suffix += 1
    joblib.dump(
        {
            "timestamp": timestamp,
            "features": features,
            "targets": targets,
            "groups": groups,
            "screen_size": screen_size,
            "feature_backend": feature_backend,
        },
        path,
    )
    return path


def list_sessions() -> list[CalibrationSession]:
    """Newest first."""
    if not SESSIONS_DIR.exists():
        return []
    sessions: list[CalibrationSession] = []
    for path in SESSIONS_DIR.glob("session_*.joblib"):
        try:
            data = joblib.load(path)
        except Exception:
            continue  # skip corrupt/partial files rather than failing the whole run
        sessions.append(
            CalibrationSession(
                timestamp=data["timestamp"],
                features=data["features"],
                targets=data["targets"],
                groups=data["groups"],
                screen_size=tuple(data["screen_size"]),
                # Sessions saved before this field existed were all captured
                # with the (only, at the time) hand-crafted feature pipeline.
                feature_backend=data.get("feature_backend", "handcrafted"),
            )
        )
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions


def pool_sessions(
    sessions: list[CalibrationSession],
    screen_size: tuple[int, int],
    max_sessions: int = MAX_POOLED_SESSIONS,
    feature_backend: str = "handcrafted",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[CalibrationSession]]:
    """Concatenates the most recent `max_sessions` sessions whose screen
    size AND feature backend match the current ones, remapping group ids to
    stay globally unique. Returns (features, targets, groups, sessions_actually_used).

    Sessions recorded at a different screen resolution are skipped
    entirely: their targets are in pixel coordinates of a different-sized
    screen, so pooling them would train the model toward positions that
    don't correspond to anything on the current display.

    Sessions captured with a different feature_backend are skipped for the
    same reason, one level down: their feature vectors have a different
    width/meaning entirely (e.g. a hand-crafted 10-dim eye-vector+head-pose
    sample vs an L2CS-Net 8-dim appearance+head-pose one -- see
    calibration.py's FEATURE_DIMS), so concatenating them wouldn't train
    toward a shared mapping, it would either crash on a shape mismatch or,
    worse, happen to share a width by coincidence and silently train on
    nonsense. Pick one backend's pass of calibrate.py and stick with it.
    """
    usable = [s for s in sessions if s.screen_size == screen_size and s.feature_backend == feature_backend][
        :max_sessions
    ]
    if not usable:
        return np.empty((0,)), np.empty((0,)), np.empty((0,)), []

    features_list, targets_list, groups_list = [], [], []
    group_offset = 0
    for session in usable:
        features_list.append(session.features)
        targets_list.append(session.targets)
        groups_list.append(session.groups + group_offset)
        group_offset += int(session.groups.max()) + 1

    return (
        np.concatenate(features_list, axis=0),
        np.concatenate(targets_list, axis=0),
        np.concatenate(groups_list, axis=0),
        usable,
    )


def clear_sessions() -> int:
    """Deletes all stored sessions. Returns how many were removed. Do this
    after any real setup change (new lighting, seat, camera position)."""
    if not SESSIONS_DIR.exists():
        return 0
    count = 0
    for path in SESSIONS_DIR.glob("session_*.joblib"):
        path.unlink()
        count += 1
    return count
