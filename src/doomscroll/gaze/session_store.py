# Stores and pools past calibration sessions (features/targets/groups) on disk.

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from doomscroll.gaze.calibration import CALIBRATION_DIR

SESSIONS_DIR = CALIBRATION_DIR / "sessions"

MAX_POOLED_SESSIONS = 3


# One saved calibration session's captured samples.
@dataclass
class CalibrationSession:
    timestamp: float
    features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    screen_size: tuple[int, int]
    protocol: str = "trajectory"

    @property
    def label(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


# Saves a calibration session's samples to a new .joblib file.
def save_session(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    screen_size: tuple[int, int],
    protocol: str = "trajectory",
) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
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
            "protocol": protocol,
        },
        path,
    )
    return path


# Loads every stored calibration session, newest first.
def list_sessions() -> list[CalibrationSession]:
    if not SESSIONS_DIR.exists():
        return []
    sessions: list[CalibrationSession] = []
    for path in SESSIONS_DIR.glob("session_*.joblib"):
        try:
            data = joblib.load(path)
        except Exception:
            continue
        sessions.append(
            CalibrationSession(
                timestamp=data["timestamp"],
                features=data["features"],
                targets=data["targets"],
                groups=data["groups"],
                screen_size=tuple(data["screen_size"]),
                protocol=data.get("protocol", "dots"),
            )
        )
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions


# Concatenates samples from the most recent matching sessions, for --pool fitting.
def pool_sessions(
    sessions: list[CalibrationSession],
    screen_size: tuple[int, int],
    max_sessions: int = MAX_POOLED_SESSIONS,
    protocol: str = "trajectory",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[CalibrationSession]]:
    usable = [
        s
        for s in sessions
        if s.screen_size == screen_size and s.protocol == protocol
    ][:max_sessions]
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


# Deletes every stored calibration session; returns how many were removed.
def clear_sessions() -> int:
    if not SESSIONS_DIR.exists():
        return 0
    count = 0
    for path in SESSIONS_DIR.glob("session_*.joblib"):
        path.unlink()
        count += 1
    return count
