"""Recorded session folder  ->  feature/label arrays (one .npz per session).

Slow step (runs MediaPipe over every frame), so it is separate from training:
build once, then train/tune as often as you like. Re-run it only when the
feature code or the label settings change.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doomscroll.ml.features import FEATURE_NAMES
from doomscroll.ml.labels import make_labels
from doomscroll.recording.session import SessionPaths

# frame -> (features or None, reason). Matches FrameFeaturizer.process, so
# tests can pass a fake instead of running MediaPipe.
FeaturizeFn = Callable[[np.ndarray, int], tuple[np.ndarray | None, str]]


@dataclass
class LabeledFrames:
    """Training-ready rows from one session (after lag + idle-cursor filtering)."""

    session_id: str
    screen_wh: tuple[int, int]
    X: np.ndarray  # (N, FEATURE_DIM)
    y_frac: np.ndarray  # (N, 2) cursor / screen size
    t_ms: np.ndarray  # (N,)
    participant: str = ""  # who was recorded; used for leave-person-out testing


@dataclass
class SessionDataset:
    """EVERY frame of a session: features (NaN row where the face/eyes were
    unusable, with the reason), raw cursor, timestamps. Labels are NOT baked
    in, so lag / idle-filter settings can be changed (and swept) at training
    time without re-running the slow MediaPipe pass."""

    session_id: str
    participant: str
    screen_wh: tuple[int, int]
    X: np.ndarray  # (N, FEATURE_DIM), NaN where invalid
    reasons: np.ndarray  # (N,) "ok" | "no_face" | "no_pose" | "blink"
    cursor_xy: np.ndarray  # (N, 2) px
    t_ms: np.ndarray  # (N,)

    def labeled(self, lag_ms: float = 0.0, window_ms: float = 600.0, min_travel_px: float = 40.0) -> LabeledFrames:
        y_frac, label_ok = make_labels(self.t_ms, self.cursor_xy, self.screen_wh, lag_ms, window_ms, min_travel_px)
        keep = label_ok & (self.reasons == "ok")
        return LabeledFrames(self.session_id, self.screen_wh, self.X[keep], y_frac[keep], self.t_ms[keep], self.participant)

    def summary(self) -> dict:
        return {k: int(v) for k, v in zip(*np.unique(self.reasons, return_counts=True))}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, X=self.X, reasons=self.reasons, cursor_xy=self.cursor_xy, t_ms=self.t_ms,
            feature_names=np.array(FEATURE_NAMES),
            info=json.dumps({"session_id": self.session_id, "participant": self.participant,
                             "screen_wh": list(self.screen_wh)}),
        )

    @classmethod
    def load(cls, path: Path) -> "SessionDataset":
        d = np.load(path, allow_pickle=False)
        if tuple(d["feature_names"]) != FEATURE_NAMES:
            raise ValueError(f"{path} was built with different features; rebuild it (scripts/build_dataset.py)")
        info = json.loads(str(d["info"]))
        return cls(info["session_id"], info["participant"], tuple(info["screen_wh"]),
                   d["X"], d["reasons"], d["cursor_xy"], d["t_ms"])


def read_frame_log(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (t_ms (N,), cursor_xy (N,2)); row i belongs to video frame i."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t_ms"]) for r in rows])
    xy = np.array([[float(r["cursor_x"]), float(r["cursor_y"])] for r in rows])
    return t, xy


def build_session_dataset(
    session_dir: Path | str,
    featurize: FeaturizeFn,
    progress: Callable[[int, int], None] | None = None,
) -> SessionDataset:
    paths = SessionPaths(session_dir)
    meta = paths.read_meta()
    t_ms, cursor_xy = read_frame_log(paths.frames_csv)

    cap = cv2.VideoCapture(str(paths.webcam_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {paths.webcam_video}")
    dim = len(FEATURE_NAMES)
    X = np.full((len(t_ms), dim), np.nan)
    reasons = np.full(len(t_ms), "video_ended", dtype="<U12")  # overwritten for every decodable frame
    try:
        for i in range(len(t_ms)):
            ok, frame = cap.read()
            if not ok:
                break  # video shorter than the log: remaining rows stay "video_ended"
            feats, reason = featurize(frame, int(t_ms[i]))
            reasons[i] = reason
            if feats is not None:
                X[i] = feats
            if progress and i % 100 == 0:
                progress(i, len(t_ms))
    finally:
        cap.release()
    return SessionDataset(meta.session_id, meta.participant, (meta.screen_w, meta.screen_h),
                          X, reasons, cursor_xy, t_ms)


def load_all(dataset_dir: Path | str) -> list[SessionDataset]:
    return [SessionDataset.load(p) for p in sorted(Path(dataset_dir).glob("*.npz"))]
