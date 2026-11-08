# Builds and stores per-session eye/face crop datasets (CNN pipeline) from recordings.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from doomscroll.ml.dataset import read_frame_log
from doomscroll.ml.eye_crops import CROP_SPEC, EYE_H, EYE_W, FACE_SIZE, extract_crops
from doomscroll.ml.features import FEATURE_DIM, FEATURE_NAMES, FrameResult
from doomscroll.ml.labels import make_labels
from doomscroll.recording.session import SessionPaths

ProcessFn = Callable[[np.ndarray, int], FrameResult]


# One session's extracted crops, features, and labels, one row per usable frame.
@dataclass
class CropSession:
    session_id: str
    participant: str
    screen_wh: tuple[int, int]
    t_ms: np.ndarray
    cursor_xy: np.ndarray
    reasons: np.ndarray
    rows: np.ndarray
    X: np.ndarray
    eyes: np.ndarray
    face: np.ndarray

    # Returns the (local index, target fraction) of rows that pass the activity/on-screen filter.
    def labeled_index(self, lag_ms=0.0, window_ms=600.0, min_travel_px=40.0) -> tuple[np.ndarray, np.ndarray]:
        y_frac, ok = make_labels(self.t_ms, self.cursor_xy, self.screen_wh, lag_ms, window_ms, min_travel_px)
        k = np.flatnonzero(ok[self.rows])
        return k, y_frac[self.rows[k]]

    # Counts frames by extraction outcome (ok / no_face / blink / etc).
    def summary(self) -> dict:
        d = {k: int(v) for k, v in zip(*np.unique(self.reasons, return_counts=True))}
        d["stored_crops"] = int(len(self.rows))
        return d

    # Saves this dataset to a compressed .npz file.
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t_ms=self.t_ms, cursor_xy=self.cursor_xy, reasons=self.reasons, rows=self.rows,
            X=self.X, eyes=self.eyes, face=self.face, feature_names=np.array(FEATURE_NAMES),
            info=json.dumps({"session_id": self.session_id, "participant": self.participant,
                             "screen_wh": list(self.screen_wh), "crop_spec": CROP_SPEC}),
        )

    # Loads a dataset previously saved with save().
    @classmethod
    def load(cls, path: Path) -> "CropSession":
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"]))
        if tuple(d["feature_names"]) != FEATURE_NAMES or info.get("crop_spec") != CROP_SPEC:
            raise ValueError(f"{path} was built with different features/crops; rebuild it (scripts/build_crops.py --force)")
        return cls(info["session_id"], info["participant"], tuple(info["screen_wh"]),
                   d["t_ms"], d["cursor_xy"], d["reasons"], d["rows"], d["X"], d["eyes"], d["face"])


# Runs process() over every frame of a recorded session's video, extracting crops for usable ones.
def build_crop_session(
    session_dir: Path | str,
    process: ProcessFn,
    stride: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> CropSession:
    paths = SessionPaths(session_dir)
    meta = paths.read_meta()
    t_ms, cursor_xy = read_frame_log(paths.frames_csv)
    n = len(t_ms)

    cap = cv2.VideoCapture(str(paths.webcam_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {paths.webcam_video}")
    reasons = np.full(n, "video_ended", dtype="<U12")
    rows, feats, eyes, faces = [], [], [], []
    try:
        for i in range(n):
            if i % stride:
                if not cap.grab():
                    break
                reasons[i] = "skipped"
                continue
            ok, frame = cap.read()
            if not ok:
                break
            r = process(frame, int(t_ms[i]))
            reasons[i] = r.reason
            if r.features is not None:
                e, f = extract_crops(r.gray, r.points_px)
                rows.append(i)
                feats.append(r.features)
                eyes.append(e)
                faces.append(f)
            if progress and i % 100 == 0:
                progress(i, n)
    finally:
        cap.release()

    k = len(rows)
    return CropSession(
        meta.session_id, meta.participant, (meta.screen_w, meta.screen_h), t_ms, cursor_xy, reasons,
        np.array(rows, dtype=np.int64),
        np.array(feats, dtype=np.float32).reshape(k, FEATURE_DIM),
        np.array(eyes, dtype=np.uint8).reshape(k, 2, EYE_H, EYE_W),
        np.array(faces, dtype=np.uint8).reshape(k, FACE_SIZE, FACE_SIZE),
    )


# Loads every crop dataset (.npz) in a directory.
def load_all_crops(crop_dir: Path | str) -> list[CropSession]:
    return [CropSession.load(p) for p in sorted(Path(crop_dir).glob("*.npz"))]
