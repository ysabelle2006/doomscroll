"""On-disk layout and metadata for one recorded data-collection session.

A session is a folder:

    <recordings>/<session_id>/
        webcam.mp4         face video (frame i  <->  row i of frames.csv)
        frames.csv         one row per webcam frame: timestamp + cursor position
        meta.json          who/what/where (screen size, camera, measured fps, ...)
        screen.mp4         OPTIONAL screen recording
        screen_frames.csv  OPTIONAL timestamps for screen.mp4 (same clock as frames.csv)

Everything is stamped by ONE clock (time.perf_counter, zeroed at session
start), which is what makes the recordings line up -- see recorder.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1
FRAME_LOG_COLUMNS = ("frame_idx", "t_ms", "cursor_x", "cursor_y")
SCREEN_LOG_COLUMNS = ("frame_idx", "t_ms")


@dataclass
class SessionMeta:
    session_id: str
    participant: str
    screen_w: int  # same coordinate space as cursor_x / cursor_y
    screen_h: int
    camera_index: int = 0
    frame_w: int = 0
    frame_h: int = 0
    measured_fps: float = 0.0  # n_frames / duration; webcams rarely hit their nominal fps
    n_frames: int = 0
    duration_s: float = 0.0
    started_at: str = ""  # wall-clock ISO time, for humans only
    has_screen_video: bool = False
    notes: str = ""  # free text: lighting, glasses, "head turned left", ...
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)


class SessionPaths:
    def __init__(self, session_dir: Path | str):
        self.dir = Path(session_dir)
        self.webcam_video = self.dir / "webcam.mp4"
        self.frames_csv = self.dir / "frames.csv"
        self.meta_json = self.dir / "meta.json"
        self.screen_video = self.dir / "screen.mp4"
        self.screen_csv = self.dir / "screen_frames.csv"

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)  # never overwrite a session

    def write_meta(self, meta: SessionMeta) -> None:
        self.meta_json.write_text(json.dumps(asdict(meta), indent=2))

    def read_meta(self) -> SessionMeta:
        raw = json.loads(self.meta_json.read_text())
        # Fields this version doesn't know (e.g. "activity" from a newer recorder)
        # are kept in `extra` instead of crashing.
        known = {f.name for f in fields(SessionMeta)}
        extra = dict(raw.pop("extra", None) or {})
        extra.update({k: raw.pop(k) for k in list(raw) if k not in known})
        return SessionMeta(**raw, extra=extra)


def find_sessions(recordings_dir: Path | str) -> list[SessionPaths]:
    """Every sub-folder that has a meta.json, sorted by name."""
    root = Path(recordings_dir)
    return [SessionPaths(p.parent) for p in sorted(root.glob("*/meta.json"))]
