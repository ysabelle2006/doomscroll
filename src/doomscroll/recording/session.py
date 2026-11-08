# Session id/metadata helpers and file layout for recorded sessions.
from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from platformdirs import user_data_dir

# CSV column names and schema version for recorded session files.
SCHEMA_VERSION = 1
FRAME_LOG_COLUMNS = ("frame_idx", "t_ms", "cursor_x", "cursor_y")
SCREEN_LOG_COLUMNS = ("frame_idx", "t_ms")

# Dataset directory paths derived from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = PROJECT_ROOT / "datasets"
CROPS_DIR = DATASETS_DIR / "crops"
FEATURES_DIR = DATASETS_DIR / "features"

# Salt and length used to hash participant names into short ids.
_PARTICIPANT_HASH_SALT = "doomscroll-gaze-v1"
_HASHID_LEN = 10

# Path to this device's persisted anonymous participant id.
DEVICE_ID_PATH = Path(user_data_dir("doomscroll")) / "device_participant_id.txt"


# Hashes a participant name into a short anonymous id.
def participant_hashid(participant: str) -> str:
    digest = hashlib.sha256(f"{_PARTICIPANT_HASH_SALT}::{participant.strip().lower()}".encode()).hexdigest()
    return digest[:_HASHID_LEN]


# Returns this device's persisted participant id, generating one if needed.
def get_default_participant() -> str:
    if DEVICE_ID_PATH.exists():
        existing = DEVICE_ID_PATH.read_text().strip()
        if existing:
            return existing
    token = secrets.token_hex(8)
    DEVICE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_ID_PATH.write_text(token)
    return token


# Matches a session id's hashid/date/record-number components.
_SESSION_ID_RE = re.compile(r"^([0-9a-f]+)_(\d{8})_(\d+)$")


# Finds the next unused record number for a participant across the given dirs.
def next_record_id(hashid: str, *dirs: Path) -> int:
    highest = 0
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            stem = p.stem if p.is_file() else p.name
            m = _SESSION_ID_RE.match(stem)
            if m and m.group(1) == hashid:
                highest = max(highest, int(m.group(3)))
    return highest + 1


# Builds a new session id from participant, date and next record number.
def make_session_id(participant: str, *existing_dirs: Path, when: datetime | None = None) -> str:
    when = when or datetime.now()
    hashid = participant_hashid(participant)
    record_id = next_record_id(hashid, *existing_dirs)
    return f"{hashid}_{when:%Y%m%d}_{record_id:03d}"


# Metadata recorded for a single capture session.
@dataclass
class SessionMeta:
    session_id: str
    participant: str
    screen_w: int
    screen_h: int
    camera_index: int = 0
    frame_w: int = 0
    frame_h: int = 0
    measured_fps: float = 0.0
    n_frames: int = 0
    duration_s: float = 0.0
    started_at: str = ""
    has_screen_video: bool = False
    notes: str = ""
    schema_version: int = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)


# File paths for a session's recorded video/CSV/metadata files.
class SessionPaths:
    # Derives all session file paths from the session directory.
    def __init__(self, session_dir: Path | str):
        self.dir = Path(session_dir)
        self.webcam_video = self.dir / "webcam.mp4"
        self.frames_csv = self.dir / "frames.csv"
        self.meta_json = self.dir / "meta.json"
        self.screen_video = self.dir / "screen.mp4"
        self.screen_csv = self.dir / "screen_frames.csv"

    # Creates the session directory.
    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)

    # Writes session metadata to meta.json.
    def write_meta(self, meta: SessionMeta) -> None:
        self.meta_json.write_text(json.dumps(asdict(meta), indent=2))

    # Reads session metadata from meta.json, preserving unknown fields in extra.
    def read_meta(self) -> SessionMeta:
        raw = json.loads(self.meta_json.read_text())
        known = {f.name for f in fields(SessionMeta)}
        extra = dict(raw.pop("extra", None) or {})
        extra.update({k: raw.pop(k) for k in list(raw) if k not in known})
        return SessionMeta(**raw, extra=extra)


# Finds all recorded sessions under a recordings directory.
def find_sessions(recordings_dir: Path | str) -> list[SessionPaths]:
    root = Path(recordings_dir)
    return [SessionPaths(p.parent) for p in sorted(root.glob("*/meta.json"))]
