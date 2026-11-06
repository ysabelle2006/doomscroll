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

`session_id` itself (and `SessionMeta.participant`) is an anonymized id,
not the participant's real name -- see participant_hashid()/
make_session_id() below. The project's actual training data (built by
scripts/build_crops.py / build_dataset.py from a session like this one)
lives in CROPS_DIR / FEATURES_DIR, at the project root, NOT under this
module's own `data/` -- every session id is shared across both: the same
<hashid>_<date>_<recordID>.npz name appears in both CROPS_DIR and
FEATURES_DIR for a given session, so a new recording's two built files
drop straight into the same pool scripts/train_cnn.py / train_model.py
already load everything from, no manual merging needed.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from platformdirs import user_data_dir

SCHEMA_VERSION = 1
FRAME_LOG_COLUMNS = ("frame_idx", "t_ms", "cursor_x", "cursor_y")
SCREEN_LOG_COLUMNS = ("frame_idx", "t_ms")

# The project's permanent, anonymized training-data archive -- see this
# module's own docstring. Separate from <recordings>/<session_id>/ (this
# file's own concern, always under data/, holding the RAW per-session
# recording a session_id here refers to): CROPS_DIR/FEATURES_DIR hold
# only the already-extracted .npz files scripts/build_crops.py /
# build_dataset.py produce FROM a raw recording, which is all
# scripts/train_cnn.py / train_model.py ever actually load.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = PROJECT_ROOT / "datasets"
CROPS_DIR = DATASETS_DIR / "crops"
FEATURES_DIR = DATASETS_DIR / "features"

# Change this and every existing hashid changes -- i.e. don't, unless you
# deliberately mean to break the link between old session ids and the
# participants they belong to.
_PARTICIPANT_HASH_SALT = "doomscroll-gaze-v1"
_HASHID_LEN = 10

# A persistent, random, anonymous stand-in for --participant when the
# caller doesn't have (and deliberately doesn't want to collect) a real
# name -- see get_default_participant() below and app.py's recording UI,
# which never asks anyone to type a name at all.
DEVICE_ID_PATH = Path(user_data_dir("doomscroll")) / "device_participant_id.txt"


def participant_hashid(participant: str) -> str:
    """Deterministic, anonymous stand-in for a participant's name, used
    in session_id/filenames (and SessionMeta.participant) instead of the
    name itself. This is obfuscation, not real cryptographic anonymity:
    with only a handful of real participants, anyone who suspects a
    specific name can hash it with this exact salt and check for a
    match -- it keeps plain names out of filenames/paths (git history,
    casual directory listings, anything copied around), not out of
    reach of someone deliberately testing a guess against it."""
    digest = hashlib.sha256(f"{_PARTICIPANT_HASH_SALT}::{participant.strip().lower()}".encode()).hexdigest()
    return digest[:_HASHID_LEN]


def get_default_participant() -> str:
    """A random token, generated once per machine/install and reused after
    that (stored at DEVICE_ID_PATH), for callers that want a --participant
    value without ever collecting or typing a real name -- app.py's
    recording UI always uses this instead of a name field. Feeding the same
    token into participant_hashid() every time means repeat recordings from
    this install still land under the same hashid and keep incrementing the
    same recordID, exactly as if a real name had been typed consistently --
    it just never was. Not derived from, and has no relationship to,
    anyone's actual name."""
    if DEVICE_ID_PATH.exists():
        existing = DEVICE_ID_PATH.read_text().strip()
        if existing:
            return existing
    token = secrets.token_hex(8)
    DEVICE_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_ID_PATH.write_text(token)
    return token


_SESSION_ID_RE = re.compile(r"^([0-9a-f]+)_(\d{8})_(\d+)$")


def next_record_id(hashid: str, *dirs: Path) -> int:
    """The next recordID for `hashid` -- 1 if it has no sessions yet in
    ANY of `dirs`. recordID is a running count PER PARTICIPANT (hashid),
    never per day and never reset, so a session's number stays unique
    and ever-increasing across its whole history regardless of how many
    sessions happen to land on the same date. Scanning every directory a
    session for this hashid could already exist in (rather than trusting
    any single one to be complete) is what lets a brand new recording
    continue a participant's existing numbering -- e.g. picking up after
    their last baseline session -- instead of restarting it at 1 and
    colliding with what's already there."""
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


def make_session_id(participant: str, *existing_dirs: Path, when: datetime | None = None) -> str:
    """A new session_id in the current <hashid>_<date>_<recordID> scheme
    -- see participant_hashid()/next_record_id(). `existing_dirs` should
    cover every place a session for this participant could already be
    sitting (CROPS_DIR, FEATURES_DIR, and wherever this recording's own
    raw <recordings> folder lives), so the assigned recordID can't
    collide with -- or restart ahead of -- one that's already there."""
    when = when or datetime.now()
    hashid = participant_hashid(participant)
    record_id = next_record_id(hashid, *existing_dirs)
    return f"{hashid}_{when:%Y%m%d}_{record_id:03d}"


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
