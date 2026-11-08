# Records one webcam+cursor (and optionally screen) data-collection session to disk.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.recording.recorder import PynputCursor, Recorder
from doomscroll.recording.session import (
    CROPS_DIR, FEATURES_DIR, SessionMeta, SessionPaths, get_default_participant, make_session_id,
    participant_hashid,
)

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "recordings"
PREVIEW_WIDTH = 320


_DESCRIPTION = """Record one data-collection session: webcam video + cursor position at
every frame (+ optional screen recording), all on one clock.

    python scripts/record_session.py --participant nicole --duration 300
    python scripts/record_session.py --participant nicole --screen --notes "head turned left"

HOW TO COLLECT GOOD DATA (this matters more than any model choice):
  * Look AT the cursor the whole time and keep moving it -- sweep corners,
    edges, zig-zags, slow and fast. Frames where the cursor is idle are discarded.
  * Vary your head: do separate sessions (or segments) with head forward,
    turned left/right, tilted up/down, leaning closer/further, and say so in --notes.
  * Vary conditions across sessions: lighting, glasses on/off, time of day.
  * Don't touch the keyboard/mouse in ways that pull your eyes from the cursor.
Ctrl+C stops early; the partial session is kept.
"""


# Parses args, then records webcam frames + cursor position until --duration elapses or Ctrl+C.
def main() -> None:
    ap = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--participant", default=None,
        help="short name, e.g. 'nicole' -- NOT used in the folder/file name or saved anywhere on disk; "
        "only its hash is (see doomscroll.recording.session.participant_hashid()). Pass the same name "
        "again for this same person's later sessions so they keep landing under the same hashid. "
        "Optional: if omitted (e.g. when called from app.py, which never collects a name), a random "
        "id generated once for this machine and reused every time (get_default_participant()) is "
        "hashed instead, so repeat recordings here still land under one consistent hashid.",
    )
    ap.add_argument("--duration", type=float, default=300.0, help="seconds to record (default 300)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--screen", action="store_true", help="also record the screen (optional, not needed for training)")
    ap.add_argument("--preview", action="store_true", help="show a small webcam preview (press q in it to stop)")
    ap.add_argument("--notes", default="", help="free text: lighting, glasses, head pose for this session")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    participant = args.participant or get_default_participant()
    session_id = make_session_id(participant, CROPS_DIR, FEATURES_DIR, args.out)
    hashid = participant_hashid(participant)
    meta = SessionMeta(session_id, hashid, screen_w, screen_h, camera_index=args.camera, notes=args.notes)
    paths = SessionPaths(args.out / session_id)

    last_print = [0.0]

    # Per-frame callback: prints progress, and optionally shows a preview window.
    def on_frame(idx: int, t_ms: float, frame) -> bool:
        if t_ms - last_print[0] >= 1000:
            last_print[0] = t_ms
            print(f"\r  {t_ms / 1000:6.1f}s / {args.duration:.0f}s   frames={idx + 1}   fps={(idx + 1) / (t_ms / 1000):4.1f}", end="", flush=True)
        if args.preview:
            small = cv2.resize(frame, (PREVIEW_WIDTH, int(frame.shape[0] * PREVIEW_WIDTH / frame.shape[1])))
            cv2.imshow("recording (q to stop)", small)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        return False

    print(f"Screen {screen_w}x{screen_h}. Session: {session_id}")
    with WebcamSource(args.camera) as cam:
        for n in range(args.countdown, 0, -1):
            print(f"Starting in {n}...  (look at the cursor and keep moving it)")
            time.sleep(1)
        recorder = Recorder(cam, PynputCursor(), paths, meta, record_screen=args.screen, on_frame=on_frame)
        try:
            meta = recorder.run(args.duration)
        except KeyboardInterrupt:
            meta = paths.read_meta()
    cv2.destroyAllWindows()

    print(f"\nSaved {meta.n_frames} frames, {meta.duration_s:.0f}s, {meta.measured_fps:.1f} fps -> {paths.dir}")
    if meta.measured_fps and meta.measured_fps < 15:
        print("WARNING: low fps. Close heavy apps / lower resolution; <15 fps makes eye tracking poor.")
    if meta.extra.get("screen_error"):
        print(f"WARNING: screen recording failed: {meta.extra['screen_error']}")


if __name__ == "__main__":
    main()
