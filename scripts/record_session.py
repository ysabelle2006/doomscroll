"""Record one data-collection session: webcam video + cursor position and
mouse-button state at every frame (+ optional screen recording), one clock.

    python scripts/record_session.py --participant nicole --activity game:osu --duration 600
    python scripts/record_session.py --participant nicole --screen --notes "head turned left"

CONTROLS (global, work while the game has focus):  F8 = stop,  F9 = pause/resume.
Pause during menus, loading screens, cutscenes, chat -- anything where you are
not looking at the cursor. Paused frames are excluded from training.

PICK GAMES WHERE THE CURSOR IS A REAL POINTER (this decides data quality):
  GOOD: point-and-click / mouse-driven games, strategy (RTS, city builders),
        card games, rhythm/aim trainers (osu!), puzzle, browser games, map-based
        games -- you look where you click and the cursor roams the whole screen.
  BAD:  first/third-person games with mouse-look. The game locks the cursor to
        the centre, so the cursor position says nothing about where you look.
        Also bad: gamepad games (no cursor), fast flick-aiming shooters.
  Run the game WINDOWED or BORDERLESS (not exclusive fullscreen), on the same
  monitor the camera sits at, so cursor coordinates match the screen size.

ALSO FOR GOOD DATA:
  * Vary your head: separate sessions with head forward, turned, tilted,
    closer/further, noted in --notes. Vary lighting and glasses too.
  * Don't use the keyboard-only parts of a game for long stretches.
Ctrl+C or F8 stops; the partial session is kept.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.ml.labels import cursor_quality_warnings
from doomscroll.ml.dataset import read_frame_log
from doomscroll.recording.recorder import HotkeyControls, PynputCursor, Recorder
from doomscroll.recording.session import SessionMeta, SessionPaths

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "recordings"
PREVIEW_WIDTH = 320


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="short name, e.g. 'nicole' (used in folder name)")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds to record (default 300)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--screen", action="store_true", help="also record the screen (optional, not needed for training)")
    ap.add_argument("--preview", action="store_true", help="show a small webcam preview (press q in it to stop)")
    ap.add_argument("--notes", default="", help="free text: lighting, glasses, head pose for this session")
    ap.add_argument("--activity", default="free", help="what you'll be doing, e.g. 'game:osu' (saved in meta.json)")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    ensure_dpi_aware()  # must precede get_screen_size so cursor and screen size share one coordinate space
    screen_w, screen_h = get_screen_size()
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "", args.participant)
    session_id = f"{safe_name}_{datetime.now():%Y%m%d_%H%M%S}"
    meta = SessionMeta(session_id, args.participant, screen_w, screen_h, camera_index=args.camera, notes=args.notes,
                       activity=args.activity)
    paths = SessionPaths(args.out / session_id)

    last_print = [0.0]

    def on_frame(idx: int, t_ms: float, frame) -> bool:
        if t_ms - last_print[0] >= 1000:
            last_print[0] = t_ms
            state = "PAUSED" if controls.paused else "rec   "
            print(f"\r  {state} {t_ms / 1000:6.1f}s / {args.duration:.0f}s   frames={idx + 1}   fps={(idx + 1) / (t_ms / 1000):4.1f}", end="", flush=True)
        if args.preview:
            small = cv2.resize(frame, (PREVIEW_WIDTH, int(frame.shape[0] * PREVIEW_WIDTH / frame.shape[1])))
            cv2.imshow("recording (q to stop)", small)
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        return False

    controls = HotkeyControls()
    cursor = PynputCursor()
    print(f"Screen {screen_w}x{screen_h}. Session: {session_id}")
    with WebcamSource(args.camera) as cam:
        for n in range(args.countdown, 0, -1):
            print(f"Starting in {n}...  (F8 stop, F9 pause)")
            time.sleep(1)
        recorder = Recorder(cam, cursor, paths, meta, record_screen=args.screen, on_frame=on_frame,
                            is_paused=lambda: controls.paused)
        try:
            meta = recorder.run(args.duration, should_stop=lambda: controls.stop_requested)
        except KeyboardInterrupt:
            meta = paths.read_meta()  # Recorder's finally-block already saved it
        finally:
            controls.close()
            cursor.close()
    cv2.destroyAllWindows()

    print(f"\nSaved {meta.n_frames} frames, {meta.duration_s:.0f}s, {meta.measured_fps:.1f} fps -> {paths.dir}")
    if meta.measured_fps and meta.measured_fps < 15:
        print("WARNING: low fps. Close heavy apps / lower resolution; <15 fps makes eye tracking poor.")
    t_ms, xy, _down, _active = read_frame_log(paths.frames_csv)
    for w in cursor_quality_warnings(t_ms, xy, (screen_w, screen_h)):
        print(f"WARNING: {w}")
    if meta.extra.get("screen_error"):
        print(f"WARNING: screen recording failed: {meta.extra['screen_error']}")


if __name__ == "__main__":
    main()
