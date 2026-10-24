"""Run MediaPipe over recorded sessions -> one features .npz per session.

    python scripts/build_dataset.py                 # all sessions in data/recordings
    python scripts/build_dataset.py --session data/recordings/nicole_20250101_120000

Slow (every frame goes through MediaPipe); skips sessions already built unless
--force. Re-run with --force after changing doomscroll/ml/features.py.
Labels (lag, idle filter) are applied later, in train_model.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.ml.dataset import build_session_dataset
from doomscroll.ml.features import FrameFeaturizer
from doomscroll.recording.session import SessionPaths, find_sessions

ROOT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets")
    ap.add_argument("--session", type=Path, action="append", help="build only this session folder (repeatable)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    sessions = [SessionPaths(p) for p in args.session] if args.session else find_sessions(args.recordings)
    if not sessions:
        sys.exit(f"No sessions found in {args.recordings}")

    featurizer = FrameFeaturizer()
    try:
        for sp in sessions:
            out = args.out / f"{sp.dir.name}.npz"
            if out.exists() and not args.force:
                print(f"skip {sp.dir.name} (already built)")
                continue
            featurizer.reset()
            print(f"building {sp.dir.name} ...")
            ds = build_session_dataset(
                sp.dir, featurizer.process,
                progress=lambda i, n: print(f"\r  frame {i}/{n}", end="", flush=True),
            )
            ds.save(out)
            print(f"\r  {ds.summary()}  -> {out}")
            usable = ds.summary().get("ok", 0) / max(len(ds.t_ms), 1)
            if usable < 0.6:
                print(f"  WARNING: only {usable:.0%} of frames usable (face lost / blinks). Check lighting & framing.")
    finally:
        featurizer.close()


if __name__ == "__main__":
    main()
