# Builds eye/face/landmark crop datasets (.npz per session) from recorded sessions via MediaPipe.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.ml.crop_dataset import CropSession, build_crop_session
from doomscroll.ml.features import FrameFeaturizer
from doomscroll.recording.session import CROPS_DIR, SessionPaths, find_sessions

ROOT = Path(__file__).resolve().parents[1] / "data"


# Saves a grid image of sampled eye/face crops, for a quick visual sanity check.
def contact_sheet(cs: CropSession, out: Path, n: int = 24) -> None:
    if len(cs.rows) == 0:
        return
    pick = np.sort(np.random.default_rng(0).choice(len(cs.rows), size=min(n, len(cs.rows)), replace=False))
    tiles = []
    for k in pick:
        eyes = np.hstack([cs.eyes[k, 0], np.full((cs.eyes.shape[2], 4), 255, np.uint8), cs.eyes[k, 1]])
        face = cv2.resize(cs.face[k], (eyes.shape[0], eyes.shape[0]))
        tiles.append(np.hstack([eyes, np.full((eyes.shape[0], 4), 255, np.uint8), face]))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack([np.pad(t, 3, constant_values=255) for t in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)]
    sheet = cv2.resize(np.vstack(rows), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out), sheet)


_DESCRIPTION = """Run MediaPipe over recorded sessions -> eye/face crops + features, one .npz per session.

    python scripts/build_crops.py                    # all sessions in data/recordings
    python scripts/build_crops.py --stride 2         # every 2nd frame: half the time/space
    python scripts/build_crops.py --session data/recordings/anni_20261024_232035
    python scripts/build_crops.py --preview          # also save a contact sheet of crops to eyeball

The CNN version of build_dataset.py (output: datasets/crops/, at the
project root, kept separate from this file's data/ -- NOT data/crops/, an
old location that's no longer read by anything). Slow, run once; skips
sessions already built unless --force. Re-run with --force after changing
ml/eye_crops.py or ml/features.py.
"""


# Builds an eye/face crop dataset for each session, skipping already-built ones unless --force.
def main() -> None:
    ap = argparse.ArgumentParser(description=_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, default=ROOT / "recordings")
    ap.add_argument("--out", type=Path, default=CROPS_DIR)
    ap.add_argument("--session", type=Path, action="append", help="build only this session folder (repeatable)")
    ap.add_argument("--stride", type=int, default=1, help="process every n-th frame (default 1 = all)")
    ap.add_argument("--preview", action="store_true", help="save <session>_preview.png next to the .npz")
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
            cs = build_crop_session(
                sp.dir, featurizer.process_full, stride=args.stride,
                progress=lambda i, n: print(f"\r  frame {i}/{n}", end="", flush=True),
            )
            cs.save(out)
            print(f"\r  {cs.summary()}  -> {out}")
            if args.preview:
                contact_sheet(cs, out.with_name(f"{sp.dir.name}_preview.png"))
            processed = int((cs.reasons != "skipped").sum())
            if len(cs.rows) / max(processed, 1) < 0.6:
                print("  WARNING: under 60% of frames usable (face lost / blinks). Check lighting & framing.")
    finally:
        featurizer.close()


if __name__ == "__main__":
    main()
