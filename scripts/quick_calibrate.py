"""~15-second per-user calibration for the CNN gaze model, fitted ON TOP
of the pretrained CNN (scripts/train_cnn.py trains that offline from
data_raw -- never here).

The CNN already knows the general eye+head -> screen mapping from many
recorded frames. This only shows 9 dots to measure how THIS
person/camera/seat differs, and fits a tiny scale+shift correction (see
ml/model.py: AffineCorrection). Compare with scripts/calibrate.py, the
math model's own calibration, which has to learn the entire mapping from
scratch via a much longer capture (see its own docstring) -- the two
models are calibrated independently and never share a capture mechanic.

Unlike scripts/calibrate.py you may move your head naturally -- the CNN
handles it.

Run: python scripts/quick_calibrate.py        (ESC aborts)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.ml.features import FrameFeaturizer
from doomscroll.ml.model import correction_path_for, fit_user_correction, load_gaze_model

# Defined locally (not imported from doomscroll.ml.cnn) so importing this
# module doesn't force torch to load before argparse has even run.
DEFAULT_CNN_PATH = Path(__file__).resolve().parents[1] / "models" / "gaze_cnn.pt"

WINDOW_NAME = "doomscroll quick calibration"
STEPS = (0.1, 0.5, 0.9)
POINTS = [(fx, fy) for fy in STEPS for fx in STEPS]  # 3x3 grid, fractions of the screen
SETTLE_S = 0.7  # let the eyes arrive
CAPTURE_S = 0.8
MIN_FRAMES_PER_POINT = 5
RADIUS = 18


def show_dot(w: int, h: int, px: tuple[int, int]) -> bool:
    """Draw one dot; returns True if ESC was pressed."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(canvas, px, RADIUS, (0, 220, 255), -1)
    cv2.circle(canvas, px, RADIUS, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, canvas)
    return (cv2.waitKey(1) & 0xFF) == 27


def main() -> None:
    ap = argparse.ArgumentParser(description="doomscroll CNN quick calibration")
    ap.add_argument(
        "--model", type=Path, default=DEFAULT_CNN_PATH,
        help=f"Pretrained CNN to calibrate (default: {DEFAULT_CNN_PATH}; scripts/train_cnn.py produces it).",
    )
    args = ap.parse_args()

    ensure_dpi_aware()
    w, h = get_screen_size()
    model = load_gaze_model(args.model, with_correction=False)  # deliberately WITHOUT any old correction
    correction_path = correction_path_for(args.model)
    featurizer = FrameFeaturizer()

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, 0, 0)  # same macOS fullscreen workaround as calibrate.py
    cv2.resizeWindow(WINDOW_NAME, w, h)
    for _ in range(5):
        cv2.imshow(WINDOW_NAME, np.zeros((h, w, 3), dtype=np.uint8))
        cv2.waitKey(30)

    predicted, truth = [], []
    try:
        with WebcamSource(0) as cam:
            for fx, fy in POINTS:
                px = (int(fx * w), int(fy * h))
                fracs = []
                start = time.monotonic()
                while time.monotonic() - start < SETTLE_S + CAPTURE_S:
                    if show_dot(w, h, px):
                        print("Aborted; nothing saved.")
                        sys.exit(1)
                    ok, frame = cam.read()
                    if not ok or time.monotonic() - start < SETTLE_S:
                        continue
                    r = featurizer.process_full(frame, cam.frame_timestamp_ms())
                    if r.features is not None:
                        fracs.append(model.predict_frac_frame(r.points_px, r.pose, r.gray))
                if len(fracs) >= MIN_FRAMES_PER_POINT:
                    predicted.append(np.median(fracs, axis=0))  # median: robust to a stray frame
                    truth.append([fx, fy])
                else:
                    print(f"  no usable frames for dot at ({fx}, {fy}); skipped")
    finally:
        featurizer.close()
        cv2.destroyAllWindows()

    if len(predicted) < 6:
        sys.exit("Too few usable dots; nothing saved. Check lighting / that your face stays in view.")
    correction, before, after = fit_user_correction(np.array(predicted), np.array(truth), (w, h))
    print(f"mean error: {before:.0f}px without correction, {after:.0f}px with (leave-one-out)")
    if after < before:
        correction.save(correction_path)
        print(f"Saved -> {correction_path}")
    else:
        print("Correction doesn't help for you; not saving (the base model is used as-is).")


if __name__ == "__main__":
    main()
