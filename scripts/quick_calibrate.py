# Quick 9-dot calibration pass for the CNN gaze model -- fits a small
# per-user correction on top of an existing trained model.

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

DEFAULT_CNN_PATH = Path(__file__).resolve().parents[1] / "models" / "gaze_cnn.pt"

# Dot grid positions (as screen fractions) and timing for the 9-dot pass.
WINDOW_NAME = "doomscroll quick calibration"
STEPS = (0.1, 0.5, 0.9)
POINTS = [(fx, fy) for fy in STEPS for fx in STEPS]
SETTLE_S = 0.7
CAPTURE_S = 0.8
MIN_FRAMES_PER_POINT = 5
RADIUS = 18

STILL_INSTRUCTION = "Keep your head COMPLETELY still -- follow the dot with your eyes only"


# Draws one calibration dot; returns True if ESC was pressed.
def show_dot(w: int, h: int, px: tuple[int, int]) -> bool:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(canvas, px, RADIUS, (0, 220, 255), -1)
    cv2.circle(canvas, px, RADIUS, (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (iw, _), _ = cv2.getTextSize(STILL_INSTRUCTION, font, 0.6, 1)
    cv2.putText(canvas, STILL_INSTRUCTION, (w // 2 - iw // 2, h - 30), font, 0.6, (150, 150, 150), 1)
    cv2.imshow(WINDOW_NAME, canvas)
    return (cv2.waitKey(1) & 0xFF) == 27


# Runs the 9-dot pass, fits a correction from predicted-vs-true gaze, and saves it.
def main() -> None:
    ap = argparse.ArgumentParser(description="doomscroll CNN quick calibration")
    ap.add_argument(
        "--model", type=Path, default=DEFAULT_CNN_PATH,
        help=f"Pretrained CNN to calibrate (default: {DEFAULT_CNN_PATH}; scripts/train_cnn.py produces it).",
    )
    args = ap.parse_args()

    ensure_dpi_aware()
    w, h = get_screen_size()
    model = load_gaze_model(args.model, with_correction=False)
    correction_path = correction_path_for(args.model)
    featurizer = FrameFeaturizer()

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
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
                        print("CALIBRATION_CANCELLED")
                        sys.exit(1)
                    ok, frame = cam.read()
                    if not ok or time.monotonic() - start < SETTLE_S:
                        continue
                    r = featurizer.process_full(frame, cam.frame_timestamp_ms())
                    if r.features is not None:
                        fracs.append(model.predict_frac_frame(r.points_px, r.pose, r.gray))
                if len(fracs) >= MIN_FRAMES_PER_POINT:
                    predicted.append(np.median(fracs, axis=0))
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
        used_err, used_label = after, "with correction"
    else:
        print("Correction doesn't help for you; not saving (the base model is used as-is).")
        used_err, used_label = before, "without correction (base model)"

    screen_diag = (w**2 + h**2) ** 0.5
    print(f"CALIBRATION_SUMMARY: err_px={used_err:.1f} err_pct={100 * used_err / screen_diag:.1f} label={used_label}")


if __name__ == "__main__":
    main()
