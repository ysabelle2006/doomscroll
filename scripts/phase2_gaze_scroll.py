"""Phase 2: end-to-end naive trigger -- calibrated gaze mapping + attention
gate + a trivial rule ("dwell in the bottom 15% of the focused window for
1s -> scroll down"). No content-awareness or pattern-based sensitivity yet
(that's Phase 3/4); this validates calibration quality and loop timing.

Run scripts/calibrate.py first. Then:
  .venv\\Scripts\\python.exe scripts\\phase2_gaze_scroll.py
Or, to use the model trained on recorded data (scripts/train_model.py) instead
of the per-user calibration:
  python scripts/phase2_gaze_scroll.py --model            # models/gaze_model.joblib
  python scripts/quick_calibrate.py                       # optional ~15 s per-user nudge
Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.debug.overlay import draw_gaze_minimap, draw_iris_points, draw_status_lines
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.fixation.attention_gate import AttentionGate
from doomscroll.fixation.mapper import to_window_relative_fraction
from doomscroll.gaze.calibration import GazeCalibrator, build_feature_vector
from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.gaze.smoothing import GazePointSmoother
from doomscroll.ml.features import extract_features
from doomscroll.ml.model import DEFAULT_CORRECTION_PATH, DEFAULT_MODEL_PATH, TrainedGazeModel
from doomscroll.landmarks.mediapipe_backend import LEFT_IRIS, RIGHT_IRIS, MediaPipeLandmarkBackend
from doomscroll.scroll import ScrollUnit, get_scroll_injector
from doomscroll.windowing import get_window_bounds_provider

BOTTOM_DWELL_FRACTION = 0.85
DWELL_MS = 1000
SCROLL_COOLDOWN_MS = 800
SCROLL_STEP_LINES = 3
DEBUG_WINDOW_TITLE = "doomscroll - Phase 2"

# A degree-2 polynomial calibration model has no bound on its output for
# inputs that drift outside the range seen during calibration. Unlike a
# single-frame spike, a *persistently* out-of-range prediction isn't caught
# by the Kalman filter's innovation gating (the filter's own state drifts to
# agree with a sustained trend, so gating stops treating it as an outlier).
# Clamp the raw prediction to a generous margin around the physical screen
# before it ever reaches smoothing, as a robustness backstop independent of
# whatever's causing the extrapolation.
EXTRAPOLATION_MARGIN = 0.5  # allow up to 50% of screen size beyond each edge


def make_predictor(use_trained_model: bool, model_path: Path, screen_wh: tuple[int, int]):
    """-> fn(points_px, head_pose, frame_gray) -> raw screen point (px).
    One seam for both gaze models, so the main loop doesn't care which is used."""
    if use_trained_model:
        model = TrainedGazeModel.load(model_path, correction_path=DEFAULT_CORRECTION_PATH)
        print(f"Using trained model {model.metadata.get('kind')!r}"
              f"{' + per-user correction' if model.correction else ''}")
        return lambda points_px, pose, gray: model.predict_px(extract_features(points_px, pose, gray), screen_wh)

    calibrator = GazeCalibrator.load()

    def predict(points_px, pose, gray):
        gaze = compute_gaze_feature(points_px, frame_gray=gray)
        return calibrator.predict(build_feature_vector(gaze.vector, (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)))

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="?", const=str(DEFAULT_MODEL_PATH), default=None, metavar="PATH",
                    help="use a model trained by scripts/train_model.py (default path if no value) "
                         "instead of the calibrate.py profile")
    args = ap.parse_args()

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    clamp_x = (-EXTRAPOLATION_MARGIN * screen_w, (1 + EXTRAPOLATION_MARGIN) * screen_w)
    clamp_y = (-EXTRAPOLATION_MARGIN * screen_h, (1 + EXTRAPOLATION_MARGIN) * screen_h)

    try:
        predict_gaze = make_predictor(args.model is not None, Path(args.model or DEFAULT_MODEL_PATH), (screen_w, screen_h))
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    smoother = GazePointSmoother()
    attention_gate = AttentionGate()
    window_provider = get_window_bounds_provider()
    injector = get_scroll_injector()

    dwell_start_ms: int | None = None
    cooldown_until_ms = 0
    last_external_bounds = None

    with WebcamSource(0) as cam:
        print("Running. Press 'q' to quit.")
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Failed to read frame from webcam.")
                break

            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)

            head_pose = None
            points_px = None
            if result is not None:
                points_px = result.points_px()
                draw_iris_points(frame, points_px, LEFT_IRIS + RIGHT_IRIS)
                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
            else:
                pose_estimator.reset()

            paused = attention_gate.update(head_pose, timestamp_ms)
            status_lines: list[str] = []
            if head_pose is not None:
                status_lines.append(
                    f"yaw={head_pose.yaw_deg:.1f} pitch={head_pose.pitch_deg:.1f} roll={head_pose.roll_deg:.1f}"
                )
            else:
                status_lines.append("no head pose (face not detected)")

            if paused:
                status_lines.append("PAUSED (looking away)")
                smoother.reset()
                dwell_start_ms = None
            elif points_px is not None and head_pose is not None and not is_gaze_usable(points_px):
                status_lines.append("blink...")
            elif points_px is not None and head_pose is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                raw_point = predict_gaze(points_px, head_pose, frame_gray)
                clamped_point = np.array(
                    [np.clip(raw_point[0], *clamp_x), np.clip(raw_point[1], *clamp_y)]
                )
                if not np.array_equal(clamped_point, raw_point):
                    status_lines.append(f"!! prediction clamped from raw=({raw_point[0]:.0f},{raw_point[1]:.0f})")
                smoothed_point = smoother.update(clamped_point, timestamp_ms)

                current_bounds = window_provider.get_focused_window_bounds()
                # Our own debug preview window can itself become the OS
                # foreground window (e.g. after a click), which would
                # otherwise hijack the "focused window" the scroll logic is
                # supposed to be reasoning about. Ignore self-focus and keep
                # using the last externally-focused window's bounds instead.
                if current_bounds is not None and current_bounds.app_name != DEBUG_WINDOW_TITLE:
                    last_external_bounds = current_bounds
                bounds = last_external_bounds

                status_lines.append(f"raw=({raw_point[0]:.0f},{raw_point[1]:.0f}) smoothed=({smoothed_point[0]:.0f},{smoothed_point[1]:.0f})")
                if bounds is None or bounds.width <= 0 or bounds.height <= 0:
                    status_lines.append("no usable focused window bounds")
                    dwell_start_ms = None
                else:
                    status_lines.append(
                        f"bounds=({bounds.x},{bounds.y},{bounds.width}x{bounds.height}) app={bounds.app_name!r}"
                    )
                    fx, fy = to_window_relative_fraction(smoothed_point, bounds)
                    status_lines.append(f"gaze frac=({fx:.2f},{fy:.2f})")
                    draw_gaze_minimap(frame, fx, fy, dwelling=(fy >= BOTTOM_DWELL_FRACTION))

                    if timestamp_ms < cooldown_until_ms:
                        status_lines.append("cooldown...")
                        dwell_start_ms = None
                    elif fy >= BOTTOM_DWELL_FRACTION:
                        if dwell_start_ms is None:
                            dwell_start_ms = timestamp_ms
                        elapsed = timestamp_ms - dwell_start_ms
                        status_lines.append(f"DWELLING {elapsed}ms / {DWELL_MS}ms")
                        if elapsed >= DWELL_MS:
                            target_point = (
                                bounds.x + bounds.width // 2,
                                bounds.y + bounds.height // 2,
                            )
                            injector.scroll(0, SCROLL_STEP_LINES, ScrollUnit.LINE, target_point=target_point)
                            print(f"[{timestamp_ms}] SCROLL triggered")
                            dwell_start_ms = None
                            cooldown_until_ms = timestamp_ms + SCROLL_COOLDOWN_MS
                    else:
                        dwell_start_ms = None
            else:
                status_lines.append("no face detected")
                dwell_start_ms = None

            draw_status_lines(frame, status_lines)
            cv2.imshow("doomscroll - Phase 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    backend.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
