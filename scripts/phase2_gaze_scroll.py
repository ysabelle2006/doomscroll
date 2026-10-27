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
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.debug.overlay import draw_gaze_minimap, draw_iris_points, draw_status_lines
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.fixation.attention_gate import AttentionGate
from doomscroll.fixation.mapper import to_window_relative_fraction
from doomscroll.gaze.calibration import GazeCalibrator, build_feature_vector
from doomscroll.gaze.full_calibration import DEFAULT_FULL_CALIBRATION_PATH, FullCalibrator
from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.gaze.smoothing import GazePointSmoother
from doomscroll.ml.features import extract_features
from doomscroll.ml.model import DEFAULT_CORRECTION_PATH, DEFAULT_MODEL_PATH, TrainedGazeModel
from doomscroll.ml.ood import GazeConfidenceGate
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


def make_predictor(
    use_trained_model: bool,
    model_path: Path,
    screen_wh: tuple[int, int],
    full_calibration_path: Path | None = None,
):
    """-> (fn(points_px, head_pose, frame_gray) -> dict with raw/corrected/point_px/ood_score,
            gate_threshold or None).
    One seam for all three gaze models, so the main loop doesn't care which is used. Only the
    trained-model path has an OOD detector (ood_score always None otherwise -> gate is a no-op,
    GazeConfidenceGate always decides "trust")."""
    if full_calibration_path is not None:
        full_calibrator = FullCalibrator.load(full_calibration_path)
        print(f"Using --full calibration profile {full_calibration_path}")

        def predict(points_px, pose, gray):
            feature = extract_features(points_px, pose, gray)
            frac = full_calibrator.predict(feature)
            wh = np.array(screen_wh, dtype=np.float64)
            point_px = frac * wh
            return {"raw_px": point_px, "corrected_px": point_px, "point_px": point_px, "ood_score": None}

        return predict, None

    if use_trained_model:
        model = TrainedGazeModel.load(model_path, correction_path=DEFAULT_CORRECTION_PATH)
        print(f"Using trained model {model.metadata.get('kind')!r}"
              f"{' + per-user correction' if model.correction else ''}"
              f"{' + OOD gate' if model.ood_detector is not None else ' (no OOD detector in this model file)'}")

        def predict(points_px, pose, gray):
            feature = extract_features(points_px, pose, gray)
            stages = model.predict_frac_stages(feature)
            wh = np.array(screen_wh, dtype=np.float64)
            return {
                "raw_px": stages["raw"] * wh,
                "corrected_px": stages["corrected"] * wh,
                "point_px": stages["clamped"] * wh,
                "ood_score": stages["ood_score"],
            }

        threshold = model.metadata.get("ood_threshold_default")
        return predict, threshold

    calibrator = GazeCalibrator.load()

    def predict(points_px, pose, gray):
        gaze = compute_gaze_feature(points_px, frame_gray=gray)
        point = calibrator.predict(build_feature_vector(gaze.vector, (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)))
        return {"raw_px": point, "corrected_px": point, "point_px": point, "ood_score": None}

    return predict, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="?", const=str(DEFAULT_MODEL_PATH), default=None, metavar="PATH",
                    help="use a model trained by scripts/train_model.py (default path if no value) "
                         "instead of the calibrate.py profile")
    ap.add_argument("--full-calibration", nargs="?", const=str(DEFAULT_FULL_CALIBRATION_PATH), default=None,
                    metavar="PATH",
                    help="use the RidgeCV profile fitted by `quick_calibrate.py --full` (default path if no "
                         "value) instead of --model or the calibrate.py profile; takes priority over --model.")
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help="override the OOD gate's default threshold (from the model file's metadata)")
    ap.add_argument("--gate-hold-ms", type=float, default=1500.0,
                    help="how long to hold the last trusted point before suppressing (GazeConfidenceGate)")
    ap.add_argument("--no-gate", action="store_true", help="disable the OOD confidence gate even if the model has a detector")
    ap.add_argument("--log-level", default="INFO", help="loguru level, e.g. DEBUG to see per-frame pipeline stages")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level,
                format="{time:HH:mm:ss.SSS} | {level: <7} | {message}")

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    clamp_x = (-EXTRAPOLATION_MARGIN * screen_w, (1 + EXTRAPOLATION_MARGIN) * screen_w)
    clamp_y = (-EXTRAPOLATION_MARGIN * screen_h, (1 + EXTRAPOLATION_MARGIN) * screen_h)

    try:
        predict_gaze, gate_threshold = make_predictor(
            args.model is not None,
            Path(args.model or DEFAULT_MODEL_PATH),
            (screen_w, screen_h),
            full_calibration_path=Path(args.full_calibration) if args.full_calibration is not None else None,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if args.gate_threshold is not None:
        gate_threshold = args.gate_threshold
    confidence_gate = None
    if not args.no_gate and gate_threshold is not None:
        confidence_gate = GazeConfidenceGate(threshold=gate_threshold, max_hold_ms=args.gate_hold_ms)
        logger.info(f"OOD confidence gate enabled: threshold={gate_threshold:.3f} max_hold_ms={args.gate_hold_ms:.0f}")
    elif not args.no_gate:
        logger.warning("No OOD detector in this model file -- confidence gate disabled (retrain with "
                        "train_model.py's default --ood-method, or pass --gate-threshold explicitly).")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    # The default max_innovation=120px is tuned for the calibrate.py profile's
    # much smaller error. The trained ML model's held-out error is ~900px mean
    # (up to ~1700px p90), so that gate rejects legitimate corrections after a
    # bad patch of predictions as if they were noise -- the smoother gets
    # stuck near wherever it last drifted to instead of recovering. Widen the
    # gate to comfortably above the model's own p90 error when it's in use.
    use_wide_innovation_gate = args.model is not None or args.full_calibration is not None
    smoother = GazePointSmoother(max_innovation=1000.0) if use_wide_innovation_gate else GazePointSmoother()
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
                if confidence_gate is not None:
                    confidence_gate.reset()
                dwell_start_ms = None
            elif points_px is not None and head_pose is not None and not is_gaze_usable(points_px):
                status_lines.append("blink...")
            elif points_px is not None and head_pose is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pred = predict_gaze(points_px, head_pose, frame_gray)
                raw_point = pred["point_px"]  # post-correction, pre-extrapolation-clamp
                clamped_point = np.array(
                    [np.clip(raw_point[0], *clamp_x), np.clip(raw_point[1], *clamp_y)]
                )
                if not np.array_equal(clamped_point, raw_point):
                    status_lines.append(f"!! prediction clamped from raw=({raw_point[0]:.0f},{raw_point[1]:.0f})")

                # --- fail-safe confidence gate (task C) ---
                gate_decision = "trust"
                gated_point = clamped_point
                if confidence_gate is not None and pred["ood_score"] is not None:
                    gate_decision, gated_point = confidence_gate.decide(pred["ood_score"], clamped_point, timestamp_ms)

                logger.debug(
                    "frame={} raw_px=({:.0f},{:.0f}) corrected_px=({:.0f},{:.0f}) ood_score={} "
                    "gate={} final_px={}",
                    timestamp_ms, pred["raw_px"][0], pred["raw_px"][1],
                    pred["corrected_px"][0], pred["corrected_px"][1],
                    f"{pred['ood_score']:.3f}" if pred["ood_score"] is not None else "n/a",
                    gate_decision,
                    f"({gated_point[0]:.0f},{gated_point[1]:.0f})" if gated_point is not None else "none (suppressed)",
                )

                if gate_decision == "suppress":
                    status_lines.append(f"LOW CONFIDENCE (score={pred['ood_score']:.2f}) -- suppressed, holding too long")
                    dwell_start_ms = None
                    draw_status_lines(frame, status_lines)
                    cv2.imshow("doomscroll - Phase 2", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                if gate_decision == "hold":
                    status_lines.append(f"LOW CONFIDENCE (score={pred['ood_score']:.2f}) -- holding last trusted point")

                smoothed_point = smoother.update(gated_point, timestamp_ms)

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
