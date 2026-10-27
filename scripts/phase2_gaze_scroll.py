"""Phase 2: gaze mapping + attention gate + content-aware scroll trigger
(fixation/scroll_trigger.py, from feat-002):

  1. Gaze in the bottom part of the focused window (BOTTOM_DWELL_FRACTION)
     for DWELL_MS -> scroll.
  2. Else, if OCR shows gaze has read the last visible line of text and moved
     past it, for DWELL_MS -> scroll.
  3. Else, if no unread text is left between gaze and the bottom of the
     window, for CONTENT_DWELL_MS -> scroll.

OCR (content/ package, Apple Vision) re-scans the focused window every ~0.4 s
on a background thread. A transparent click-through overlay (debug/screen_overlay.py)
is drawn over the window you're reading: green box = line already read, red = not
read yet, blue = last line on screen, yellow line = "end of content" threshold,
magenta dot = where you're looking. If OCR or the overlay aren't available
(pyobjc-framework-Vision not installed, not macOS) it falls back to rule 1 only.

Run scripts/calibrate.py first. Then:
  .venv\\Scripts\\python.exe scripts\\phase2_gaze_scroll.py
Or, to use the model trained on recorded data (scripts/train_model.py) instead
of the per-user calibration:
  python scripts/phase2_gaze_scroll.py --model            # models/gaze_model.joblib
  python scripts/phase2_gaze_scroll.py --model models/gaze_cnn.pt   # the CNN (train_cnn.py)
  python scripts/quick_calibrate.py                       # optional ~15 s per-user nudge
  python scripts/phase2_gaze_scroll.py --model models/gaze_cnn_v2.pt --content-only  # test rules 2+3 alone
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
from doomscroll.content import get_content_scanner
from doomscroll.content.gaze_content import BackgroundContentScanner, ContentGazeTracker
from doomscroll.debug.overlay import draw_gaze_minimap, draw_iris_points, draw_status_lines
from doomscroll.debug.screen_overlay import ScreenOverlay
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.fixation.attention_gate import AttentionGate
from doomscroll.fixation.mapper import to_window_relative_fraction
from doomscroll.fixation.scroll_trigger import ScrollTrigger, ScrollTriggerConfig
from doomscroll.gaze.calibration import GazeCalibrator, build_feature_vector
from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.gaze.smoothing import GazePointSmoother
from doomscroll.ml.model import DEFAULT_MODEL_PATH, load_gaze_model
from doomscroll.landmarks.mediapipe_backend import LEFT_IRIS, RIGHT_IRIS, MediaPipeLandmarkBackend
from doomscroll.scroll import ScrollUnit, get_scroll_injector
from doomscroll.windowing import get_window_bounds_provider

# Scroll-trigger settings (passed into ScrollTriggerConfig below)
BOTTOM_DWELL_FRACTION = 0.80   # rule 1: bottom 20% of the focused window
DWELL_MS = 700                 # rules 1+2: how long before scrolling (feat-002's default is 150)
CONTENT_DWELL_MS = 1000        # rule 3: "nothing unread below gaze" -- broadest rule, so slower
SCROLL_COOLDOWN_MS = 800
SCROLL_STEP_LINES = 6
DEBUG_WINDOW_TITLE = "doomscroll - Phase 2"  # just a label; windowing/macos_backend.py ignores our own windows

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
        model = load_gaze_model(model_path)  # .joblib (landmark model) or .pt (CNN)
        print(f"Using trained model {model.metadata.get('kind')!r}"
              f"{' + per-user correction' if model.correction else ''}")
        wh = np.array(screen_wh, dtype=np.float64)
        return lambda points_px, pose, gray: model.predict_frac_frame(points_px, pose, gray) * wh

    calibrator = GazeCalibrator.load()

    def predict(points_px, pose, gray):
        gaze = compute_gaze_feature(points_px, frame_gray=gray)
        return calibrator.predict(build_feature_vector(gaze.vector, (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)))

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="?", const=str(DEFAULT_MODEL_PATH), default=None, metavar="PATH",
                    help="use a model trained by scripts/train_model.py or train_cnn.py (.pt) (default path if no value) "
                         "instead of the calibrate.py profile")
    ap.add_argument("--content-only", action="store_true",
                    help="test flag: turn off rule 1 (bottom zone) so only the OCR rules can scroll")
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

    last_external_bounds = None

    # --- content-aware parts (OCR + overlay + 3-rule trigger) ---------------
    # The background OCR thread asks for the bounds repeatedly, so give it a
    # function (reads the latest value) rather than the value itself.
    def _current_bounds():
        return last_external_bounds

    try:
        content_scanner = get_content_scanner()
        content_available = True
    except Exception as e:
        print(f"Content-aware scrolling unavailable ({e}) -- bottom-of-window rule only.")
        content_scanner, content_available = None, False
    bg_scanner = BackgroundContentScanner(content_scanner, get_bounds=_current_bounds) if content_scanner else None
    content_tracker = ContentGazeTracker()

    try:
        screen_overlay = ScreenOverlay()
    except Exception as e:
        print(f"Live content overlay unavailable ({e}) -- continuing without it.")
        screen_overlay = None
    overlay_broken = False  # set once if the overlay fails at runtime, so it errors once, not every frame

    trigger = ScrollTrigger(content_tracker, ScrollTriggerConfig(
        # inf = rule 1 can never fire (window fractions are clamped to 0..1)
        bottom_definite_fraction=float("inf") if args.content_only else BOTTOM_DWELL_FRACTION,
        fast_dwell_ms=DWELL_MS,
        content_tier_dwell_ms=CONTENT_DWELL_MS,
        cooldown_ms=SCROLL_COOLDOWN_MS,
    ))
    if args.content_only:
        print("--content-only: bottom-zone rule off; only OCR rules can scroll.")
    if bg_scanner is not None:
        bg_scanner.start()

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
                trigger.reset_dwell()
            elif points_px is not None and head_pose is not None and not is_gaze_usable(points_px):
                status_lines.append("blink...")
                trigger.reset_dwell()
            elif points_px is not None and head_pose is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                raw_point = predict_gaze(points_px, head_pose, frame_gray)
                clamped_point = np.array(
                    [np.clip(raw_point[0], *clamp_x), np.clip(raw_point[1], *clamp_y)]
                )
                if not np.array_equal(clamped_point, raw_point):
                    status_lines.append(f"!! prediction clamped from raw=({raw_point[0]:.0f},{raw_point[1]:.0f})")
                smoothed_point = smoother.update(clamped_point, timestamp_ms)

                # Track which OCR'd lines the gaze has dwelled on ("seen"), every tick.
                blocks = bg_scanner.latest_blocks() if bg_scanner is not None else []
                content_tracker.update(blocks, smoothed_point, timestamp_ms)

                # get_focused_window_bounds() returns None while THIS process
                # (e.g. the preview window) is frontmost -- see
                # windowing/macos_backend.py -- so keep the last external window.
                current_bounds = window_provider.get_focused_window_bounds()
                if current_bounds is not None:
                    last_external_bounds = current_bounds
                bounds = last_external_bounds

                if screen_overlay is not None and not overlay_broken and bounds is not None \
                        and bounds.width > 0 and bounds.height > 0:
                    try:
                        screen_overlay.update(blocks, bounds, smoothed_point,
                                              content_tracker.seen_ids, content_tracker.end_margin_px)
                    except Exception as e:
                        print(f"Live content overlay broke ({e}) -- disabling it for the rest of this run.")
                        overlay_broken = True

                status_lines.append(f"raw=({raw_point[0]:.0f},{raw_point[1]:.0f}) smoothed=({smoothed_point[0]:.0f},{smoothed_point[1]:.0f})")
                status_lines.append(f"content: {'on' if content_available else 'off'} "
                                    f"lines={len(blocks)} seen={len(content_tracker.seen_ids)}")
                if bounds is None or bounds.width <= 0 or bounds.height <= 0:
                    status_lines.append("no usable focused window bounds")
                    trigger.reset_dwell()
                else:
                    status_lines.append(
                        f"bounds=({bounds.x},{bounds.y},{bounds.width}x{bounds.height}) app={bounds.app_name!r}"
                    )
                    fx, fy = to_window_relative_fraction(smoothed_point, bounds)
                    status_lines.append(f"gaze frac=({fx:.2f},{fy:.2f})")

                    # OCR rules only when OCR has actually found text: with zero lines,
                    # "nothing unread below gaze" is trivially true and would scroll
                    # nonstop (startup, blank pages, videos, OCR failures).
                    should_fire, label = trigger.update(
                        fy, blocks, smoothed_point, bounds.y + bounds.height, timestamp_ms,
                        content_available and len(blocks) > 0,
                    )
                    status_lines.append(label)
                    draw_gaze_minimap(frame, fx, fy, dwelling=not label.startswith(("not scrolling", "cooldown")))

                    if should_fire:
                        target_point = (
                            bounds.x + bounds.width // 2,
                            bounds.y + bounds.height // 2,
                        )
                        injector.scroll(0, SCROLL_STEP_LINES, ScrollUnit.LINE, target_point=target_point)
                        trigger.note_scroll_fired(timestamp_ms)
                        print(f"[{timestamp_ms}] SCROLL triggered ({label})")
            else:
                status_lines.append("no face detected")
                trigger.reset_dwell()

            draw_status_lines(frame, status_lines)
            cv2.imshow(DEBUG_WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if bg_scanner is not None:
        bg_scanner.stop()
    if screen_overlay is not None:
        try:
            screen_overlay.close()
        except Exception:
            pass
    backend.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
