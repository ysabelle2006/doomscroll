"""Phase 2: end-to-end trigger -- calibrated gaze mapping + attention gate +
a 3-tier content-aware scroll decision (see fixation/scroll_trigger.py):

  1. Gaze in the bottom 15% of the focused window -> definitely scroll
     (short dwell).
  2. Else, if OCR shows gaze has read down to the last visible on-screen
     text and moved past it -> scroll.
  3. Else, ask the content tracker whether anything unseen remains between
     gaze and the bottom of the window -- if not, it's fine to scroll
     (longer, more conservative dwell since this tier is the broadest).

OCR (content/ package) runs on a background thread, re-scanning every
~0.4s, so screen-capture + Vision latency never blocks this loop's camera
framerate. If no ContentScanner is available for this platform/install
(see content/__init__.py), tiers 2 and 3 are simply unreachable and this
falls back to tier 1 only -- not a crash.

A live, click-through overlay (see debug/screen_overlay.py) is drawn
directly on top of whatever window you're reading, showing every
OCR-recognized block's box plus the current gaze dot, so you can see
directly whether OCR is finding the right text and whether the gaze dot
is landing where you're really looking, before assuming the
scroll-trigger logic itself is wrong. Box colors: blue = the lowest
block on screen (what has_reached_end() depends on), green = already
"seen", red = not yet seen; the yellow line is the lowest block's "end
of content" threshold; the magenta dot is the current gaze point. The
overlay window is excluded from both the OCR scan itself and from
focused-window detection (see that module's docstring for how/why), so
it can't confuse either one. If pyobjc's Cocoa isn't available, the
overlay is silently skipped -- everything else still runs.

--model (default: math) picks which calibrated pipeline predicts the
gaze point: 'math' loads the GazeCalibrator profile scripts/calibrate.py
saves by default; 'cnn' instead loads the CNN trained offline on
data_raw (scripts/train_cnn.py) plus whatever per-user AffineCorrection
scripts/calibrate.py --model cnn has saved for it (see
gaze/cnn_calibration.py) -- uncorrected if calibrate.py --model cnn
hasn't been run yet.

Run scripts/calibrate.py first (--model math or --model cnn, matching
whichever --model you pass here). Then:
  .venv\\Scripts\\python.exe scripts\\phase2_gaze_scroll.py
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
from doomscroll.gaze.cnn_calibration import DEFAULT_CNN_PATH
from doomscroll.gaze.feature_backend import GazeVectorExtractor, make_gaze_extractor
from doomscroll.ml.model import load_gaze_model
from doomscroll.gaze.gaze_vector import DEFAULT_MIN_EYE_ASPECT_RATIO, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.gaze.smoothing import GazePointSmoother
from doomscroll.landmarks.mediapipe_backend import LEFT_IRIS, RIGHT_IRIS, MediaPipeLandmarkBackend
from doomscroll.scroll import ScrollUnit, get_scroll_injector
from doomscroll.windowing import get_window_bounds_provider

SCROLL_STEP_LINES = 6
DEBUG_WINDOW_TITLE = "doomscroll - Phase 2"  # a label only -- see windowing/macos_backend.py for how self-focus is actually excluded

# A degree-2 polynomial calibration model has no bound on its output for
# inputs that drift outside the range seen during calibration. Unlike a
# single-frame spike, a *persistently* out-of-range prediction isn't caught
# by the Kalman filter's innovation gating (the filter's own state drifts to
# agree with a sustained trend, so gating stops treating it as an outlier).
# Clamp the raw prediction to a generous margin around the physical screen
# before it ever reaches smoothing, as a robustness backstop independent of
# whatever's causing the extrapolation.
EXTRAPOLATION_MARGIN = 0.5  # allow up to 50% of screen size beyond each edge


# --- Backend setup -------------------------------------------------------
#
# Loads the saved calibration profile (see calibration.py's
# GazeCalibrator.load()) and builds the matching live feature extractor.
# Everything below this function -- the capture loop, build_feature_vector(),
# calibrator.predict() -- just calls gaze_extractor.extract() and trusts
# calibrator was fit on exactly that output shape.


def _init_gaze_backend() -> tuple[GazeCalibrator, GazeVectorExtractor]:
    try:
        calibrator = GazeCalibrator.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    gaze_extractor = make_gaze_extractor()
    return calibrator, gaze_extractor


def _init_cnn_backend():
    """--model cnn: the CNN trained offline by scripts/train_cnn.py, with
    whatever per-user AffineCorrection scripts/calibrate.py --model cnn
    has saved for it already applied (see gaze/cnn_calibration.py) --
    uncorrected if that hasn't been run yet, same as the math pipeline
    running without having calibrated."""
    try:
        return load_gaze_model(DEFAULT_CNN_PATH, with_correction=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: calibrated gaze mapping + content-aware scroll trigger")
    parser.add_argument(
        "--model",
        choices=["math", "cnn"],
        default="math",
        help="Which calibrated pipeline predicts the gaze point (default: math) -- 'math' loads "
        "the GazeCalibrator profile scripts/calibrate.py saves by default; 'cnn' loads the CNN "
        "trained offline on data_raw plus whatever per-user correction scripts/calibrate.py "
        "--model cnn has saved for it (see gaze/cnn_calibration.py).",
    )
    parser.add_argument(
        "--content-only",
        action="store_true",
        help=(
            "TEMPORARY TEST FLAG: disables the bottom-15%% tier so scrolling relies purely on "
            "content analysis (tiers 2+3 -- reached-end-of-content / nothing-unseen-ahead). For "
            "gauging how well content-awareness alone performs; not meant for regular use -- the "
            "default (no flag) keeps all 3 tiers."
        ),
    )
    parser.add_argument(
        "--smoothing-process-var",
        type=float,
        default=8.0,
        help="GazePointSmoother's Kalman process_var (default: 8.0, down from the smoothing "
        "module's own 20.0 default). Lower = trusts the motion model more / raw measurements "
        "less -- less per-frame jitter, a bit more lag. Raise this if the point feels sluggish "
        "to catch up with real eye movement.",
    )
    parser.add_argument(
        "--smoothing-measurement-var",
        type=float,
        default=1.5,
        help="GazePointSmoother's Kalman measurement_var (default: 1.5, up from the smoothing "
        "module's own 0.5 default). Higher = trusts each raw measurement less -- less jitter, "
        "more lag.",
    )
    parser.add_argument(
        "--smoothing-ema-alpha",
        type=float,
        default=0.7,
        help="GazePointSmoother's extra EMA pass on top of the Kalman output (default: 0.7, up "
        "from the smoothing module's own 0.5 default; 0.0 disables it). Higher = smoother, more "
        "lag.",
    )
    parser.add_argument(
        "--blink-min-ear",
        type=float,
        default=DEFAULT_MIN_EYE_ASPECT_RATIO,
        help=f"Eye-aspect-ratio floor below which a frame is treated as a blink and skipped "
        f"entirely (not fed to the smoother) rather than updating the gaze point (default: "
        f"{DEFAULT_MIN_EYE_ASPECT_RATIO}, gaze_vector.py's own default). This is deliberately "
        "looser than scripts/calibrate.py's CALIBRATION_MIN_EAR (0.12): a narrowed-but-not-"
        "closed eye (which happens on a genuine downward glance, not just a blink) still gives "
        "a biased iris reading, but rejecting those frames too eagerly here risks treating a "
        "real glance as a blink and skipping it. If your gaze point visibly dips and curves "
        "(a short semicircle/diagonal down-and-back) right around every real blink rather than "
        "just freezing, that's exactly this gap: a few narrowed-eye frames just before/after "
        "full closure are passing this looser gate and feeding a biased reading into the "
        "smoother. Try raising this toward 0.12 (matching calibration) to reject more of "
        "them -- at the cost of possibly skipping a couple of frames on a fast downward glance "
        "too, which should be far less noticeable than the dip itself.",
    )
    args = parser.parse_args()

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    clamp_x = (-EXTRAPOLATION_MARGIN * screen_w, (1 + EXTRAPOLATION_MARGIN) * screen_w)
    clamp_y = (-EXTRAPOLATION_MARGIN * screen_h, (1 + EXTRAPOLATION_MARGIN) * screen_h)

    cnn_model = None
    calibrator = gaze_extractor = None
    if args.model == "cnn":
        cnn_model = _init_cnn_backend()
    else:
        calibrator, gaze_extractor = _init_gaze_backend()

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    smoother = GazePointSmoother(
        process_var=args.smoothing_process_var,
        measurement_var=args.smoothing_measurement_var,
        ema_alpha=args.smoothing_ema_alpha,
    )
    attention_gate = AttentionGate()
    window_provider = get_window_bounds_provider()
    injector = get_scroll_injector()

    last_external_bounds = None

    # BackgroundContentScanner calls this repeatedly, on its own thread,
    # over the lifetime of the run -- pass a callable (Python closures are
    # late-binding) rather than the current value of last_external_bounds,
    # which would just freeze at whatever it was when the thread started.
    def _current_bounds():
        return last_external_bounds

    content_available = True
    try:
        content_scanner = get_content_scanner()
    except Exception as e:
        print(f"Content-aware scrolling unavailable ({e}) -- falling back to bottom-of-window-only scrolling.")
        content_scanner = None
        content_available = False

    bg_scanner = BackgroundContentScanner(content_scanner, get_bounds=_current_bounds) if content_scanner else None
    content_tracker = ContentGazeTracker()

    try:
        screen_overlay = ScreenOverlay()
    except Exception as e:
        print(f"Live content overlay unavailable ({e}) -- continuing without it.")
        screen_overlay = None
    # Flips to True the first time screen_overlay.update() throws at
    # runtime, so a persistently broken overlay prints once and then gets
    # out of the way instead of spamming a traceback every frame.
    overlay_broken = False

    trigger_config = ScrollTriggerConfig()
    if args.content_only:
        print("--content-only: bottom-15% tier disabled -- scrolling relies purely on content analysis.")
        # bottom_definite_fraction=inf: fy is clamped to [0, 1]
        # (to_window_relative_fraction), so no real fy can ever reach an
        # infinite threshold -- tier 1 becomes permanently unreachable
        # without touching ScrollTrigger's own tier-selection logic at all.
        trigger_config = ScrollTriggerConfig(bottom_definite_fraction=float("inf"))
    trigger = ScrollTrigger(content_tracker, trigger_config)

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
            elif points_px is not None and head_pose is not None and not is_gaze_usable(points_px, min_ear=args.blink_min_ear):
                status_lines.append("blink...")
                trigger.reset_dwell()
            elif points_px is not None and head_pose is not None and args.model == "math" and (
                gaze_vector := gaze_extractor.extract(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), points_px)
            ) is None:
                status_lines.append("gaze vector unavailable this frame...")
                trigger.reset_dwell()
            elif points_px is not None and head_pose is not None:
                if args.model == "cnn":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    raw_point = cnn_model.predict_px(points_px, head_pose, gray, (screen_w, screen_h))
                    status_lines.append(f"cnn pred=({raw_point[0]:.0f},{raw_point[1]:.0f})")
                else:
                    feature = build_feature_vector(
                        gaze_vector,
                        (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg),
                        tuple(head_pose.translation_vector.flatten()),
                    )
                    # Diagnostic: shows the raw leading gaze-vector (eye-in-head
                    # offset) plus head pose, so we can see whether the live
                    # feature is actually tracking where you're looking before
                    # blaming the fitted model.
                    status_lines.append(
                        "feat gaze=(" + ",".join(f"{v:.3f}" for v in gaze_vector) + ") "
                        f"head=({head_pose.yaw_deg:.1f},{head_pose.pitch_deg:.1f},{head_pose.roll_deg:.1f})"
                    )
                    raw_point = calibrator.predict(feature)
                clamped_point = np.array(
                    [np.clip(raw_point[0], *clamp_x), np.clip(raw_point[1], *clamp_y)]
                )
                if not np.array_equal(clamped_point, raw_point):
                    status_lines.append(f"!! prediction clamped from raw=({raw_point[0]:.0f},{raw_point[1]:.0f})")
                smoothed_point = smoother.update(clamped_point, timestamp_ms)

                # Feed the content-aware dwell/"seen" tracking on every tick
                # we have a usable gaze point -- independent of whether a
                # scroll decision is being made this tick -- so dwell timing
                # toward "seen" doesn't reset just because bounds/trigger
                # evaluation below happens to be skipped.
                blocks = bg_scanner.latest_blocks() if bg_scanner is not None else []
                content_tracker.update(blocks, smoothed_point, timestamp_ms)

                current_bounds = window_provider.get_focused_window_bounds()
                # get_focused_window_bounds() itself already returns None
                # whenever this very process is the frontmost application
                # (see windowing/macos_backend.py) -- so our own webcam
                # preview window can never come back here as "the focused
                # window" in the first place. Just keep whatever the last
                # genuinely-external window was when that happens.
                if current_bounds is not None:
                    last_external_bounds = current_bounds
                bounds = last_external_bounds

                if screen_overlay is not None and not overlay_broken and bounds is not None and bounds.width > 0 and bounds.height > 0:
                    try:
                        screen_overlay.update(
                            blocks, bounds, smoothed_point, content_tracker.seen_ids, content_tracker.end_margin_px
                        )
                    except Exception as e:
                        print(f"Live content overlay broke ({e}) -- disabling it for the rest of this run.")
                        overlay_broken = True

                status_lines.append(f"raw=({raw_point[0]:.0f},{raw_point[1]:.0f}) smoothed=({smoothed_point[0]:.0f},{smoothed_point[1]:.0f})")
                status_lines.append(
                    f"content: {'on' if content_available else 'off'} blocks={len(blocks)} seen={len(content_tracker.seen_ids)}"
                )
                if bounds is None or bounds.width <= 0 or bounds.height <= 0:
                    status_lines.append("no usable focused window bounds")
                    trigger.reset_dwell()
                else:
                    status_lines.append(
                        f"bounds=({bounds.x},{bounds.y},{bounds.width}x{bounds.height}) app={bounds.app_name!r}"
                    )
                    fx, fy = to_window_relative_fraction(smoothed_point, bounds)
                    status_lines.append(f"gaze frac=({fx:.2f},{fy:.2f})")

                    viewport_bottom = bounds.y + bounds.height
                    should_fire, label = trigger.update(
                        fy, blocks, smoothed_point, viewport_bottom, timestamp_ms, content_available
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
