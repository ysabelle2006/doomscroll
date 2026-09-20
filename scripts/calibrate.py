"""Phase 2: 16-point on-screen gaze calibration, plus a head-pose variation
pass.

Main pass: keep your head facing forward, roughly toward the center of the
screen, and use ONLY your eyes to look at each dot as it appears -- don't
turn your head toward it. Small incidental fidgeting is fine (that's what
teaches the model to ignore it later), but deliberately turning your head to
face each dot would make head pose spuriously correlate with target
position in the training data, and the model would end up predicting screen
position mostly from head pose instead of from your actual eye movement --
backwards from what we want.

Head-pose variation pass: immediately after, you'll be asked to hold your
head in a few different natural positions (leaning left/right, closer/back,
tilting chin up/down) while repeating a short 5-point sequence at each --
still using only your eyes to track each dot, even while your head is
tilted. The main pass alone gives the model almost no variation in head
rotation or position to learn from (you're asked to stay still and forward
throughout), so at runtime, real head movement -- pitch (up/down) in
particular, since it's easy to tilt the chin without noticing -- lands
outside anything the model ever saw and the mapping degrades badly. This
pass gives the same target under genuinely different head positions, which
is what lets the model learn to correct for head motion instead of just
extrapolating blindly.

A short validation pass follows, reporting mean pixel error.
Press ESC at any time to abort.

Run: .venv\\Scripts\\python.exe scripts\\calibrate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.gaze.calibration import FEATURE_DIM, GazeCalibrator, build_feature_vector
from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

WINDOW_NAME = "doomscroll calibration"
SETTLE_S = 0.6
CAPTURE_S = 1.4
POINT_RADIUS = 18
TRANSITION_S = 2.2  # how long each head-pose-variant instruction is shown before capture starts

STILL_INSTRUCTION = "keep your head still & facing forward -- move only your eyes to the dot"

# 4x4 grid (16 points) instead of the earlier 3x3 (9) for denser spatial
# coverage -- more independent calibration events for RidgeCV's per-region
# regularization search to work with, and better coverage near the edges
# where the dwell-based scroll trigger actually needs to fire.
_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
TRAIN_GRID_FRACTIONS = [(fx, fy) for fy in _GRID_STEPS for fx in _GRID_STEPS]

# Held out, deliberately off the training grid.
VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]

# Reduced 5-point set (edge midpoints + center) repeated once per head-pose
# variant below. Deliberately distinct from both TRAIN_GRID_FRACTIONS and
# VALIDATION_FRACTIONS so this pass doesn't overlap either -- its purpose is
# pose variation, not additional spatial coverage.
POSE_VARIATION_FRACTIONS = [(0.08, 0.5), (0.92, 0.5), (0.5, 0.08), (0.5, 0.92), (0.5, 0.5)]

# Natural head-position variants: rotation (lean left/right turns the head
# slightly) and translation (closer/back changes distance and, on most desk
# setups, also height/angle a little) -- both of which the model previously
# never saw any variation in at all. Kept to "slight"/"natural" amounts
# deliberately: the point is to teach the model to tolerate incidental
# real-world head motion, not to calibrate for extreme poses it will rarely
# see in normal use.
POSE_VARIANTS = [
    ("lean LEFT", "Lean your head slightly to the LEFT and hold that position -- keep tracking dots with your eyes only."),
    ("lean RIGHT", "Lean your head slightly to the RIGHT and hold that position -- keep tracking dots with your eyes only."),
    ("lean CLOSER", "Lean slightly CLOSER to the screen and hold that position -- keep tracking dots with your eyes only."),
    ("sit BACK", "Sit slightly further BACK from the screen and hold that position -- keep tracking dots with your eyes only."),
    ("tilt UP", "Tilt your head slightly UP and hold that position -- keep tracking dots with your eyes only, don't drop your eyes to compensate."),
    ("tilt DOWN", "Tilt your head slightly DOWN (not too far) and hold that position -- keep tracking dots with your eyes only, don't raise your eyes to compensate."),
]


_display_size: tuple[int, int] | None = None  # set once in main(), see _detect_display_size()


def _detect_display_size(screen_w: int, screen_h: int) -> tuple[int, int]:
    """Ask OpenCV what pixel size it actually gave the fullscreen window,
    instead of trusting our own screen-size query (get_screen_size()) to
    exactly match it. Switching that query to Quartz (see dpi.py) was meant
    to fix the grey strip left uncovered along one edge of the window, but a
    gap was still observed afterwards -- so rather than keep guessing at the
    exact cause (Retina backing-store scaling, multi-monitor placement, or
    something else in how the Cocoa highgui backend sizes its content view),
    ask OpenCV directly and scale every rendered frame to fill *that* size.
    Falls back to (screen_w, screen_h) -- today's status quo -- if the query
    isn't supported or the window hasn't settled yet."""
    blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    # present twice: some backends only finish resizing the content view to
    # the fullscreen frame after the window has already been shown once
    for _ in range(2):
        cv2.imshow(WINDOW_NAME, blank)
        cv2.waitKey(150)
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
    except Exception:
        w, h = 0, 0
    if w > 0 and h > 0:
        return w, h
    return screen_w, screen_h


def _present(frame: np.ndarray) -> None:
    """Shows `frame` (drawn in the logical, DPI-aware screen_w x screen_h
    coordinate space that calibration math and the runtime gaze->scroll
    mapping both rely on) scaled to fill the real detected window size, so
    the two can differ without leaving any of the window uncovered."""
    if _display_size is not None and _display_size != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, _display_size, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME, frame)


def _wrap_line(text: str, font: int, font_scale: float, thickness: int, max_width: int) -> list[str]:
    """Greedily wraps `text` on spaces so every rendered line measures at
    most max_width pixels (via cv2.getTextSize). Without this, a long
    instruction string is drawn as a single line and silently runs off the
    right edge of the screen instead of wrapping -- which is exactly what
    happened with the longer pose-variant instructions (e.g. "tilt UP"/
    "tilt DOWN")."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_target(screen_w: int, screen_h: int, point_px: tuple[int, int], instruction: str = STILL_INSTRUCTION) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)

    # Wrap the bottom status text so long instructions (the pose-variant
    # strings, in particular) stay on screen instead of running off the
    # right edge -- see _wrap_line().
    font, font_scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 20
    lines = _wrap_line(instruction, font, font_scale, thickness, screen_w - 2 * margin)
    line_height = 26
    base_y = screen_h - 30  # baseline the single-line version used to sit on
    start_y = base_y - line_height * (len(lines) - 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (margin, start_y + i * line_height), font, font_scale, (150, 150, 150), thickness)
    return frame


def show_transition_message(screen_w: int, screen_h: int, message: str, duration_s: float = TRANSITION_S) -> None:
    """Displays a full-screen instruction (e.g. asking the user to shift into
    a new head position) before the next mini-sequence starts capturing, so
    they have time to get into position and hold it."""
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    font, font_scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2, 60

    # Each " -- "-separated segment is wrapped independently (the longer
    # pose-variant messages, e.g. "tilt DOWN", otherwise ran off the right
    # edge of the screen), and every resulting line is centered using its
    # own measured width rather than a fixed offset that only happened to
    # center one specific line length.
    lines: list[str] = []
    for segment in message.split(" -- "):
        lines.extend(_wrap_line(segment, font, font_scale, thickness, screen_w - 2 * margin))

    line_height = 40
    start_y = screen_h // 2 - (line_height * (len(lines) - 1)) // 2
    for i, line in enumerate(lines):
        (text_w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = screen_w // 2 - text_w // 2
        cv2.putText(frame, line, (x, start_y + i * line_height), font, font_scale, (0, 220, 255), thickness)

    _present(frame)
    deadline = time.time() + duration_s
    while time.time() < deadline:
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt


def collect_samples_for_point(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    point_px: tuple[int, int],
    instruction: str = STILL_INSTRUCTION,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    # Diagnostic counters -- when a point captures few/zero samples, these
    # say *why* (no face in frame vs. blink-rejected vs. head-pose solve
    # rejected) instead of leaving it a silent mystery. A run of "no_face"
    # almost always means the pose asked for pushed the face out of the
    # webcam's detection cone (e.g. laptop webcams already look down at you
    # from above the screen, so tilting the chin down further is the
    # direction most likely to do this).
    frames_seen = 0
    no_face = 0
    blink_rejected = 0
    pose_rejected = 0

    _present(draw_target(screen_w, screen_h, point_px, instruction))
    cv2.waitKey(1)

    settle_deadline = time.time() + SETTLE_S
    while time.time() < settle_deadline:
        cam.read()  # drain frames so capture below starts fresh
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    capture_deadline = time.time() + CAPTURE_S
    while time.time() < capture_deadline:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        frames_seen += 1
        timestamp_ms = cam.frame_timestamp_ms()
        result = backend.detect(frame, timestamp_ms)
        if result is None:
            no_face += 1
            continue
        points_px = result.points_px()
        if not is_gaze_usable(points_px):
            blink_rejected += 1
            continue  # mid-blink frame -- iris position unreliable, skip it
        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
        if head_pose is None:
            pose_rejected += 1
            continue
        gaze_feature = compute_gaze_feature(points_px)
        feature = build_feature_vector(
            gaze_feature.vector,
            (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg),
            tuple(head_pose.translation_vector.flatten()),
        )
        features.append(feature)
        targets.append(np.array(point_px, dtype=np.float64))

        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    if len(features) < 10:
        print(
            f"    !! only {len(features)}/{frames_seen} frames usable "
            f"(no_face={no_face}, blink={blink_rejected}, pose_rejected={pose_rejected})"
        )
        if no_face > frames_seen * 0.5:
            print("       -- face was mostly not detected here; this pose may be pushing your face out of the webcam's view")

    return features, targets


def run_sequence(
    cam,
    backend,
    pose_estimator,
    screen_w,
    screen_h,
    fractions,
    instruction: str = STILL_INSTRUCTION,
    group_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_groups: list[int] = []
    for i, (fx, fy) in enumerate(fractions):
        point_px = (int(fx * screen_w), int(fy * screen_h))
        print(f"  Point {i + 1}/{len(fractions)} at {point_px}...")
        features, targets = collect_samples_for_point(
            cam, backend, pose_estimator, screen_w, screen_h, point_px, instruction
        )
        print(f"    captured {len(features)} samples")
        all_features.extend(features)
        all_targets.extend(targets)
        all_groups.extend([group_offset + i] * len(features))  # tag every sample with which point it came from

    # np.array([]) on an empty list collapses to shape (0,) rather than
    # (0, FEATURE_DIM)/(0, 2) -- harmless on its own, but np.concatenate()
    # against the running (N, FEATURE_DIM) accumulator in main() then fails
    # with a dimension mismatch. This happens for real whenever every single
    # point in a sequence captures zero samples (e.g. a pose variant that
    # loses face tracking throughout) -- not just a theoretical edge case.
    # Force the correct 2D shape explicitly so an all-empty sequence is a
    # harmless no-op contribution instead of a crash.
    if not all_features:
        return np.empty((0, FEATURE_DIM)), np.empty((0, 2)), np.empty((0,), dtype=int)
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


def main() -> None:
    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    global _display_size
    _display_size = _detect_display_size(screen_w, screen_h)
    if _display_size != (screen_w, screen_h):
        print(f"Note: fullscreen window renders at {_display_size[0]}x{_display_size[1]} "
              f"vs logical {screen_w}x{screen_h}; scaling frames to fill it exactly.")

    try:
        with WebcamSource(0) as cam:
            print("Calibration starting. Keep your head still & forward -- move only your eyes to each dot. ESC to abort.")
            print("Training points:")
            all_features, all_targets, all_groups = run_sequence(
                cam, backend, pose_estimator, screen_w, screen_h, TRAIN_GRID_FRACTIONS, STILL_INSTRUCTION
            )
            next_group_id = len(TRAIN_GRID_FRACTIONS)

            print("\nHead-pose variation pass -- this teaches the model to keep tracking")
            print("correctly after you shift position, instead of only ever seeing you")
            print("sit still and forward.")
            for label, instruction in POSE_VARIANTS:
                show_transition_message(screen_w, screen_h, instruction)
                print(f"  Variant: {label}")
                features, targets, groups = run_sequence(
                    cam,
                    backend,
                    pose_estimator,
                    screen_w,
                    screen_h,
                    POSE_VARIATION_FRACTIONS,
                    instruction,
                    group_offset=next_group_id,
                )
                next_group_id += len(POSE_VARIATION_FRACTIONS)
                if features.shape[0] == 0:
                    print(
                        f"    !! captured nothing usable for '{label}' -- this variant contributes "
                        "no data to the model. The pose was likely too extreme (face lost by the "
                        "webcam) or held too briefly. Consider re-running with a gentler version of "
                        "this pose."
                    )
                all_features = np.concatenate([all_features, features])
                all_targets = np.concatenate([all_targets, targets])
                all_groups = np.concatenate([all_groups, groups])

            calibrator = GazeCalibrator()
            calibrator.fit(all_features, all_targets, groups=all_groups)
            print(f"\nFitted on {len(all_features)} samples across {next_group_id} calibration points.")
            print(f"RidgeCV selected alpha (x, y): {calibrator.selected_alpha}")

            # Gain/compression check: a heavily-regularized fit can minimize
            # validation error while still shrinking the model's dynamic
            # range -- i.e. predicted points cluster closer to the center
            # than the true targets do, so you have to look further than the
            # real edge to reach the mapped edge. Compare predicted spread to
            # actual target spread directly, over the training data (which
            # spans the full calibration grid, unlike the smaller held-out set).
            train_preds = np.array([calibrator.predict(f) for f in all_features])
            target_range = all_targets.max(axis=0) - all_targets.min(axis=0)
            pred_range = train_preds.max(axis=0) - train_preds.min(axis=0)
            gain = pred_range / np.where(target_range == 0, 1, target_range)
            print(f"Gain (predicted range / actual range), x,y: {gain} -- 1.0 is ideal, well below 1.0 means compressed")

            print("Validation points:")
            # Without this, validation starts immediately after the last
            # pose variant (tilt DOWN) with only the small grey status text
            # changing back to the "keep still" instruction -- easy to miss,
            # leaving it unclear to the user that they should stop tilting
            # their head and return to a normal position first.
            show_transition_message(
                screen_w,
                screen_h,
                "Return your head to a normal, centered position -- keep tracking dots with your eyes only.",
            )
            val_features, val_targets, _val_groups = run_sequence(
                cam, backend, pose_estimator, screen_w, screen_h, VALIDATION_FRACTIONS
            )
            if len(val_features) > 0:
                mean_error_px = calibrator.validate(val_features, val_targets)
                screen_diag = (screen_w**2 + screen_h**2) ** 0.5
                print(
                    f"Validation mean error: {mean_error_px:.1f}px "
                    f"({100 * mean_error_px / screen_diag:.1f}% of screen diagonal)"
                )
                if mean_error_px / screen_diag > 0.08:
                    print("Accuracy looks poor -- consider recalibrating with better/more even lighting.")
            else:
                print("No validation samples captured (face not detected during validation).")

            calibrator.save()
            print("Calibration profile saved.")
    except KeyboardInterrupt:
        print("Calibration aborted.")
    finally:
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
