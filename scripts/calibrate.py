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
from doomscroll.gaze.calibration import GazeCalibrator, build_feature_vector
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
    ("tilt UP", "Tilt your head slightly UP (chin up) and hold that position -- keep tracking dots with your eyes only, don't drop your eyes to compensate."),
    ("tilt DOWN", "Tilt your head slightly DOWN (chin down) and hold that position -- keep tracking dots with your eyes only, don't raise your eyes to compensate."),
]


def draw_target(screen_w: int, screen_h: int, point_px: tuple[int, int], instruction: str = STILL_INSTRUCTION) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)
    cv2.putText(
        frame,
        instruction,
        (20, screen_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )
    return frame


def show_transition_message(screen_w: int, screen_h: int, message: str, duration_s: float = TRANSITION_S) -> None:
    """Displays a full-screen instruction (e.g. asking the user to shift into
    a new head position) before the next mini-sequence starts capturing, so
    they have time to get into position and hold it."""
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    for i, line in enumerate(message.split(" -- ")):
        cv2.putText(
            frame,
            line,
            (screen_w // 2 - 420, screen_h // 2 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 255),
            2,
        )
    cv2.imshow(WINDOW_NAME, frame)
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

    cv2.imshow(WINDOW_NAME, draw_target(screen_w, screen_h, point_px, instruction))
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
        timestamp_ms = cam.frame_timestamp_ms()
        result = backend.detect(frame, timestamp_ms)
        if result is None:
            continue
        points_px = result.points_px()
        if not is_gaze_usable(points_px):
            continue  # mid-blink frame -- iris position unreliable, skip it
        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
        if head_pose is None:
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
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


def main() -> None:
    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
