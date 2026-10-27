"""Multi-angle on-screen gaze calibration.

Feature vector combines MediaPipe iris-offset + head-pose features
(gaze_vector.py, head_pose.py), where the iris position itself is refined
per-frame via a classical image-processing pass (pupil_refine.py) layered
on top of MediaPipe's landmark. That refined feature vector feeds
calibration.py's GazeCalibrator, which now also fits a local
residual-correction stage on top of the base linear model.

WITHIN each angle block below, keep your head fixed at that block's
orientation and use ONLY your eyes to follow the dots -- exactly as in the
original single-angle calibration, and for the same reason: letting head
pose vary *with* target position inside a block would make the model learn
"screen position predicts head pose" instead of using your actual eye
movement. ACROSS blocks, head orientation is deliberately varied by design,
so the model sees how the same set of screen positions maps under different
head orientations, rather than only ever training on (near-)forward-facing
head pose and hoping small incidental jitter generalizes far enough. This
is an extension of the plan's original head-pose-as-regressor-input idea,
not a departure from it.

A short validation pass follows (front-facing), reporting mean pixel error.
Press ESC at any time to abort.

Run: python scripts/calibrate.py
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.gaze.calibration import GazeCalibrator, build_feature_vector
from doomscroll.gaze.gaze_vector import compute_gaze_feature, is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.gaze.session_store import (
    clear_sessions,
    list_sessions,
    pool_sessions,
    save_session,
)
from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

WINDOW_NAME = "doomscroll calibration"
SETTLE_S = 0.6
CAPTURE_S = 1.4
POINT_RADIUS = 18

# Main (front-facing) grid: denser than a single-angle calibration needs on
# its own, since this now anchors the largest, most-trusted block, and
# feeds the local-correction stage's neighbor density (see calibration.py).
_MAIN_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
MAIN_GRID_FRACTIONS = [(fx, fy) for fy in _MAIN_GRID_STEPS for fx in _MAIN_GRID_STEPS]

# Off-angle blocks use a lighter 3x3 grid each -- enough to teach the model
# how gaze-to-screen mapping shifts under that head pose, without pushing
# total calibration time too high across five blocks.
_ANGLE_GRID_STEPS = [0.15, 0.5, 0.85]
ANGLE_GRID_FRACTIONS = [(fx, fy) for fy in _ANGLE_GRID_STEPS for fx in _ANGLE_GRID_STEPS]

VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]


@dataclass
class AngleBlock:
    label: str
    instruction: str
    grid_fractions: list[tuple[float, float]]
    # Rough target range for HeadPoseEstimator's yaw_deg/pitch_deg during
    # this block, used only to show the user live feedback -- not enforced.
    yaw_range: tuple[float, float] = (-10.0, 10.0)
    pitch_range: tuple[float, float] = (-10.0, 10.0)


ANGLE_BLOCKS = [
    AngleBlock("front", "Face FORWARD, toward the center of the screen", MAIN_GRID_FRACTIONS),
    AngleBlock(
        "left", "Turn your HEAD LEFT ~25 degrees, keep eyes free to move", ANGLE_GRID_FRACTIONS,
        yaw_range=(-40.0, -15.0),
    ),
    AngleBlock(
        "right", "Turn your HEAD RIGHT ~25 degrees, keep eyes free to move", ANGLE_GRID_FRACTIONS,
        yaw_range=(15.0, 40.0),
    ),
    AngleBlock(
        "up", "Tilt your HEAD slightly UP (chin up ~15 degrees)", ANGLE_GRID_FRACTIONS,
        pitch_range=(10.0, 30.0),
    ),
    AngleBlock(
        "down", "Tilt your HEAD slightly DOWN (chin down ~15 degrees)", ANGLE_GRID_FRACTIONS,
        pitch_range=(-30.0, -10.0),
    ),
]


def draw_target(screen_w: int, screen_h: int, point_px: tuple[int, int], label: str) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"[{label}] move only your eyes to the dot",
        (20, screen_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (150, 150, 150),
        1,
    )
    return frame


def wait_for_ready(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    block: AngleBlock,
) -> None:
    """Shows the block's instruction plus a live head-pose readout, and
    waits for SPACE before starting that block's dot grid. Self-paced
    rather than a fixed timer, since finding and holding an off-angle head
    pose takes people a different amount of time than facing forward does."""
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        timestamp_ms = cam.frame_timestamp_ms()
        result = backend.detect(frame, timestamp_ms)
        yaw_deg = pitch_deg = None
        if result is not None:
            points_px = result.points_px()
            head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
            if head_pose is not None:
                yaw_deg, pitch_deg = head_pose.yaw_deg, head_pose.pitch_deg

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.putText(canvas, block.instruction, (40, screen_h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if yaw_deg is not None:
            in_range = (
                block.yaw_range[0] <= yaw_deg <= block.yaw_range[1]
                and block.pitch_range[0] <= pitch_deg <= block.pitch_range[1]
            )
            color = (0, 220, 0) if in_range else (0, 165, 255)
            cv2.putText(
                canvas,
                f"yaw={yaw_deg:.1f} (target {block.yaw_range})  pitch={pitch_deg:.1f} (target {block.pitch_range})",
                (40, screen_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
        else:
            cv2.putText(canvas, "face not detected", (40, screen_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(canvas, "Press SPACE when in position (ESC to abort)", (40, screen_h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:  # SPACE
            return


def collect_samples_for_point(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    point_px: tuple[int, int],
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    cv2.imshow(WINDOW_NAME, draw_target(screen_w, screen_h, point_px, label))
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

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaze_feature = compute_gaze_feature(points_px, frame_gray=frame_gray)
        feature = build_feature_vector(
            gaze_feature.vector, (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg)
        )
        features.append(feature)
        targets.append(np.array(point_px, dtype=np.float64))

        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    return features, targets


def run_block(
    cam, backend, pose_estimator, screen_w, screen_h, block: AngleBlock, group_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wait_for_ready(cam, backend, pose_estimator, screen_w, screen_h, block)

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_groups: list[int] = []
    for i, (fx, fy) in enumerate(block.grid_fractions):
        point_px = (int(fx * screen_w), int(fy * screen_h))
        print(f"  [{block.label}] point {i + 1}/{len(block.grid_fractions)} at {point_px}...")
        features, targets = collect_samples_for_point(
            cam, backend, pose_estimator, screen_w, screen_h, point_px, block.label
        )
        print(f"    captured {len(features)} samples")
        all_features.extend(features)
        all_targets.extend(targets)
        all_groups.extend([group_offset + i] * len(features))
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


def run_sequence(
    cam, backend, pose_estimator, screen_w, screen_h, fractions
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_groups: list[int] = []
    for i, (fx, fy) in enumerate(fractions):
        point_px = (int(fx * screen_w), int(fy * screen_h))
        print(f"  Validation point {i + 1}/{len(fractions)} at {point_px}...")
        features, targets = collect_samples_for_point(
            cam, backend, pose_estimator, screen_w, screen_h, point_px, "validation"
        )
        print(f"    captured {len(features)} samples")
        all_features.extend(features)
        all_targets.extend(targets)
        all_groups.extend([i] * len(features))
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="doomscroll multi-angle gaze calibration")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Pool this session with recent stored sessions when fitting (see session_store.py "
        "for why this is opt-in rather than the default). Prints both the this-session-only and "
        "pooled validation errors so you can compare.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Delete all stored calibration sessions and exit. Do this after any real setup "
        "change (new lighting, seating, camera position).",
    )
    parser.add_argument(
        "--list-history",
        action="store_true",
        help="List stored calibration sessions and exit.",
    )
    args = parser.parse_args()

    if args.clear_history:
        removed = clear_sessions()
        print(f"Cleared {removed} stored calibration session(s).")
        return

    if args.list_history:
        sessions = list_sessions()
        if not sessions:
            print("No stored calibration sessions.")
        for s in sessions:
            print(f"  {s.label}  {len(s.features):5d} samples  screen={s.screen_size[0]}x{s.screen_size[1]}")
        return

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # On macOS, setWindowProperty(FULLSCREEN) alone doesn't reliably stretch
    # the window to the true screen bounds on the first frame -- explicit
    # moveWindow/resizeWindow plus a short settle/redraw cycle is the
    # standard workaround for OpenCV's Cocoa-backed HighGUI fullscreen
    # quirks. Harmless no-op on other platforms.
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    warmup_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    for _ in range(5):
        cv2.imshow(WINDOW_NAME, warmup_frame)
        cv2.waitKey(30)

    try:
        with WebcamSource(0) as cam:
            print("Calibration starting. ESC to abort at any time.")
            train_features_list, train_targets_list, train_groups_list = [], [], []
            group_offset = 0
            for block in ANGLE_BLOCKS:
                feats, targs, groups = run_block(
                    cam, backend, pose_estimator, screen_w, screen_h, block, group_offset
                )
                if len(feats) > 0:
                    train_features_list.append(feats)
                    train_targets_list.append(targs)
                    train_groups_list.append(groups)
                group_offset += len(block.grid_fractions)

            session_features = np.concatenate(train_features_list, axis=0)
            session_targets = np.concatenate(train_targets_list, axis=0)
            session_groups = np.concatenate(train_groups_list, axis=0)

            # Always persist the raw samples, regardless of --pool: storing
            # them costs little and makes pooling available later, whereas
            # discarding them is irreversible.
            save_session(session_features, session_targets, session_groups, (screen_w, screen_h))
            print(f"Session saved ({len(session_features)} samples).")

            print("Validation points (front-facing):")
            val_features, val_targets, _val_groups = run_sequence(
                cam, backend, pose_estimator, screen_w, screen_h, VALIDATION_FRACTIONS
            )
            screen_diag = (screen_w**2 + screen_h**2) ** 0.5

            def fit_and_report(features, targets, groups, label: str) -> tuple[GazeCalibrator, float | None]:
                calibrator = GazeCalibrator()
                calibrator.fit(features, targets, groups=groups)
                n_points = len(np.unique(groups))
                print(f"\n[{label}] fitted on {len(features)} samples across {n_points} points.")
                print(f"[{label}] RidgeCV selected alpha (x, y): {calibrator.selected_alpha}")

                train_preds = np.array([calibrator.predict(f) for f in features])
                target_range = targets.max(axis=0) - targets.min(axis=0)
                pred_range = train_preds.max(axis=0) - train_preds.min(axis=0)
                gain = pred_range / np.where(target_range == 0, 1, target_range)
                print(f"[{label}] gain (pred range / actual range) x,y: {gain} -- 1.0 ideal, well below 1.0 = compressed")

                err = None
                if len(val_features) > 0:
                    err = calibrator.validate(val_features, val_targets)
                    print(
                        f"[{label}] validation mean error: {err:.1f}px "
                        f"({100 * err / screen_diag:.1f}% of screen diagonal)"
                    )
                return calibrator, err

            session_calibrator, session_err = fit_and_report(
                session_features, session_targets, session_groups, "this session only"
            )
            best_calibrator, best_err, best_label = session_calibrator, session_err, "this session only"

            if args.pool:
                pooled_features, pooled_targets, pooled_groups, used = pool_sessions(
                    list_sessions(), (screen_w, screen_h)
                )
                if len(used) <= 1:
                    print("\nNothing to pool yet (this is the only stored session at this screen size).")
                else:
                    print(f"\nPooling {len(used)} sessions: " + ", ".join(s.label for s in used))
                    pooled_calibrator, pooled_err = fit_and_report(
                        pooled_features, pooled_targets, pooled_groups, "pooled"
                    )
                    # Keep whichever actually validates better -- pooling is
                    # not assumed to help (see session_store.py docstring).
                    if pooled_err is not None and session_err is not None and pooled_err < session_err:
                        best_calibrator, best_err, best_label = pooled_calibrator, pooled_err, "pooled"

            if best_err is not None and best_err / screen_diag > 0.08:
                print("\nAccuracy looks poor -- consider recalibrating with better/more even lighting.")
            if len(val_features) == 0:
                print("\nNo validation samples captured (face not detected during validation).")

            best_calibrator.save()
            print(f"\nCalibration profile saved (using: {best_label}).")
    except KeyboardInterrupt:
        print("Calibration aborted.")
    finally:
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
