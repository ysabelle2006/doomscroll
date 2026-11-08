# Math-model gaze calibration (GazeCalibrator): dot-grid or trajectory
# protocol, plus a shared head-pose variation pass and validation.

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
from doomscroll.gaze.calibration import DEFAULT_PROFILE_PATH, FEATURE_DIM, GazeCalibrator, build_feature_vector
from doomscroll.gaze.feature_backend import GazeVectorExtractor, make_gaze_extractor
from doomscroll.gaze.gaze_vector import is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator, HeadPose
from doomscroll.gaze.session_store import clear_sessions, list_sessions, pool_sessions, save_session
from doomscroll.gaze.trajectory import (
    ANCHOR_SETTLE_MS,
    build_keyframes,
    build_waypoints,
    measure_velocity,
    turn_discard_mask,
)
from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

WINDOW_NAME = "doomscroll calibration"
POINT_RADIUS = 18

CALIBRATION_MIN_EAR = 0.12


STILL_INSTRUCTION = "Keep your head COMPLETELY still -- follow the dot with your eyes only"

SETTLE_S = 0.6
CAPTURE_S = 1.4

# 4x4 grid of calibration dot positions (as screen fractions).
_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
TRAIN_GRID_FRACTIONS = [(fx, fy) for fy in _GRID_STEPS for fx in _GRID_STEPS]

# Held-out dot positions used to measure calibration accuracy.
VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]

# Dot positions used during each head-pose variant pass.
POSE_VARIATION_FRACTIONS = [(0.08, 0.5), (0.92, 0.5), (0.5, 0.08), (0.5, 0.92), (0.5, 0.5)]

FRONT_YAW_PITCH_TOLERANCE_DEG = 8.0
FRONT_READY_INSTRUCTION = (
    "Sit facing forward, centered in frame, at whatever distance feels natural "
    "-- this is where the main pass will track your eyes from."
)

FACE_CROP_PAD_FRAC = 0.25


TRAJECTORY_DURATION_S = 40.0
TRAJECTORY_POSE_VARIANT_DURATION_S = 15.0
TRAJECTORY_VALIDATION_DURATION_S = 18.0

TRAJECTORY_LAG_MS = 220.0

TRAJECTORY_SETTLE_S = 1.0

TRAJECTORY_TURN_DISCARD_MS = 250.0

TRAJECTORY_GROUP_WINDOW_MS = 1500.0

# Actual fullscreen window size, detected once in main().
_display_size: tuple[int, int] | None = None


# One head-pose instruction plus the pose range considered "in range" for it.
@dataclass
class PoseVariant:
    label: str
    instruction: str
    yaw_range: tuple[float, float] | None = None
    pitch_range: tuple[float, float] | None = None
    z_delta_range: tuple[float, float] | None = None


# The six head-pose variants captured during training (left/right/closer/back/up/down).
POSE_VARIANTS = [
    PoseVariant(
        "lean LEFT",
        "Lean your head slightly to the LEFT and hold that position -- keep tracking with your eyes only.",
        yaw_range=(-30.0, -20.0),
    ),
    PoseVariant(
        "lean RIGHT",
        "Lean your head slightly to the RIGHT and hold that position -- keep tracking with your eyes only.",
        yaw_range=(20.0, 30.0),
    ),
    PoseVariant(
        "lean CLOSER",
        "Lean slightly CLOSER to the screen and hold that position -- keep tracking with your eyes only.",
        z_delta_range=(-200.0, -150.0),
    ),
    PoseVariant(
        "sit BACK",
        "Sit slightly further BACK from the screen and hold that position -- keep tracking with your eyes only.",
        z_delta_range=(100.0, 130.0),
    ),
    PoseVariant(
        "tilt UP",
        "Tilt your head slightly UP and hold that position -- keep tracking with your eyes only, don't drop your eyes to compensate.",
        pitch_range=(15.0, 20.0),
    ),
    PoseVariant(
        "tilt DOWN",
        "Tilt your head slightly DOWN (not too far) and hold that position -- keep tracking with your eyes only, don't raise your eyes to compensate.",
        pitch_range=(-15.0, -8.0),
    ),
]


# Detects the fullscreen window's actual rendered pixel size.
def _detect_display_size(screen_w: int, screen_h: int) -> tuple[int, int]:
    blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
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


# Scales a point from the window's rendered size to logical screen coordinates.
def _map_real_to_logical(
    point: tuple[float, float], screen_w: int, screen_h: int, display_size: tuple[int, int] | None
) -> tuple[float, float]:
    if display_size is None or display_size == (screen_w, screen_h):
        return point
    scale_x = screen_w / display_size[0]
    scale_y = screen_h / display_size[1]
    return (point[0] * scale_x, point[1] * scale_y)


# Shows a frame in the calibration window, resizing it to the display size if needed.
def _present(frame: np.ndarray) -> None:
    if _display_size is not None and _display_size != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, _display_size, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME, frame)


# Resizes a webcam frame to screen size, or returns a black frame if none.
def _camera_background(frame: np.ndarray | None, screen_w: int, screen_h: int) -> np.ndarray:
    if frame is None:
        return np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    return cv2.resize(frame, (screen_w, screen_h))


# Darkens a horizontal band of the canvas, used behind instruction text.
def _darken_band(canvas: np.ndarray, y0: int, y1: int, alpha: float = 0.55) -> None:
    y0, y1 = max(0, y0), min(canvas.shape[0], y1)
    if y1 <= y0:
        return
    canvas[y0:y1] = (canvas[y0:y1].astype(np.float32) * (1 - alpha)).astype(np.uint8)


# Wraps text into lines that fit within max_width.
def _wrap_line(text: str, font: int, font_scale: float, thickness: int, max_width: int) -> list[str]:
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


# Returns whether a face is in frame and its padded bounding box.
def _face_frame_status(
    points_px: np.ndarray | None, image_width: int, image_height: int
) -> tuple[str, bool, tuple[float, float, float, float] | None]:
    if points_px is None:
        return "no face detected -- move into the camera's view", False, None

    x_min, y_min = points_px.min(axis=0)
    x_max, y_max = points_px.max(axis=0)
    pad_x = (x_max - x_min) * FACE_CROP_PAD_FRAC
    pad_y = (y_max - y_min) * FACE_CROP_PAD_FRAC
    box = (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)
    return "in frame", True, box


# Returns a live readout of how close the current head pose is to the target variant.
def _pose_variant_readout(head_pose: HeadPose | None, variant: PoseVariant, baseline_z: float | None) -> tuple[str, bool, float | None]:
    if head_pose is None:
        return "face not detected", False, baseline_z
    if variant.yaw_range is not None:
        lo, hi = variant.yaw_range
        in_range = lo <= head_pose.yaw_deg <= hi
        return f"yaw={head_pose.yaw_deg:+.1f} deg  (roughly {lo:+.0f} to {hi:+.0f})", in_range, baseline_z
    if variant.pitch_range is not None:
        lo, hi = variant.pitch_range
        in_range = lo <= head_pose.pitch_deg <= hi
        return f"pitch={head_pose.pitch_deg:+.1f} deg  (roughly {lo:+.0f} to {hi:+.0f})", in_range, baseline_z
    if variant.z_delta_range is not None:
        current_z = float(head_pose.translation_vector[2, 0])
        if baseline_z is None:
            baseline_z = current_z
        delta_z = current_z - baseline_z
        lo, hi = variant.z_delta_range
        in_range = lo <= delta_z <= hi
        return f"distance change={delta_z:+.0f}mm  (roughly {lo:+.0f} to {hi:+.0f}; negative=closer)", in_range, baseline_z
    return "", True, baseline_z


# Shows a head-pose instruction screen and waits for SPACE (or ESC to abort).
def wait_for_pose_ready(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    variant: PoseVariant,
) -> None:
    font, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 2, 60
    instruction_lines = _wrap_line(variant.instruction, font, 0.9, thickness, screen_w - 2 * margin)
    baseline_z: float | None = None

    while True:
        ok, frame = cam.read()
        head_pose = None
        face_status, in_frame, face_box = "no face detected -- move into the camera's view", False, None
        if ok and frame is not None:
            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)
            if result is not None:
                points_px = result.points_px()
                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                face_status, in_frame, face_box = _face_frame_status(points_px, result.image_width, result.image_height)

        readout, in_range, baseline_z = _pose_variant_readout(head_pose, variant, baseline_z)

        canvas = _camera_background(frame if ok else None, screen_w, screen_h)

        line_height = 40
        start_y = screen_h // 2 - 100 - (line_height * (len(instruction_lines) - 1)) // 2
        _darken_band(canvas, start_y - 40, screen_h // 2 + 150)
        for i, line in enumerate(instruction_lines):
            (text_w, _), _ = cv2.getTextSize(line, font, 0.9, thickness)
            x = screen_w // 2 - text_w // 2
            cv2.putText(canvas, line, (x, start_y + i * line_height), font, 0.9, (0, 220, 255), thickness)

        face_color = (0, 220, 0) if in_frame else (0, 0, 255)
        (face_w, _), _ = cv2.getTextSize(face_status, font, 0.7, 2)
        cv2.putText(canvas, face_status, (screen_w // 2 - face_w // 2, screen_h // 2 + 30), font, 0.7, face_color, 2)

        if readout:
            color = (0, 0, 255) if readout == "face not detected" else ((0, 220, 0) if in_range else (0, 165, 255))
            (text_w, _), _ = cv2.getTextSize(readout, font, 0.7, 2)
            cv2.putText(canvas, readout, (screen_w // 2 - text_w // 2, screen_h // 2 + 65), font, 0.7, color, 2)

        hint = "Press SPACE when ready (ESC to abort)"
        (hint_w, _), _ = cv2.getTextSize(hint, font, 0.7, 2)
        cv2.putText(canvas, hint, (screen_w // 2 - hint_w // 2, screen_h // 2 + 110), font, 0.7, (150, 150, 150), 2)

        _present(canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:
            return


# Returns a live readout of whether the head is facing forward.
def _front_readout(head_pose: HeadPose | None) -> tuple[str, bool]:
    if head_pose is None:
        return "face not detected", False
    yaw_deg, pitch_deg = head_pose.yaw_deg, head_pose.pitch_deg
    distance_mm = float(head_pose.translation_vector[2, 0])
    straight = abs(yaw_deg) <= FRONT_YAW_PITCH_TOLERANCE_DEG and abs(pitch_deg) <= FRONT_YAW_PITCH_TOLERANCE_DEG
    status = "straight" if straight else "not quite straight"
    return f"yaw={yaw_deg:+.1f} deg  pitch={pitch_deg:+.1f} deg  distance={distance_mm:.0f}mm  ({status})", straight


# Shows the "face forward" instruction screen and waits for SPACE (or ESC to abort).
def wait_for_front_ready(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    instruction: str = FRONT_READY_INSTRUCTION,
) -> None:
    font, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 2, 60
    instruction_lines = _wrap_line(instruction, font, 0.9, thickness, screen_w - 2 * margin)

    while True:
        ok, frame = cam.read()
        head_pose = None
        face_status, in_frame, face_box = "no face detected -- move into the camera's view", False, None
        if ok and frame is not None:
            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)
            if result is not None:
                points_px = result.points_px()
                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                face_status, in_frame, face_box = _face_frame_status(points_px, result.image_width, result.image_height)

        canvas = _camera_background(frame if ok else None, screen_w, screen_h)

        line_height = 40
        start_y = screen_h // 2 - 100 - (line_height * (len(instruction_lines) - 1)) // 2
        _darken_band(canvas, start_y - 40, screen_h // 2 + 150)
        for i, line in enumerate(instruction_lines):
            (text_w, _), _ = cv2.getTextSize(line, font, 0.9, thickness)
            x = screen_w // 2 - text_w // 2
            cv2.putText(canvas, line, (x, start_y + i * line_height), font, 0.9, (0, 220, 255), thickness)

        face_color = (0, 220, 0) if in_frame else (0, 0, 255)
        (face_w, _), _ = cv2.getTextSize(face_status, font, 0.7, 2)
        cv2.putText(canvas, face_status, (screen_w // 2 - face_w // 2, screen_h // 2 + 30), font, 0.7, face_color, 2)

        readout, straight = _front_readout(head_pose)
        color = (0, 0, 255) if readout == "face not detected" else ((0, 220, 0) if straight else (0, 165, 255))
        (text_w, _), _ = cv2.getTextSize(readout, font, 0.7, 2)
        cv2.putText(canvas, readout, (screen_w // 2 - text_w // 2, screen_h // 2 + 65), font, 0.7, color, 2)

        hint = "Press SPACE when ready (ESC to abort)"
        (hint_w, _), _ = cv2.getTextSize(hint, font, 0.7, 2)
        cv2.putText(canvas, hint, (screen_w // 2 - hint_w // 2, screen_h // 2 + 110), font, 0.7, (150, 150, 150), 2)

        _present(canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:
            return




# Draws one calibration dot with the still-head instruction below it.
def draw_target(screen_w: int, screen_h: int, point_px: tuple[int, int], instruction: str = STILL_INSTRUCTION) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)

    font, font_scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 20
    lines = _wrap_line(instruction, font, font_scale, thickness, screen_w - 2 * margin)
    line_height = 26
    base_y = screen_h - 30
    start_y = base_y - line_height * (len(lines) - 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (margin, start_y + i * line_height), font, font_scale, (150, 150, 150), thickness)
    return frame


# Captures gaze samples while the user holds still on one dot.
def collect_samples_for_point(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    point_px: tuple[int, int],
    gaze_extractor: GazeVectorExtractor,
    instruction: str = STILL_INSTRUCTION,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frames_seen = 0
    no_face = 0
    blink_rejected = 0
    pose_rejected = 0

    _present(draw_target(screen_w, screen_h, point_px, instruction))
    cv2.waitKey(1)

    settle_deadline = time.time() + SETTLE_S
    while time.time() < settle_deadline:
        cam.read()
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
        if not is_gaze_usable(points_px, min_ear=CALIBRATION_MIN_EAR):
            blink_rejected += 1
            continue
        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
        if head_pose is None:
            pose_rejected += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaze_vector = gaze_extractor.extract(frame, gray, points_px)
        if gaze_vector is None:
            no_face += 1
            continue
        feature = build_feature_vector(
            gaze_vector,
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
        print(
            "       -- discarding this point's samples: too few to trust. A handful of frames that barely "
            "squeaked past the blink check tend to be half-closed-eye frames with a degraded iris reading "
            "(often reading as a near-neutral/centered offset regardless of the true target), and the local "
            "correction stage is sensitive enough to a few such mislabeled samples that keeping them risks "
            "biasing nearby live predictions (e.g. true center gaze getting pulled toward this point's target)."
        )
        return [], []

    return features, targets


# Runs collect_samples_for_point() over a list of dot positions.
def run_sequence(
    cam,
    backend,
    pose_estimator,
    screen_w,
    screen_h,
    fractions,
    gaze_extractor: GazeVectorExtractor,
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
            cam, backend, pose_estimator, screen_w, screen_h, point_px, gaze_extractor, instruction
        )
        print(f"    captured {len(features)} samples")
        all_features.extend(features)
        all_targets.extend(targets)
        all_groups.extend([group_offset + i] * len(features))

    if not all_features:
        return np.empty((0, FEATURE_DIM)), np.empty((0, 2)), np.empty((0,), dtype=int)
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


# Draws the moving dot for the trajectory protocol, with a progress readout.
def _draw_trajectory_frame(
    screen_w: int, screen_h: int, point_px: tuple[int, int], progress: float
) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{min(progress, 1.0) * 100:.0f}%", (20, 40), font, 0.7, (150, 150, 150), 2)
    (iw, _), _ = cv2.getTextSize(STILL_INSTRUCTION, font, 0.6, 1)
    cv2.putText(frame, STILL_INSTRUCTION, (screen_w // 2 - iw // 2, screen_h - 30), font, 0.6, (150, 150, 150), 1)
    return frame


# Builds a shorter zig-zag path used for the pose-variant trajectory passes.
def _short_trajectory_waypoints() -> list[tuple[float, float]]:
    rows = (0.2, 0.5, 0.8)
    cols = np.linspace(0.1, 0.9, 6)
    waypoints: list[tuple[float, float]] = []
    for i, y in enumerate(rows):
        xs = cols if i % 2 == 0 else cols[::-1]
        waypoints.extend((float(x), float(y)) for x in xs)
    return waypoints


# Captures gaze samples continuously while a dot sweeps a path across the screen.
def run_trajectory_capture(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
    duration_s: float = TRAJECTORY_DURATION_S,
    lag_ms: float = TRAJECTORY_LAG_MS,
    turn_discard_ms: float = TRAJECTORY_TURN_DISCARD_MS,
    anchor_settle_ms: float = ANCHOR_SETTLE_MS,
    group_offset: int = 0,
    full_path: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if full_path:
        path_fracs = build_waypoints()
        keyframes = build_keyframes(total_duration_ms=duration_s * 1000.0)
    else:
        path_fracs = _short_trajectory_waypoints()
        keyframes = build_keyframes(total_duration_ms=duration_s * 1000.0, waypoints=path_fracs, n_anchors=0)
    path_px = [(fx * screen_w, fy * screen_h) for fx, fy in path_fracs]

    vel = measure_velocity(keyframes)
    over = vel.segments_over(30.0).sum()
    print(
        f"    target speed: mean={vel.mean_deg_s:.1f} deg/s, p90={vel.p90_deg_s:.1f} deg/s, "
        f"max={vel.max_deg_s:.1f} deg/s ({over} segment(s) over the ~30 deg/s comfort ceiling)"
    )

    start_px = (int(path_px[0][0]), int(path_px[0][1]))
    settle_deadline = time.time() + TRAJECTORY_SETTLE_S
    while time.time() < settle_deadline:
        cam.read()
        _present(_draw_trajectory_frame(screen_w, screen_h, start_px, 0.0))
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    capture_t_ms: list[float] = []
    features: list[np.ndarray] = []
    no_face = blink_rejected = pose_rejected = 0

    start = time.time()
    duration_ms = keyframes.duration_ms
    while True:
        now_ms = (time.time() - start) * 1000.0
        if now_ms >= duration_ms:
            break
        x_frac, y_frac = keyframes.target_at(now_ms)
        point_px = (
            int(np.clip(x_frac, 0.0, 1.0) * screen_w),
            int(np.clip(y_frac, 0.0, 1.0) * screen_h),
        )
        _present(_draw_trajectory_frame(screen_w, screen_h, point_px, now_ms / duration_ms))
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

        ok, frame = cam.read()
        if not ok or frame is None:
            no_face += 1
            continue
        timestamp_ms = cam.frame_timestamp_ms()
        result = backend.detect(frame, timestamp_ms)
        if result is None:
            no_face += 1
            continue
        points_px = result.points_px()
        if not is_gaze_usable(points_px, min_ear=CALIBRATION_MIN_EAR):
            blink_rejected += 1
            continue
        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
        if head_pose is None:
            pose_rejected += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaze_vector = gaze_extractor.extract(frame, gray, points_px)
        sample = None if gaze_vector is None else build_feature_vector(
            gaze_vector,
            (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg),
            tuple(head_pose.translation_vector.flatten()),
        )
        if sample is None:
            no_face += 1
            continue
        capture_t_ms.append(now_ms)
        features.append(sample)

    print(
        f"    ({no_face} no-face, {blink_rejected} blink, {pose_rejected} pose-reject frames "
        f"skipped; {len(features)} raw frames captured)"
    )

    if not features:
        return np.empty((0, FEATURE_DIM)), np.empty((0, 2)), np.empty((0,), dtype=int)

    capture_t_ms_arr = np.array(capture_t_ms, dtype=np.float64)
    features_arr = np.array(features)

    label_t_ms = np.clip(capture_t_ms_arr - lag_ms, 0.0, duration_ms)
    targets_frac = keyframes.target_at(label_t_ms)
    targets_px = targets_frac * np.array([screen_w, screen_h], dtype=np.float64)

    turn_discard = turn_discard_mask(label_t_ms, keyframes.turn_t_ms, turn_discard_ms)
    settle_discard = turn_discard_mask(label_t_ms, keyframes.pause_arrival_t_ms, anchor_settle_ms)
    kept = ~(turn_discard | settle_discard)
    print(
        f"    discarding {int(turn_discard.sum())} frames near a direction change, "
        f"{int(settle_discard.sum())} near an anchor arrival -- {int(kept.sum())}/{len(kept)} kept"
    )

    features_arr = features_arr[kept]
    targets_px = targets_px[kept]
    capture_t_ms_arr = capture_t_ms_arr[kept]

    groups = group_offset + (capture_t_ms_arr // TRAJECTORY_GROUP_WINDOW_MS).astype(np.int64)
    return features_arr, targets_px, groups


# Returns the next unused group id, continuing after the given groups array.
def _next_group_id(current: int, groups: np.ndarray) -> int:
    if len(groups) == 0:
        return current
    return int(groups.max()) + 1




# Runs the dot-grid training pass: the main 4x4 grid, then each head-pose variant.
def _run_grid_training(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wait_for_front_ready(cam, backend, pose_estimator, screen_w, screen_h)
    print("Keep your head still & forward -- move only your eyes to each dot.")
    print("Training points:")
    all_features, all_targets, all_groups = run_sequence(
        cam, backend, pose_estimator, screen_w, screen_h, TRAIN_GRID_FRACTIONS, gaze_extractor, STILL_INSTRUCTION
    )
    next_group_id = len(TRAIN_GRID_FRACTIONS)

    print("\nHead-pose variation pass -- this teaches the model to keep tracking")
    print("correctly after you shift position, instead of only ever seeing you")
    print("sit still and forward.")
    for variant in POSE_VARIANTS:
        wait_for_pose_ready(cam, backend, pose_estimator, screen_w, screen_h, variant)
        print(f"  Variant: {variant.label}")
        features, targets, groups = run_sequence(
            cam,
            backend,
            pose_estimator,
            screen_w,
            screen_h,
            POSE_VARIATION_FRACTIONS,
            gaze_extractor,
            variant.instruction,
            group_offset=next_group_id,
        )
        next_group_id += len(POSE_VARIATION_FRACTIONS)
        if features.shape[0] == 0:
            print(
                f"    !! captured nothing usable for '{variant.label}' -- this variant contributes "
                "no data to the model. The pose was likely too extreme (face lost by the "
                "webcam) or held too briefly. Consider re-running with a gentler version of "
                "this pose."
            )
        all_features = np.concatenate([all_features, features])
        all_targets = np.concatenate([all_targets, targets])
        all_groups = np.concatenate([all_groups, groups])

    return all_features, all_targets, all_groups


# Runs the held-out dot-grid validation pass.
def _run_grid_validation(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    print("Validation points:")
    wait_for_front_ready(
        cam,
        backend,
        pose_estimator,
        screen_w,
        screen_h,
        instruction="Return your head to a normal, centered position -- keep tracking dots with your eyes only.",
    )
    val_features, val_targets, _val_groups = run_sequence(
        cam, backend, pose_estimator, screen_w, screen_h, VALIDATION_FRACTIONS, gaze_extractor
    )
    return val_features, val_targets


# Runs the trajectory training pass: the main sweep, then each head-pose variant.
def _run_trajectory_training(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
    lag_ms: float = TRAJECTORY_LAG_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wait_for_front_ready(cam, backend, pose_estimator, screen_w, screen_h)
    print(f"Main pass: a dot will sweep the whole screen on its own for ~{TRAJECTORY_DURATION_S:.0f}s -- keep your head still, just follow it with your eyes.")
    all_features, all_targets, all_groups = run_trajectory_capture(
        cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor,
        duration_s=TRAJECTORY_DURATION_S, lag_ms=lag_ms,
    )
    next_group_id = _next_group_id(0, all_groups)

    print("\nHead-pose variation pass -- this teaches the model to keep tracking")
    print("correctly after you shift position, instead of only ever seeing you")
    print("sit still and forward.")
    for variant in POSE_VARIANTS:
        wait_for_pose_ready(cam, backend, pose_estimator, screen_w, screen_h, variant)
        print(f"  Variant: {variant.label}")
        features, targets, groups = run_trajectory_capture(
            cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor,
            duration_s=TRAJECTORY_POSE_VARIANT_DURATION_S, group_offset=next_group_id, full_path=False,
            lag_ms=lag_ms,
        )
        next_group_id = _next_group_id(next_group_id, groups)
        if features.shape[0] == 0:
            print(
                f"    !! captured nothing usable for '{variant.label}' -- this variant contributes "
                "no data to the model. The pose was likely too extreme (face lost by the "
                "webcam) or held too briefly. Consider re-running with a gentler version of "
                "this pose."
            )
        all_features = np.concatenate([all_features, features])
        all_targets = np.concatenate([all_targets, targets])
        all_groups = np.concatenate([all_groups, groups])

    return all_features, all_targets, all_groups


# Runs the held-out trajectory validation pass.
def _run_trajectory_validation(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
    next_group_id: int = 0,
    lag_ms: float = TRAJECTORY_LAG_MS,
) -> tuple[np.ndarray, np.ndarray]:
    wait_for_front_ready(
        cam,
        backend,
        pose_estimator,
        screen_w,
        screen_h,
        instruction="Return your head to a normal, centered position -- keep following the dot with your eyes only.",
    )
    val_features, val_targets, _val_groups = run_trajectory_capture(
        cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor,
        duration_s=TRAJECTORY_VALIDATION_DURATION_S, group_offset=next_group_id, full_path=False,
        lag_ms=lag_ms,
    )
    return val_features, val_targets


# Parses args, runs the chosen calibration protocol, fits and saves the best calibrator.
def main() -> None:
    parser = argparse.ArgumentParser(description="doomscroll gaze calibration")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Also fit against this session pooled with recent stored sessions (see session_store.py "
        "for why this is opt-in rather than the default), and keep whichever fit actually validates "
        "better. Prints both the session-only and pooled validation error so you can compare.",
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
    parser.add_argument(
        "--protocol",
        choices=["dots", "trajectory"],
        default="trajectory",
        help="Capture mechanic to use (default: trajectory) -- a single continuously-moving "
        "dot that sweeps the whole screen in a programmed path (anchors, then a raster sweep) "
        f"for about {TRAJECTORY_DURATION_S:.0f}s, fitting the same GazeCalibrator directly on "
        "the continuously-captured frames instead of a handful of per-dot samples -- see "
        "run_trajectory_capture() and the module docstring. 'dots' instead uses the original "
        "4x4 dot-grid protocol (run_sequence()/collect_samples_for_point()).",
    )
    parser.add_argument(
        "--lag-ms",
        type=float,
        default=TRAJECTORY_LAG_MS,
        help=f"--protocol trajectory only: how far back (ms) to shift each captured frame's "
        f"target label, to correct for smooth-pursuit lag (default: {TRAJECTORY_LAG_MS:.0f}). "
        "If live predictions sit systematically off in one direction after calibrating with "
        "--protocol trajectory (e.g. always below where you're actually looking), that's this "
        "number being off for you specifically, not a bug: too LOW a value under-corrects and "
        "biases predictions toward wherever the dot tends to end up shortly after your eyes "
        "really were (e.g. further down mid-sweep); too HIGH over-corrects the other way. Try "
        "nudging this up or down (e.g. +-50ms) and recalibrating to see which direction helps.",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=None,
        help="Degree of the polynomial feature expansion GazeCalibrator fits. Defaults to 2 "
        "(quadratic) for --protocol trajectory and 1 (linear) for --protocol dots -- see "
        "GazeCalibrator's own docstring for why: a degree-2 fit captures real "
        "curvature in the eye-rotation-to-screen mapping, but on the dot-grid protocol's ~16 "
        "independent calibration points it was observed to extrapolate wildly, whereas "
        "--protocol trajectory's continuous capture produces far more independent groups "
        "(dozens, not ~16) -- exactly the 'substantially more independent points' condition "
        "that justifies it there. Pass this explicitly to override either default, e.g. "
        "--poly-degree 1 if trajectory's degree-2 default ever looks worse on your setup.",
    )
    args = parser.parse_args()

    if args.poly_degree is None:
        args.poly_degree = 2 if args.protocol == "trajectory" else 1

    if args.clear_history:
        removed = clear_sessions()
        print(f"Cleared {removed} stored calibration session(s).")
        return

    if args.list_history:
        sessions = list_sessions()
        if not sessions:
            print("No stored calibration sessions.")
        for s in sessions:
            print(
                f"  {s.label}  {len(s.features):5d} samples  screen={s.screen_size[0]}x{s.screen_size[1]}  "
                f"protocol={s.protocol}"
            )
        return

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    gaze_extractor = make_gaze_extractor()
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    global _display_size
    _display_size = _detect_display_size(screen_w, screen_h)
    if _display_size != (screen_w, screen_h):
        print(f"Note: fullscreen window renders at {_display_size[0]}x{_display_size[1]} "
              f"vs logical {screen_w}x{screen_h}; scaling frames (and click readings) to agree with it.")

    try:
        with WebcamSource(0) as cam:
            print("Calibration starting. ESC to abort.")

            if args.protocol == "trajectory":
                all_features, all_targets, all_groups = _run_trajectory_training(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor, lag_ms=args.lag_ms
                )
                next_group_id = _next_group_id(0, all_groups)
            else:
                all_features, all_targets, all_groups = _run_grid_training(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor
                )
                next_group_id = 0

            save_session(
                all_features, all_targets, all_groups, (screen_w, screen_h),
                protocol=args.protocol,
            )
            print(f"\nSession saved ({len(all_features)} samples).")

            # Fits a calibrator on the given data and prints its training/validation stats.
            def fit_and_report(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, label: str) -> tuple[GazeCalibrator, float | None]:
                calibrator = GazeCalibrator(poly_degree=args.poly_degree)
                calibrator.fit(features, targets, groups=groups)
                n_points = len(np.unique(groups))
                print(f"\n[{label}] fitted on {len(features)} samples across {n_points} time-window groups.")
                print(f"[{label}] RidgeCV selected alpha (x, y): {calibrator.selected_alpha}")

                train_preds = np.array([calibrator.predict(f) for f in features])
                target_range = targets.max(axis=0) - targets.min(axis=0)
                pred_range = train_preds.max(axis=0) - train_preds.min(axis=0)
                gain = pred_range / np.where(target_range == 0, 1, target_range)
                print(f"[{label}] gain (predicted range / actual range), x,y: {gain} -- 1.0 is ideal, well below 1.0 means compressed")

                err = None
                if len(val_features) > 0:
                    err = calibrator.validate(val_features, val_targets)
                    print(
                        f"[{label}] validation mean error: {err:.1f}px "
                        f"({100 * err / screen_diag:.1f}% of screen diagonal)"
                    )
                return calibrator, err

            if args.protocol == "trajectory":
                val_features, val_targets = _run_trajectory_validation(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor, next_group_id, lag_ms=args.lag_ms
                )
            else:
                val_features, val_targets = _run_grid_validation(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor
                )
            screen_diag = (screen_w**2 + screen_h**2) ** 0.5

            session_calibrator, session_err = fit_and_report(
                all_features, all_targets, all_groups, "this session only"
            )
            best_calibrator, best_err, best_label = session_calibrator, session_err, "this session only"

            if args.pool:
                pooled_features, pooled_targets, pooled_groups, used = pool_sessions(
                    list_sessions(), (screen_w, screen_h), protocol=args.protocol
                )
                if len(used) <= 1:
                    print(
                        "\nNothing to pool yet (this is the only stored session at this screen size)."
                    )
                else:
                    print(f"\nPooling {len(used)} sessions: " + ", ".join(s.label for s in used))
                    pooled_calibrator, pooled_err = fit_and_report(
                        pooled_features, pooled_targets, pooled_groups, "pooled"
                    )
                    if pooled_err is not None and session_err is not None and pooled_err < session_err:
                        best_calibrator, best_err, best_label = pooled_calibrator, pooled_err, "pooled"
                    elif pooled_err is not None and session_err is not None:
                        print(f"\nPooled fit did not improve on this session alone -- keeping '{best_label}'.")

            if best_err is not None and best_err / screen_diag > 0.10:
                print("\nAccuracy looks poor -- consider recalibrating with better/more even lighting.")

            if len(val_features) == 0:
                print("\nNo validation samples captured (face not detected during validation).")

            best_calibrator.save()
            print(f"Calibration profile saved to {DEFAULT_PROFILE_PATH} (using: {best_label}).")
            if best_err is not None:
                print(
                    f"CALIBRATION_SUMMARY: err_px={best_err:.1f} "
                    f"err_pct={100 * best_err / screen_diag:.1f} label={best_label}"
                )
            else:
                print(f"CALIBRATION_SUMMARY: err_px=none err_pct=none label={best_label}")
    except KeyboardInterrupt:
        print("Calibration aborted.")
        print("CALIBRATION_CANCELLED")
        sys.exit(1)
    finally:
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
