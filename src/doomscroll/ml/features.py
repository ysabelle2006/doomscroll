"""THE single definition of "what the gaze model sees".

Both dataset building (offline, from recorded video) and live inference call
`extract_features`, so training and runtime cannot drift apart. If you add or
reorder a feature, change FEATURE_NAMES and extract_features together; saved
models store FEATURE_NAMES and refuse to load if they no longer match.

Compared to the old 7-number vector (calibration.build_feature_vector) this
adds what the head-angle problem was missing:
  - head TRANSLATION (where the face is in the frame, and how far away): the
    same eye rotation maps to a different screen point when you lean/shift.
  - eyelid-relative vertical iris position: vertical gaze is poorly captured
    by corner-normalized offsets because the eyelids move with gaze.
"""
from __future__ import annotations

import numpy as np

from doomscroll.gaze.gaze_vector import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_TOP,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_TOP,
    RIGHT_IRIS_CENTER,
    compute_gaze_feature,
    is_gaze_usable,
)
from doomscroll.gaze.head_pose import HeadPose

FEATURE_NAMES = (
    "gaze_mean_x", "gaze_mean_y", "gaze_diff_x", "gaze_diff_y",  # iris in eye, corner-normalized
    "lid_ratio_y",                                              # iris between top/bottom eyelid, 0..1
    "yaw", "pitch", "roll",                                     # head rotation, degrees
    "head_tx", "head_ty", "head_tz",                            # head position from solvePnP (model units ~mm)
)
FEATURE_DIM = len(FEATURE_NAMES)
YAW_INDEX = FEATURE_NAMES.index("yaw")


def _lid_ratio(points_px: np.ndarray, top: int, bottom: int, iris: int) -> float:
    top_y, bottom_y, iris_y = points_px[top][1], points_px[bottom][1], points_px[iris][1]
    span = bottom_y - top_y
    return float((iris_y - top_y) / span) if abs(span) > 1e-6 else 0.5


def extract_features(
    points_px: np.ndarray, head_pose: HeadPose, frame_gray: np.ndarray | None = None
) -> np.ndarray:
    """points_px: (478, 2) landmarks in pixels. Returns (FEATURE_DIM,) float64.
    Callers should check is_gaze_usable() first (blink) -- see FrameFeaturizer."""
    gaze = compute_gaze_feature(points_px, frame_gray=frame_gray).vector
    lid = 0.5 * (
        _lid_ratio(points_px, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER)
        + _lid_ratio(points_px, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER)
    )
    t = head_pose.translation_vector.flatten()
    return np.concatenate(
        [gaze, [lid], [head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg], t]
    ).astype(np.float64)


class FrameFeaturizer:
    """frame (BGR) -> feature vector, or None plus a reason string.
    Stateful (head-pose solver is seeded by the previous frame), so feed it
    frames in order and call reset() between videos."""

    def __init__(self, backend=None, pose_estimator=None):
        # Imported lazily so code that only needs the pure functions above
        # (tests, training) doesn't need a working mediapipe install.
        from doomscroll.gaze.head_pose import HeadPoseEstimator
        from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

        self._backend = backend or MediaPipeLandmarkBackend()
        self._pose = pose_estimator or HeadPoseEstimator()

    def reset(self) -> None:
        self._pose.reset()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, str]:
        import cv2

        result = self._backend.detect(frame_bgr, timestamp_ms)
        if result is None:
            self._pose.reset()
            return None, "no_face"
        points_px = result.points_px()
        pose = self._pose.estimate(points_px, result.image_width, result.image_height)
        if pose is None:
            return None, "no_pose"
        if not is_gaze_usable(points_px):
            return None, "blink"
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return extract_features(points_px, pose, gray), "ok"

    def close(self) -> None:
        self._backend.close()
