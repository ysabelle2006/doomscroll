# Extracts the CNN/ridge-model's per-frame feature vector (gaze, eyelid, head pose).

from __future__ import annotations

from dataclasses import dataclass

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

# Names of each value in the feature vector, in order.
FEATURE_NAMES = (
    "gaze_mean_x", "gaze_mean_y", "gaze_diff_x", "gaze_diff_y",
    "lid_ratio_y",
    "yaw", "pitch", "roll",
    "head_tx", "head_ty", "head_tz",
)
FEATURE_DIM = len(FEATURE_NAMES)
YAW_INDEX = FEATURE_NAMES.index("yaw")


# Returns how far the iris sits between the eyelids (0=top, 1=bottom).
def _lid_ratio(points_px: np.ndarray, top: int, bottom: int, iris: int) -> float:
    top_y, bottom_y, iris_y = points_px[top][1], points_px[bottom][1], points_px[iris][1]
    span = bottom_y - top_y
    return float((iris_y - top_y) / span) if abs(span) > 1e-6 else 0.5


# Builds one frame's full feature vector from landmarks and head pose.
def extract_features(
    points_px: np.ndarray, head_pose: HeadPose, frame_gray: np.ndarray | None = None
) -> np.ndarray:
    gaze = compute_gaze_feature(points_px, frame_gray=frame_gray).vector
    lid = 0.5 * (
        _lid_ratio(points_px, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER)
        + _lid_ratio(points_px, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER)
    )
    t = head_pose.translation_vector.flatten()
    return np.concatenate(
        [gaze, [lid], [head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg], t]
    ).astype(np.float64)


# One frame's extraction result: features (if usable) plus the reason if not.
@dataclass
class FrameResult:

    features: np.ndarray | None
    reason: str
    points_px: np.ndarray | None = None
    pose: HeadPose | None = None
    gray: np.ndarray | None = None


# Runs landmark detection, pose estimation, and feature extraction for a sequence of frames.
class FrameFeaturizer:

    def __init__(self, backend=None, pose_estimator=None):
        from doomscroll.gaze.head_pose import HeadPoseEstimator
        from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

        self._backend = backend or MediaPipeLandmarkBackend()
        self._pose = pose_estimator or HeadPoseEstimator()

    # Resets the head-pose estimator's previous-frame state (e.g. between sessions).
    def reset(self) -> None:
        self._pose.reset()

    # Returns just (features, reason) for one frame; see process_full for the full result.
    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, str]:
        r = self.process_full(frame_bgr, timestamp_ms)
        return r.features, r.reason

    # Runs the full per-frame pipeline: landmarks, pose, blink check, then feature extraction.
    def process_full(self, frame_bgr: np.ndarray, timestamp_ms: int) -> "FrameResult":
        import cv2

        result = self._backend.detect(frame_bgr, timestamp_ms)
        if result is None:
            self._pose.reset()
            return FrameResult(None, "no_face")
        points_px = result.points_px()
        pose = self._pose.estimate(points_px, result.image_width, result.image_height)
        if pose is None:
            return FrameResult(None, "no_pose", points_px)
        if not is_gaze_usable(points_px):
            return FrameResult(None, "blink", points_px, pose)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return FrameResult(extract_features(points_px, pose, gray), "ok", points_px, pose, gray)

    # Releases the landmark backend's resources.
    def close(self) -> None:
        self._backend.close()
