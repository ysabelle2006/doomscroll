# Estimates 3D head pose (yaw/pitch/roll) from face landmarks via solvePnP.

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from doomscroll.landmarks.mediapipe_backend import (
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    NOSE_TIP,
    RIGHT_EYE_RIGHT_CORNER,
)

# Generic 3D face model points (nose, chin, eye corners, mouth corners), in mm.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

# Landmark indices matching the 3D model points above, in the same order.
_LANDMARK_INDICES = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    RIGHT_EYE_RIGHT_CORNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
]

# Fallback initial guess for solvePnP when there's no previous frame's pose.
_DEFAULT_RVEC = np.zeros((3, 1))
_DEFAULT_TVEC = np.array([[0.0], [0.0], [400.0]])


# One frame's estimated head rotation (yaw/pitch/roll) and position.
@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation_vector: np.ndarray
    translation_vector: np.ndarray


# Builds an approximate camera intrinsics matrix from the image size.
def _camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    return np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


# Unwraps pitch across the +-90 degree seam into a continuous range.
def _unwrap_pitch(pitch: float) -> float:
    if pitch < -90:
        return -(180 + pitch)
    if pitch > 90:
        return 180 - pitch
    return pitch


# Estimates head pose frame-by-frame, using the previous pose as solvePnP's initial guess.
class HeadPoseEstimator:

    def __init__(self):
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

    # Clears the previous-pose guess, e.g. after the face is lost.
    def reset(self) -> None:
        self._prev_rvec = None
        self._prev_tvec = None

    # Solves for this frame's head pose; returns None if the solve fails or looks invalid.
    def estimate(self, landmarks_px: np.ndarray, image_width: int, image_height: int) -> HeadPose | None:
        image_points = landmarks_px[_LANDMARK_INDICES].astype(np.float64)
        camera_matrix = _camera_matrix(image_width, image_height)
        dist_coeffs = np.zeros((4, 1))

        rvec_guess = self._prev_rvec.copy() if self._prev_rvec is not None else _DEFAULT_RVEC.copy()
        tvec_guess = self._prev_tvec.copy() if self._prev_tvec is not None else _DEFAULT_TVEC.copy()

        ok, rotation_vector, translation_vector = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=rvec_guess,
            tvec=tvec_guess,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.reset()
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat([rotation_matrix, translation_vector])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        pitch, yaw, roll = (float(a) for a in euler_angles.flatten())
        pitch = _unwrap_pitch(pitch)

        if abs(roll) > 90 or translation_vector[2, 0] <= 0:
            self.reset()
            return None

        self._prev_rvec = rotation_vector
        self._prev_tvec = translation_vector

        return HeadPose(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )
