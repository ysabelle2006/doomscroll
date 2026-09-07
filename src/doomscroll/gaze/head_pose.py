"""Calculates 3D head orientation (yaw, pitch, roll) from 2D facial landmarks using OpenCV solvePnP."""
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

# Standard 3D face model points in millimeters, centered at the nose tip (0, 0, 0)
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),   # Right mouth corner
    ],
    dtype=np.float64,
)

# Indices of the 6 landmarks corresponding to the 3D model points
_LANDMARK_INDICES = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    RIGHT_EYE_RIGHT_CORNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
]

# Default starting guesses: looking straight ahead, 60cm from the camera
_DEFAULT_RVEC = np.zeros((3, 1))
_DEFAULT_TVEC = np.array([[0.0], [0.0], [600.0]])


@dataclass
class HeadPose:
    """Stores calculated head orientation angles and transformation vectors."""
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation_vector: np.ndarray  # Rodrigues (axis-angle) form, as returned by solvePnP
    translation_vector: np.ndarray  # camera-frame position of the model's origin (nose tip)


def _camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """Estimates camera internal focal length and optical center based on frame dimensions."""
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


def _unwrap_pitch(pitch: float) -> float:
    """Corrects pitch angle wrapping near +/-180 degrees back to near 0 for a frontal face."""
    if pitch < -90:
        return -(180 + pitch)
    if pitch > 90:
        return 180 - pitch
    return pitch


class HeadPoseEstimator:
    """Estimates head pose frame-by-frame, using previous estimates as a starting guess to stay stable."""

    def __init__(self):
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

    def reset(self) -> None:
        """Clears stored pose history when tracking drops."""
        self._prev_rvec = None
        self._prev_tvec = None

    def estimate(self, landmarks_px: np.ndarray, image_width: int, image_height: int) -> HeadPose | None:
        # Extract only the 6 landmarks used by our 3D face model
        image_points = landmarks_px[_LANDMARK_INDICES].astype(np.float64)
        camera_matrix = _camera_matrix(image_width, image_height)
        dist_coeffs = np.zeros((4, 1))  # Assume minimal lens distortion

        # Seed solvePnP with previous frame's pose to prevent flipped solutions
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

        # Convert rotation vector to 3x3 matrix and extract Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat([rotation_matrix, translation_vector])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        pitch, yaw, roll = (float(a) for a in euler_angles.flatten())
        pitch = _unwrap_pitch(pitch)

        # Sanity check: reject poses with extreme head roll (>90 deg) or behind the camera (Z <= 0)
        if abs(roll) > 90 or translation_vector[2, 0] <= 0:
            self.reset()
            return None

        # Save current pose as the starting guess for the next frame
        self._prev_rvec = rotation_vector
        self._prev_tvec = translation_vector

        return HeadPose(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )
