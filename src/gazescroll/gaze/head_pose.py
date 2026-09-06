"""Head pose (yaw/pitch/roll) via OpenCV solvePnP on a generic 3D face model.

Independent of the gaze-vector estimate; mature and low-noise compared to
appearance-based approaches, per the plan's design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gazescroll.landmarks.mediapipe_backend import (
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    NOSE_TIP,
    RIGHT_EYE_RIGHT_CORNER,
)

# Generic 3D face model points (mm, arbitrary origin at nose tip), the
# standard 6-point correspondence set used for webcam head-pose PnP.
_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # left eye left corner
        (225.0, 170.0, -135.0),  # right eye right corner
        (-150.0, -150.0, -125.0),  # mouth left corner
        (150.0, -150.0, -125.0),  # mouth right corner
    ],
    dtype=np.float64,
)

_LANDMARK_INDICES = [
    NOSE_TIP,
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    RIGHT_EYE_RIGHT_CORNER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
]

_DEFAULT_RVEC = np.zeros((3, 1))
_DEFAULT_TVEC = np.array([[0.0], [0.0], [600.0]])


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation_vector: np.ndarray  # Rodrigues (axis-angle) form, as returned by solvePnP
    translation_vector: np.ndarray  # camera-frame position of the model's origin (nose tip)


def _camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """Pinhole camera intrinsic matrix K, mapping a 3D camera-frame point
    (X, Y, Z) to pixel coordinates via u = fx*X/Z + cx, v = fy*Y/Z + cy.
    We don't have a real calibrated camera, so focal length is approximated
    as the image width in pixels -- a common rough stand-in when the actual
    lens/sensor parameters are unknown -- and the principal point (cx, cy)
    is assumed to be the image center."""
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
    # The model's Y-up convention vs. image Y-down means decomposeProjectionMatrix
    # can return pitch wrapped near +-180 deg for a frontal face instead of near 0;
    # unwrap it back into an intuitive small-angle range.
    if pitch < -90:
        return -(180 + pitch)
    if pitch > 90:
        return 180 - pitch
    return pitch


class HeadPoseEstimator:
    """Stateful PnP solve: seeds each frame with the previous frame's pose
    (or a sane frontal default on the first frame / after a lost track).

    With only 6 near-planar correspondence points, an unseeded single-frame
    solvePnP call can converge to an "ambiguous flipped" local minimum --
    low reprojection error, but physically wrong (symptom: roll reading near
    +-180 deg for a normally-seated user). Seeding with a temporally-close
    prior biases the iterative solver toward the correct solution instead.
    """

    def __init__(self):
        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_rvec = None
        self._prev_tvec = None

    def estimate(self, landmarks_px: np.ndarray, image_width: int, image_height: int) -> HeadPose | None:
        """landmarks_px: (N, 2) pixel-coordinate landmarks indexed as the full
        478-point mediapipe mesh (only the 6 PnP correspondence points are used)."""
        image_points = landmarks_px[_LANDMARK_INDICES].astype(np.float64)
        camera_matrix = _camera_matrix(image_width, image_height)
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        # Seed the solver with the previous frame's pose rather than letting
        # it start cold each time -- see the class docstring for why this
        # matters (it's what keeps the iterative solve out of the ambiguous
        # "flipped" local minimum). First call / after a lost track falls
        # back to a sane default: facing the camera, ~600mm away.
        rvec_guess = self._prev_rvec.copy() if self._prev_rvec is not None else _DEFAULT_RVEC.copy()
        tvec_guess = self._prev_tvec.copy() if self._prev_tvec is not None else _DEFAULT_TVEC.copy()

        # Iteratively refines (rvec, tvec) via Levenberg-Marquardt to minimize
        # total reprojection error: project each of the 6 model points through
        # (rvec, tvec, camera_matrix) and compare to where it was actually
        # observed (image_points), starting from the guess above.
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

        # rotation_vector is Rodrigues (axis-angle) form; expand it to a full
        # 3x3 rotation matrix, then decompose that into Euler angles. Both
        # steps are standard OpenCV utilities -- the actual head-pose math is
        # already done by solvePnP above, this just re-expresses the result
        # in a human-readable yaw/pitch/roll form.
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat([rotation_matrix, translation_vector])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        pitch, yaw, roll = (float(a) for a in euler_angles.flatten())
        pitch = _unwrap_pitch(pitch)

        # Plausibility guard: a correctly tracked, forward-facing head should
        # never read anywhere near +-180 deg roll, and the face must be in
        # front of the camera. If the solve still lands on a flipped/invalid
        # solution, drop the temporal seed and report failure -- downstream
        # logic (attention gate) treats this as "no confident pose", which is
        # the correct conservative behavior, rather than acting on a wild angle.
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
