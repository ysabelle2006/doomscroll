"""MediaPipe Face Landmarker setup to detect 478 face and iris points across video frames."""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from doomscroll.landmarks.base import FaceLandmarkResult, LandmarkBackend

# Default location of the pre-trained MediaPipe task model
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


class MediaPipeLandmarkBackend(LandmarkBackend):
    """Runs MediaPipe Face Landmarker in video mode to extract facial landmark coordinates."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH, num_faces: int = 1):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {model_path}. "
                "Download face_landmarker.task from the MediaPipe model hub."
            )
        
        # Configure MediaPipe to run on CPU in VIDEO mode for temporal smoothing
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarkResult | None:
        """Processes a video frame and returns the detected 3D landmark points."""
        # MediaPipe VIDEO mode requires timestamps to strictly increase
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # Convert OpenCV BGR image to RGB as required by MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        # Extract normalized (x, y, z) coordinates for the primary face
        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        h, w = frame_bgr.shape[:2]
        return FaceLandmarkResult(points=points, image_width=w, image_height=h)

    def close(self) -> None:
        """Closes the underlying detector and frees resources."""
        self._detector.close()


# Key landmark indices in the MediaPipe 478-point mesh
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 263
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# Indices for iris tracking (5 points per eye)
LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))
