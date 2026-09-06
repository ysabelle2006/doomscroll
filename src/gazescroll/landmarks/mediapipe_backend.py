"""MediaPipe Face Landmarker backend: 478-point mesh incl. iris landmarks
(indices 468-477), run in VIDEO mode for temporal stability."""
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

from gazescroll.landmarks.base import FaceLandmarkResult, LandmarkBackend

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


class MediaPipeLandmarkBackend(LandmarkBackend):
    """Wraps MediaPipe's FaceLandmarker Tasks API behind the project's
    LandmarkBackend interface (see landmarks/base.py), so a fallback
    detector could later be swapped in without touching any downstream
    code that consumes FaceLandmarkResult."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH, num_faces: int = 1):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {model_path}. "
                "Download face_landmarker.task from the MediaPipe model hub."
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            # VIDEO (not IMAGE) mode lets MediaPipe use the previous frame's
            # detection as a prior for the current one internally, which is
            # both faster and more temporally stable than treating every
            # frame as an independent still image.
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            # Neither is used anywhere in this project -- skip computing them.
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarkResult | None:
        """Runs one frame through the detector. Returns None if no face was
        found (a legitimate, expected outcome -- callers treat it the same
        as "no confident tracking this tick", not an error)."""
        # mediapipe's VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        # MediaPipe expects RGB; OpenCV captures/decodes frames as BGR.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        # Only the first detected face is used (num_faces=1 by default
        # anyway) -- this project assumes a single user in frame.
        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        h, w = frame_bgr.shape[:2]
        return FaceLandmarkResult(points=points, image_width=w, image_height=h)

    def close(self) -> None:
        """Releases the underlying MediaPipe detector's native resources."""
        self._detector.close()


# Canonical MediaPipe FaceMesh landmark indices used elsewhere in the pipeline
# (head pose PnP correspondences, eye/iris regions for the gaze vector). These
# are fixed positions in MediaPipe's 478-point topology -- not detected or
# computed, just where these particular facial features always land in its
# output ordering. "Left"/"right" follow MediaPipe's own convention: the
# subject's anatomical left/right, not the image's left/right (so, for a
# front-facing camera, the subject's "left eye" appears on the right side of
# the captured frame).
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 263
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# Iris landmarks: index 0 of each range is the iris center, the remaining 4
# are boundary points around it (unused elsewhere in this project so far).
LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))
