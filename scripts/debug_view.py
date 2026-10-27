"""Phase 1 validation: webcam capture + face-mesh landmarks + head pose,
rendered live. Confirms landmark/pose quality before building calibration or
content-awareness on top.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.gaze.head_pose import HeadPoseEstimator
from doomscroll.landmarks.mediapipe_backend import (
    LEFT_IRIS,
    RIGHT_IRIS,
    MediaPipeLandmarkBackend,
)


def draw_axes(frame: np.ndarray, origin: tuple[int, int], head_pose) -> None:
    rotation_matrix, _ = cv2.Rodrigues(head_pose.rotation_vector)
    axis_len = 80
    axes_3d = np.float64([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x=red, y=green, z=blue
    for axis, color in zip(axes_3d, colors):
        # Project axis endpoint by rotating it and adding a fixed offset in
        # front of the face, then draw relative to the origin point on screen.
        rotated = rotation_matrix @ axis
        end = (int(origin[0] + rotated[0]), int(origin[1] - rotated[1]))
        cv2.line(frame, origin, end, color, 2)


def main() -> None:
    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    with WebcamSource(0) as cam:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("Failed to read frame from webcam.")
                break

            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)

            if result is not None:
                points_px = result.points_px()

                for x, y in points_px.astype(int):
                    cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)

                for idx in LEFT_IRIS + RIGHT_IRIS:
                    x, y = points_px[idx].astype(int)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                if head_pose is not None:
                    nose_x, nose_y = points_px[1].astype(int)
                    draw_axes(frame, (nose_x, nose_y), head_pose)
                    text = f"yaw={head_pose.yaw_deg:.1f} pitch={head_pose.pitch_deg:.1f} roll={head_pose.roll_deg:.1f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Pose solve failed/rejected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                pose_estimator.reset()
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("doomscroll - Phase 1 debug view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    backend.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
