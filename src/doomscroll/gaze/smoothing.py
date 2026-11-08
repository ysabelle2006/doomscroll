# Smooths predicted gaze points with a constant-velocity Kalman filter plus an optional EMA pass.

from __future__ import annotations

import cv2
import numpy as np

MAX_DT_S = 0.25


# Builds a constant-velocity Kalman filter over (x, y, vx, vy).
def make_kalman_filter(process_var: float = 20.0, measurement_var: float = 0.5) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_var
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_var
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePre = np.zeros((4, 1), np.float32)
    kf.statePost = np.zeros((4, 1), np.float32)
    return kf


# Kalman-filters a stream of 2D points, clamping large jumps (innovation) and rescaling noise by dt.
class KalmanSmoother:

    def __init__(self, process_var: float = 20.0, measurement_var: float = 0.5, max_innovation: float = 120.0):
        self._process_var = process_var
        self._measurement_var = measurement_var
        self._max_innovation = max_innovation
        self._kf = make_kalman_filter(process_var, measurement_var)
        self._initialized = False
        self._last_timestamp_ms: int | None = None

    # Updates the filter with a new point; returns the smoothed point.
    def update(self, point: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray:
        meas = np.array([[float(point[0])], [float(point[1])]], dtype=np.float32)

        if not self._initialized:
            self._kf.statePre[:2] = meas
            self._kf.statePost[:2] = meas
            self._initialized = True
            self._last_timestamp_ms = timestamp_ms
            return np.array([point[0], point[1]], dtype=np.float64)

        dt_s = 1.0
        if timestamp_ms is not None and self._last_timestamp_ms is not None:
            dt_s = min(max((timestamp_ms - self._last_timestamp_ms) / 1000.0, 1e-3), MAX_DT_S)
            self._kf.transitionMatrix[0, 2] = dt_s
            self._kf.transitionMatrix[1, 3] = dt_s
            self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * self._process_var * dt_s
        self._last_timestamp_ms = timestamp_ms

        predicted = self._kf.predict()

        if self._max_innovation is not None:
            innovation = meas[:, 0] - predicted[:2, 0]
            innovation_norm = float(np.linalg.norm(innovation))
            if innovation_norm > self._max_innovation:
                scale = self._max_innovation / innovation_norm
                meas = (predicted[:2] + (innovation * scale).reshape(2, 1)).astype(np.float32)

        self._kf.correct(meas)
        return np.array([self._kf.statePost[0, 0], self._kf.statePost[1, 0]], dtype=np.float64)

    # Resets the filter to an uninitialized state.
    def reset(self) -> None:
        self._kf = make_kalman_filter(self._process_var, self._measurement_var)
        self._initialized = False
        self._last_timestamp_ms = None


# KalmanSmoother plus an extra exponential-moving-average pass on top.
class GazePointSmoother(KalmanSmoother):

    def __init__(
        self,
        process_var: float = 20.0,
        measurement_var: float = 0.5,
        max_innovation: float = 120.0,
        ema_alpha: float = 0.5,
    ):
        super().__init__(process_var, measurement_var, max_innovation)
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0.0, 1.0], got {ema_alpha}")
        self._ema_alpha = ema_alpha
        self._ema_point: np.ndarray | None = None

    # Updates the Kalman filter, then blends the result into the running EMA.
    def update(self, point: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray:
        kalman_point = super().update(point, timestamp_ms)

        a = self._ema_alpha
        if a == 0.0:
            return kalman_point

        if self._ema_point is None:
            self._ema_point = kalman_point.copy()
        else:
            self._ema_point = a * self._ema_point + (1.0 - a) * kalman_point
        return self._ema_point

    # Resets both the Kalman filter and the EMA state.
    def reset(self) -> None:
        super().reset()
        self._ema_point = None
