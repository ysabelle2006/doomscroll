"""Kalman filtering for the mapped screen-space gaze point: a 4-state
constant-velocity model ([x, y, vx, vy], measuring [x, y]), with
innovation gating against single-frame outliers, optionally followed by
a light EMA pass. Process/measurement noise are tuned as an explicit
trust tradeoff, and the velocity state lets the filter anticipate smooth
motion while damping single-frame jumps.

KalmanSmoother.update() sets dt (the transition matrix's velocity
coupling) from the actual elapsed time between calls, since the capture
loop doesn't run at a fixed rate. dt is clamped to a maximum (MAX_DT_S),
since an unclamped dt during a processing hiccup (a blink, a slow
inference tick) would make the constant-velocity extrapolation
overconfident: position error from accumulated velocity noise grows
linearly with dt, while an unscaled processNoiseCov doesn't account for
the extra elapsed-time uncertainty.

update() returns the corrected (post-measurement) state rather than
predict()'s pre-correction output, since this project's dwell trigger
cares about stability far more than minimizing perceived latency: if the
velocity state is ever noisy, a pre-correction output would extrapolate
from that noise every frame before correction reins it back in.
"""
from __future__ import annotations

import cv2
import numpy as np

# Cap dt so a processing hiccup (a blink, a slow inference tick) can't make
# the constant-velocity extrapolation overconfident -- position uncertainty
# from accumulated velocity noise grows with dt, and clamping it bounds the
# worst case rather than letting one long gap between frames blow up the
# prediction.
MAX_DT_S = 0.25


def make_kalman_filter(process_var: float = 20.0, measurement_var: float = 0.5) -> cv2.KalmanFilter:
    """4-state constant-velocity Kalman filter: state [x, y, vx, vy], measuring [x, y].

    process_var: how much the filter trusts its own motion model (higher =
    follows measurements more readily, less lag, more residual jitter).
    measurement_var: how much it trusts each raw measurement (higher = more
    smoothing/lag). dt (the transition matrix's velocity coupling) is set
    per-call in KalmanSmoother.update() rather than fixed here.
    """
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


class KalmanSmoother:
    """Kalman-filtered gaze point with innovation gating.

    This project's dwell/pattern-based scroll trigger cares more about
    rejecting single-frame outliers (a blink-corrupted landmark, a brief
    head jerk) than about minimizing lag -- and naive Kalman correction is
    actually worse than no filtering for that: an outlier measurement gets
    absorbed into the *velocity* state too, causing the filter to overshoot
    even further on the *next* frame while it "catches up". Innovation
    gating (a standard robust-Kalman technique) clips how far a single
    measurement can pull the state before correct() is applied, bounding
    that damage while leaving normal smooth-motion tracking unaffected.

    process_var lower and measurement_var higher means the filter leans
    more on its motion model and less on each individual noisy
    measurement -- less responsive, but visibly less jittery, which is
    what this project's content-aware dwell tracking (gaze needs to sit
    still on one spot to register as "read") benefits from.
    """

    def __init__(self, process_var: float = 20.0, measurement_var: float = 0.5, max_innovation: float = 120.0):
        self._process_var = process_var
        self._measurement_var = measurement_var
        self._max_innovation = max_innovation
        self._kf = make_kalman_filter(process_var, measurement_var)
        self._initialized = False
        self._last_timestamp_ms: int | None = None

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
            # Scale process noise with dt: more elapsed time between
            # measurements means more genuine uncertainty in where the
            # constant-velocity model's extrapolation actually lands.
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
        # Return the corrected (post-measurement) state -- see module
        # docstring for why.
        return np.array([self._kf.statePost[0, 0], self._kf.statePost[1, 0]], dtype=np.float64)

    def reset(self) -> None:
        self._kf = make_kalman_filter(self._process_var, self._measurement_var)
        self._initialized = False
        self._last_timestamp_ms = None


class GazePointSmoother(KalmanSmoother):
    """Kalman output run through a light EMA on top, for extra smoothness.
    Used by scripts/phase2_gaze_scroll.py as the live gaze-point smoother.

    ema_alpha: 0.0 disables the EMA pass (pure Kalman output); closer to
    1.0 means stronger smoothing and more lag.
    """

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

    def reset(self) -> None:
        super().reset()
        self._ema_point = None
