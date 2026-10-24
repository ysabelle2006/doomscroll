"""Tests for doomscroll.gaze.smoothing: the Kalman filter that smooths the
predicted gaze point before it drives scrolling. Pure numeric logic, no
webcam/MediaPipe involved.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.smoothing import GazePointSmoother, KalmanSmoother


def test_first_update_returns_the_raw_point():
    smoother = KalmanSmoother()
    out = smoother.update(np.array([100.0, 200.0]), timestamp_ms=0)
    np.testing.assert_allclose(out, [100.0, 200.0])


def test_smooths_toward_a_steady_target():
    """Repeatedly feeding the same point (with a touch of noise) should
    converge the filter's output close to that point, not drift away."""
    rng = np.random.default_rng(0)
    smoother = KalmanSmoother(process_var=50.0, measurement_var=0.2)
    target = np.array([500.0, 400.0])
    out = smoother.update(target, timestamp_ms=0)
    for i in range(1, 60):
        noisy = target + rng.normal(scale=1.5, size=2)
        out = smoother.update(noisy, timestamp_ms=i * 33)
    np.testing.assert_allclose(out, target, atol=5.0)


def test_rejects_single_frame_outlier_via_innovation_gating():
    """A single wild jump (e.g. a bad landmark detection on one frame)
    should be clipped rather than immediately pulling the output to the
    outlier's position -- that's the whole point of max_innovation."""
    smoother = KalmanSmoother(process_var=50.0, measurement_var=0.2, max_innovation=120.0)
    steady = np.array([300.0, 300.0])
    for i in range(10):
        smoother.update(steady, timestamp_ms=i * 33)

    outlier = np.array([3000.0, 3000.0])
    out = smoother.update(outlier, timestamp_ms=10 * 33)

    distance_moved = float(np.linalg.norm(out - steady))
    distance_to_outlier = float(np.linalg.norm(outlier - steady))
    assert distance_moved < distance_to_outlier * 0.5


def test_reset_clears_state_so_next_update_is_treated_as_first():
    smoother = KalmanSmoother()
    smoother.update(np.array([100.0, 100.0]), timestamp_ms=0)
    smoother.update(np.array([110.0, 100.0]), timestamp_ms=33)
    smoother.reset()
    out = smoother.update(np.array([900.0, 900.0]), timestamp_ms=0)
    # Immediately after reset, the first update should pass the raw point
    # straight through again (same behavior as a fresh smoother).
    np.testing.assert_allclose(out, [900.0, 900.0])


def test_large_dt_is_clamped_and_does_not_explode_state():
    """A long gap between frames (a slow inference tick) shouldn't make the
    constant-velocity prediction blow up -- MAX_DT_S exists to bound this."""
    smoother = KalmanSmoother()
    smoother.update(np.array([200.0, 200.0]), timestamp_ms=0)
    smoother.update(np.array([210.0, 200.0]), timestamp_ms=33)
    # a 5-second gap -- without clamping, accumulated velocity extrapolation
    # could put the prediction far outside any reasonable screen position
    out = smoother.update(np.array([220.0, 200.0]), timestamp_ms=5033)
    assert np.all(np.isfinite(out))
    assert abs(out[0] - 220.0) < 2000.0  # sanity bound, not a tight tolerance


def test_gaze_point_smoother_ema_alpha_validation():
    with pytest.raises(ValueError):
        GazePointSmoother(ema_alpha=1.5)
    with pytest.raises(ValueError):
        GazePointSmoother(ema_alpha=-0.1)
    GazePointSmoother(ema_alpha=0.0)
    GazePointSmoother(ema_alpha=1.0)


def test_gaze_point_smoother_ema_zero_matches_pure_kalman():
    """ema_alpha=0.0 should disable the extra EMA pass entirely, so
    GazePointSmoother's output matches a plain KalmanSmoother fed the same
    sequence."""
    rng = np.random.default_rng(1)
    points = [np.array([100.0, 100.0]) + rng.normal(scale=2.0, size=2) for _ in range(20)]

    plain = KalmanSmoother()
    eased = GazePointSmoother(ema_alpha=0.0)
    for i, p in enumerate(points):
        out_plain = plain.update(p, timestamp_ms=i * 33)
        out_eased = eased.update(p, timestamp_ms=i * 33)
        np.testing.assert_allclose(out_eased, out_plain)


def test_gaze_point_smoother_higher_alpha_smooths_more():
    """A higher ema_alpha should produce output that moves less in response
    to the same noisy input sequence (more lag/smoothing)."""
    rng = np.random.default_rng(2)
    points = [np.array([0.0, 0.0]) + rng.normal(scale=10.0, size=2) for _ in range(40)]

    low = GazePointSmoother(ema_alpha=0.1)
    high = GazePointSmoother(ema_alpha=0.9)
    low_outs = [low.update(p, timestamp_ms=i * 33) for i, p in enumerate(points)]
    high_outs = [high.update(p, timestamp_ms=i * 33) for i, p in enumerate(points)]

    low_variation = float(np.mean([np.linalg.norm(low_outs[i] - low_outs[i - 1]) for i in range(1, len(low_outs))]))
    high_variation = float(np.mean([np.linalg.norm(high_outs[i] - high_outs[i - 1]) for i in range(1, len(high_outs))]))
    assert high_variation < low_variation
