# Tests for KalmanSmoother and GazePointSmoother gaze-point filtering.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.gaze.smoothing import GazePointSmoother, KalmanSmoother


# Checks the first update returns the raw point unchanged.
def test_first_update_returns_the_raw_point():
    smoother = KalmanSmoother()
    out = smoother.update(np.array([100.0, 200.0]), timestamp_ms=0)
    np.testing.assert_allclose(out, [100.0, 200.0])


# Checks repeated noisy updates converge toward a steady target.
def test_smooths_toward_a_steady_target():
    rng = np.random.default_rng(0)
    smoother = KalmanSmoother(process_var=50.0, measurement_var=0.2)
    target = np.array([500.0, 400.0])
    out = smoother.update(target, timestamp_ms=0)
    for i in range(1, 60):
        noisy = target + rng.normal(scale=1.5, size=2)
        out = smoother.update(noisy, timestamp_ms=i * 33)
    np.testing.assert_allclose(out, target, atol=5.0)


# Checks a single outlier frame is mostly rejected via innovation gating.
def test_rejects_single_frame_outlier_via_innovation_gating():
    smoother = KalmanSmoother(process_var=50.0, measurement_var=0.2, max_innovation=120.0)
    steady = np.array([300.0, 300.0])
    for i in range(10):
        smoother.update(steady, timestamp_ms=i * 33)

    outlier = np.array([3000.0, 3000.0])
    out = smoother.update(outlier, timestamp_ms=10 * 33)

    distance_moved = float(np.linalg.norm(out - steady))
    distance_to_outlier = float(np.linalg.norm(outlier - steady))
    assert distance_moved < distance_to_outlier * 0.5


# Checks reset() clears state so the next update behaves like the first.
def test_reset_clears_state_so_next_update_is_treated_as_first():
    smoother = KalmanSmoother()
    smoother.update(np.array([100.0, 100.0]), timestamp_ms=0)
    smoother.update(np.array([110.0, 100.0]), timestamp_ms=33)
    smoother.reset()
    out = smoother.update(np.array([900.0, 900.0]), timestamp_ms=0)
    np.testing.assert_allclose(out, [900.0, 900.0])


# Checks a large time gap is clamped and doesn't blow up the filter state.
def test_large_dt_is_clamped_and_does_not_explode_state():
    smoother = KalmanSmoother()
    smoother.update(np.array([200.0, 200.0]), timestamp_ms=0)
    smoother.update(np.array([210.0, 200.0]), timestamp_ms=33)
    out = smoother.update(np.array([220.0, 200.0]), timestamp_ms=5033)
    assert np.all(np.isfinite(out))
    assert abs(out[0] - 220.0) < 2000.0


# Checks GazePointSmoother validates ema_alpha is within [0, 1].
def test_gaze_point_smoother_ema_alpha_validation():
    with pytest.raises(ValueError):
        GazePointSmoother(ema_alpha=1.5)
    with pytest.raises(ValueError):
        GazePointSmoother(ema_alpha=-0.1)
    GazePointSmoother(ema_alpha=0.0)
    GazePointSmoother(ema_alpha=1.0)


# Checks ema_alpha=0 makes GazePointSmoother match plain KalmanSmoother.
def test_gaze_point_smoother_ema_zero_matches_pure_kalman():
    rng = np.random.default_rng(1)
    points = [np.array([100.0, 100.0]) + rng.normal(scale=2.0, size=2) for _ in range(20)]

    plain = KalmanSmoother()
    eased = GazePointSmoother(ema_alpha=0.0)
    for i, p in enumerate(points):
        out_plain = plain.update(p, timestamp_ms=i * 33)
        out_eased = eased.update(p, timestamp_ms=i * 33)
        np.testing.assert_allclose(out_eased, out_plain)


# Checks a higher ema_alpha smooths output more than a lower one.
def test_gaze_point_smoother_higher_alpha_smooths_more():
    rng = np.random.default_rng(2)
    points = [np.array([0.0, 0.0]) + rng.normal(scale=10.0, size=2) for _ in range(40)]

    low = GazePointSmoother(ema_alpha=0.1)
    high = GazePointSmoother(ema_alpha=0.9)
    low_outs = [low.update(p, timestamp_ms=i * 33) for i, p in enumerate(points)]
    high_outs = [high.update(p, timestamp_ms=i * 33) for i, p in enumerate(points)]

    low_variation = float(np.mean([np.linalg.norm(low_outs[i] - low_outs[i - 1]) for i in range(1, len(low_outs))]))
    high_variation = float(np.mean([np.linalg.norm(high_outs[i] - high_outs[i - 1]) for i in range(1, len(high_outs))]))
    assert high_variation < low_variation
