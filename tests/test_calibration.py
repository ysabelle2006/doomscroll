"""Tests for doomscroll.gaze.calibration: the Ridge regression that maps a
(eye-in-head + head-pose) feature vector to a screen position, and whose
accuracy is what calibrate.py ultimately reports at the end of a real
session.

Covers two things:
  1. Ordinary unit tests of GazeCalibrator's API contract (minimum sample
     count, fit-before-predict, save/load round-trip).
  2. test_calibration_accuracy_regression_benchmark -- a synthetic,
     fully-reproducible "did my change to calibration.py help or hurt"
     check, using a known ground-truth mapping (see synthetic_calibration.py)
     instead of a real human calibration session. A hard pytest fail here
     means something broke badly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doomscroll.gaze.calibration import (
    MIN_CALIBRATION_SAMPLES,
    FEATURE_DIM,
    GazeCalibrator,
    build_feature_vector,
)
from tests.synthetic_calibration import (
    generate_protocol_comparison_session,
    generate_pursuit_session,
    generate_synthetic_session,
)

# A healthy GazeCalibrator fit on generate_synthetic_session(seed=0) lands
# at ~10-12px mean validation error across reruns (see the commented values
# this was calibrated against). This threshold is deliberately generous --
# several times that -- so ordinary seed-to-seed/version-to-version noise
# never fails the build; it exists to catch a real regression (e.g. the
# alpha search reaching for far more regularization than it should, which
# empirically pushes this same benchmark's error past 200px).
MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX = 40.0


def test_build_feature_vector_concatenates_in_order():
    feature = build_feature_vector(
        gaze_vector=np.array([0.1, 0.2, 0.3, 0.4]),
        head_pose_deg=(5.0, -6.0, 7.0),
        head_translation_mm=(10.0, 20.0, 300.0),
    )
    assert feature.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(feature, [0.1, 0.2, 0.3, 0.4, 5.0, -6.0, 7.0, 10.0, 20.0, 300.0])


def test_fit_rejects_too_few_samples():
    too_few = MIN_CALIBRATION_SAMPLES - 1
    features = np.zeros((too_few, FEATURE_DIM))
    targets = np.zeros((too_few, 2))
    calibrator = GazeCalibrator()
    with pytest.raises(ValueError):
        calibrator.fit(features, targets)


def test_predict_before_fit_raises():
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.predict(np.zeros(FEATURE_DIM))


def test_selected_alpha_before_fit_raises():
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        _ = calibrator.selected_alpha


def test_validate_before_fit_raises():
    """validate() delegates to predict() per sample (so local correction
    applies during validation too), so it raises the same clear
    RuntimeError as predict()/selected_alpha rather than leaking sklearn's
    internal NotFittedError."""
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.validate(np.zeros((1, FEATURE_DIM)), np.zeros((1, 2)))


def test_fit_predict_recovers_a_simple_known_mapping():
    """Sanity check on the API with the simplest possible case: a mapping
    where screen position is *exactly* 1000x the first feature dimension,
    nothing else -- predict() on a held-out point near the training range
    should recover it closely."""
    rng = np.random.default_rng(42)
    n_points = 16
    samples_per_point = 15
    features, targets, groups = [], [], []
    for point_id in range(n_points):
        base = rng.uniform(-0.2, 0.2)
        for _ in range(samples_per_point):
            f = np.zeros(FEATURE_DIM)
            f[0] = base + rng.normal(scale=0.005)
            features.append(f)
            targets.append([1000.0 * base, 500.0])
            groups.append(point_id)

    calibrator = GazeCalibrator()
    calibrator.fit(np.array(features), np.array(targets), groups=np.array(groups))

    probe = np.zeros(FEATURE_DIM)
    probe[0] = 0.1
    prediction = calibrator.predict(probe)
    assert prediction[0] == pytest.approx(100.0, abs=15.0)
    assert prediction[1] == pytest.approx(500.0, abs=15.0)


def test_save_load_round_trip_preserves_predictions(tmp_path: Path):
    sess = generate_synthetic_session(seed=1)
    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)

    before = np.array([calibrator.predict(f) for f in sess.val_features])

    profile_path = tmp_path / "profile.joblib"
    calibrator.save(profile_path)
    assert profile_path.exists()

    loaded = GazeCalibrator.load(profile_path)
    after = np.array([loaded.predict(f) for f in sess.val_features])

    np.testing.assert_allclose(after, before)


def test_load_missing_profile_raises():
    with pytest.raises(FileNotFoundError):
        GazeCalibrator.load("/nonexistent/path/does-not-exist.joblib")


def test_calibration_accuracy_regression_benchmark():
    """The main 'did a change to calibration.py help or hurt' check -- see
    module docstring and MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX above."""
    sess = generate_synthetic_session(seed=0)
    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    mean_error_px = calibrator.validate(sess.val_features, sess.val_targets)

    diag = (sess.screen_w**2 + sess.screen_h**2) ** 0.5
    print(f"\n[calibration benchmark] mean validation error: {mean_error_px:.2f}px ({100*mean_error_px/diag:.2f}% of screen diagonal)")

    assert mean_error_px < MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX, (
        f"Synthetic calibration accuracy regressed: {mean_error_px:.1f}px, "
        f"expected < {MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX}px."
    )


def test_local_correction_reduces_error_on_the_nonlinear_synthetic_benchmark():
    """synthetic_calibration.py deliberately bakes in a small cubic
    nonlinearity the degree-1 global model can't fit (see its module
    docstring on _NONLINEARITY_COEF) -- exactly the regime the local KNN
    residual correction exists for. Comparing against
    local_correction_max_weight=0.0 (which makes _local_correction always
    return zero) isolates its effect from everything else GazeCalibrator
    does."""
    sess = generate_synthetic_session(seed=0)

    baseline = GazeCalibrator(local_correction_max_weight=0.0)
    baseline.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    baseline_err = baseline.validate(sess.val_features, sess.val_targets)

    corrected = GazeCalibrator()
    corrected.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    corrected_err = corrected.validate(sess.val_features, sess.val_targets)

    assert corrected_err < baseline_err


def test_local_correction_is_zero_far_from_any_training_sample():
    """A live input far outside anything seen during calibration shouldn't
    get nudged by nearby-but-irrelevant residuals -- it should fall back to
    the pure global prediction, same as if local correction were absent."""
    sess = generate_synthetic_session(seed=2)
    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)

    far_feature = np.full(FEATURE_DIM, 1e6)
    X_far = calibrator._preprocessor.transform(far_feature.reshape(1, -1))
    np.testing.assert_allclose(calibrator._local_correction(X_far), np.zeros(2))


def test_local_correction_survives_save_load_round_trip(tmp_path: Path):
    """save()/load() must round-trip the local-correction training data
    too, not just the ridge models -- otherwise a freshly-loaded calibrator
    would silently lose the correction and predict differently than it did
    right before saving."""
    sess = generate_synthetic_session(seed=0)
    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    before = np.array([calibrator.predict(f) for f in sess.val_features])

    profile_path = tmp_path / "profile.joblib"
    calibrator.save(profile_path)
    loaded = GazeCalibrator.load(profile_path)

    assert loaded._train_X is not None and loaded._train_residuals is not None
    after = np.array([loaded.predict(f) for f in sess.val_features])
    np.testing.assert_allclose(after, before)


# The pursuit protocol has an intentionally modeled, *higher* error floor
# than MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX: it bakes in a systematic
# lag/undershoot mismatch between where the eye physically is and what the
# recorded label claims (see synthetic_calibration.py's module docstring on
# PURSUIT_LATENCY_S / PURSUIT_GAIN). That's the whole point of the model,
# not a bug -- empirically this lands around 55-70px across seeds, vs.
# ~10-12px for the still-point grid. This threshold (~2x the observed max)
# exists only to catch a genuinely broken fit, not to compare protocols
# against each other.
MAX_ACCEPTABLE_PURSUIT_SYNTHETIC_ERROR_PX = 120.0


def test_pursuit_session_fits_and_validates_without_error():
    """Sanity check for the cursor-pursuit protocol generator (see
    synthetic_calibration.py) -- this is deliberately NOT asserting pursuit
    is better or worse than the still-point grid (that remains an open,
    unautomated empirical question); it just confirms the pursuit-shaped
    dataset is well-formed and GazeCalibrator can fit/validate against it
    without error, within its own (higher, intentionally-modeled) error
    floor -- see MAX_ACCEPTABLE_PURSUIT_SYNTHETIC_ERROR_PX above."""
    sess = generate_pursuit_session(seed=0)
    assert sess.train_features.shape[1] == FEATURE_DIM
    assert sess.train_features.shape[0] == sess.train_targets.shape[0] == sess.train_groups.shape[0]

    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    mean_error_px = calibrator.validate(sess.val_features, sess.val_targets)
    print(f"\n[pursuit protocol] mean validation error: {mean_error_px:.2f}px")

    assert mean_error_px < MAX_ACCEPTABLE_PURSUIT_SYNTHETIC_ERROR_PX, (
        f"Pursuit-protocol synthetic benchmark regressed: {mean_error_px:.1f}px, "
        f"expected < {MAX_ACCEPTABLE_PURSUIT_SYNTHETIC_ERROR_PX}px."
    )


def test_protocol_comparison_session_shapes_are_consistent():
    """Shape/wiring sanity check for generate_protocol_comparison_session():
    both training sets use the real FEATURE_DIM layout, each has matching
    feature/target/group counts, and the validation set (shared by both
    protocols, by construction) is internally consistent."""
    session = generate_protocol_comparison_session(seed=0)

    assert session.still_train_features.shape[1] == FEATURE_DIM
    assert session.pursuit_train_features.shape[1] == FEATURE_DIM
    assert session.val_features.shape[1] == FEATURE_DIM

    assert session.still_train_features.shape[0] == session.still_train_targets.shape[0] == session.still_train_groups.shape[0]
    assert session.pursuit_train_features.shape[0] == session.pursuit_train_targets.shape[0] == session.pursuit_train_groups.shape[0]
    assert session.val_features.shape[0] == session.val_targets.shape[0]
