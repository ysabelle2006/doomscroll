# Tests for GazeCalibrator fitting/prediction/persistence and accuracy benchmarks.
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

# Error budget for the regression benchmark against a synthetic session.
MAX_ACCEPTABLE_SYNTHETIC_ERROR_PX = 40.0


# Checks build_feature_vector concatenates gaze/pose/translation in order.
def test_build_feature_vector_concatenates_in_order():
    feature = build_feature_vector(
        gaze_vector=np.array([0.1, 0.2, 0.3, 0.4]),
        head_pose_deg=(5.0, -6.0, 7.0),
        head_translation_mm=(10.0, 20.0, 300.0),
    )
    assert feature.shape == (FEATURE_DIM,)
    np.testing.assert_allclose(feature, [0.1, 0.2, 0.3, 0.4, 5.0, -6.0, 7.0, 10.0, 20.0, 300.0])


# Checks fit() rejects fewer than the minimum required samples.
def test_fit_rejects_too_few_samples():
    too_few = MIN_CALIBRATION_SAMPLES - 1
    features = np.zeros((too_few, FEATURE_DIM))
    targets = np.zeros((too_few, 2))
    calibrator = GazeCalibrator()
    with pytest.raises(ValueError):
        calibrator.fit(features, targets)


# Checks predict() before fit() raises.
def test_predict_before_fit_raises():
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.predict(np.zeros(FEATURE_DIM))


# Checks selected_alpha before fit() raises.
def test_selected_alpha_before_fit_raises():
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        _ = calibrator.selected_alpha


# Checks validate() before fit() raises.
def test_validate_before_fit_raises():
    calibrator = GazeCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.validate(np.zeros((1, FEATURE_DIM)), np.zeros((1, 2)))


# Checks fit/predict recovers a simple known linear mapping.
def test_fit_predict_recovers_a_simple_known_mapping():
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


# Checks save/load round-trips a fitted calibrator's predictions unchanged.
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


# Checks loading a missing profile path raises FileNotFoundError.
def test_load_missing_profile_raises():
    with pytest.raises(FileNotFoundError):
        GazeCalibrator.load("/nonexistent/path/does-not-exist.joblib")


# Regression benchmark: checks mean validation error stays below budget on synthetic data.
def test_calibration_accuracy_regression_benchmark():
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


# Checks local correction reduces error versus the uncorrected baseline on nonlinear data.
def test_local_correction_reduces_error_on_the_nonlinear_synthetic_benchmark():
    sess = generate_synthetic_session(seed=0)

    baseline = GazeCalibrator(local_correction_max_weight=0.0)
    baseline.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    baseline_err = baseline.validate(sess.val_features, sess.val_targets)

    corrected = GazeCalibrator()
    corrected.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)
    corrected_err = corrected.validate(sess.val_features, sess.val_targets)

    assert corrected_err < baseline_err


# Checks local correction is zero for a feature far from any training sample.
def test_local_correction_is_zero_far_from_any_training_sample():
    sess = generate_synthetic_session(seed=2)
    calibrator = GazeCalibrator()
    calibrator.fit(sess.train_features, sess.train_targets, groups=sess.train_groups)

    far_feature = np.full(FEATURE_DIM, 1e6)
    X_far = calibrator._preprocessor.transform(far_feature.reshape(1, -1))
    np.testing.assert_allclose(calibrator._local_correction(X_far), np.zeros(2))


# Checks local correction state survives a save/load round trip.
def test_local_correction_survives_save_load_round_trip(tmp_path: Path):
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


# Error budget for the pursuit-protocol synthetic benchmark.
MAX_ACCEPTABLE_PURSUIT_SYNTHETIC_ERROR_PX = 120.0


# Checks a pursuit-protocol session fits and validates without error, within budget.
def test_pursuit_session_fits_and_validates_without_error():
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


# Checks the protocol-comparison session's still/pursuit/val arrays have consistent shapes.
def test_protocol_comparison_session_shapes_are_consistent():
    session = generate_protocol_comparison_session(seed=0)

    assert session.still_train_features.shape[1] == FEATURE_DIM
    assert session.pursuit_train_features.shape[1] == FEATURE_DIM
    assert session.val_features.shape[1] == FEATURE_DIM

    assert session.still_train_features.shape[0] == session.still_train_targets.shape[0] == session.still_train_groups.shape[0]
    assert session.pursuit_train_features.shape[0] == session.pursuit_train_targets.shape[0] == session.pursuit_train_groups.shape[0]
    assert session.val_features.shape[0] == session.val_targets.shape[0]
