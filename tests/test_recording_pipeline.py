# End-to-end tests covering recording, dataset building, training and model persistence.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.ml.dataset import SessionDataset, build_session_dataset, read_frame_log
from doomscroll.ml.features import FEATURE_DIM, FEATURE_NAMES, YAW_INDEX
from doomscroll.ml.labels import activity_mask, cursor_at, make_labels
from doomscroll.ml.model import AffineCorrection, TrainedGazeModel
from doomscroll.ml.training import CANDIDATES, evaluate_candidates, fit_final, pool
from doomscroll.recording.recorder import Recorder
from doomscroll.recording.screen import nearest_index
from doomscroll.recording.session import SessionMeta, SessionPaths

# Shared synthetic screen size.
SCREEN = (1000, 600)


# Controllable fake clock for deterministic timing in tests.
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# Fake camera that returns solid-color frames, optionally failing periodically.
class FakeCamera:

    def __init__(self, clock: FakeClock, fail_every: int = 0):
        self.clock, self.i, self.fail_every, self.reads = clock, 0, fail_every, 0

    def read(self):
        self.clock.t += 1 / 30
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            return False, None
        frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.i += 1
        return True, frame


# Fake cursor that sweeps horizontally across the screen over time.
class SweepingCursor:

    def __init__(self, clock: FakeClock):
        self.clock = clock

    def position(self):
        return (self.clock.t % 1.0) * SCREEN[0], 300.0


# Records a session with fake camera/cursor sources for the given duration.
def record(tmp_path, duration=4.0, fail_every=0):
    clock = FakeClock()
    paths = SessionPaths(tmp_path / "s1")
    meta = SessionMeta("s1", "tester", *SCREEN)
    rec = Recorder(FakeCamera(clock, fail_every), SweepingCursor(clock), paths, meta, clock=clock)
    return paths, rec.run(duration)


# Checks every written video frame has exactly one matching CSV log row.
def test_every_video_frame_has_exactly_one_log_row(tmp_path):
    import cv2

    paths, meta = record(tmp_path, fail_every=7)
    t, xy = read_frame_log(paths.frames_csv)
    cap = cv2.VideoCapture(str(paths.webcam_video))
    n_video = 0
    while cap.read()[0]:
        n_video += 1
    assert n_video == len(t) == meta.n_frames
    assert np.all(np.diff(t) > 0)
    assert meta.measured_fps == pytest.approx(30 * 6 / 7, rel=0.05)
    assert paths.read_meta().n_frames == meta.n_frames


# Checks recording into an existing session directory raises instead of overwriting.
def test_session_never_overwritten(tmp_path):
    record(tmp_path)
    with pytest.raises(FileExistsError):
        record(tmp_path)


# Checks an interrupted recording still writes partial metadata to disk.
def test_partial_session_keeps_meta_when_interrupted(tmp_path):
    clock = FakeClock()
    paths = SessionPaths(tmp_path / "s2")
    cam = FakeCamera(clock)

    def explode(*_):
        if cam.i > 10:
            raise KeyboardInterrupt
        return False

    rec = Recorder(cam, SweepingCursor(clock), paths, SessionMeta("s2", "t", *SCREEN), clock=clock, on_frame=explode)
    with pytest.raises(KeyboardInterrupt):
        rec.run(100)
    assert paths.read_meta().n_frames > 10


# Checks activity_mask/make_labels/cursor_at behave correctly for still vs moving cursors.
def test_labels_lag_and_idle_filter():
    t = np.arange(0, 3000, 33.3)
    still = np.tile([500.0, 300.0], (len(t), 1))
    assert not activity_mask(t, still).any()
    moving = np.column_stack([np.linspace(0, 1000, len(t)), np.full(len(t), 300.0)])
    assert activity_mask(t, moving).all()
    y, keep = make_labels(t, moving, SCREEN, lag_ms=0)
    assert y[0, 0] == pytest.approx(0.0) and keep.all()
    y_lag, _ = make_labels(t, moving, SCREEN, lag_ms=300)
    assert y_lag[10, 0] > y[10, 0]
    np.testing.assert_allclose(cursor_at(np.array([50.0]), np.array([0.0, 100.0]), np.array([[0, 0], [10, 20.0]])), [[5, 10]])


# Checks nearest_index finds the closest timestamp for various queries.
def test_nearest_index():
    ts = np.array([0.0, 100.0, 200.0])
    assert [nearest_index(q, ts) for q in (-5, 40, 60, 149, 151, 999)] == [0, 0, 1, 1, 2, 2]


# Builds a fake featurizer that reports a blink every 10th frame.
def fake_featurize_factory():
    calls = {"i": 0}

    def featurize(frame, ts_ms):
        i = calls["i"]
        calls["i"] += 1
        if i % 10 == 0:
            return None, "blink"
        return np.zeros(FEATURE_DIM), "ok"

    return featurize


# Checks building a session dataset drops blinks correctly and round-trips via save/load.
def test_dataset_build_drops_and_labels(tmp_path):
    paths, meta = record(tmp_path)
    ds = build_session_dataset(paths.dir, fake_featurize_factory())
    assert len(ds.t_ms) == meta.n_frames
    assert ds.summary()["blink"] == pytest.approx(meta.n_frames / 10, abs=1)
    lab = ds.labeled()
    assert len(lab.X) < meta.n_frames and not np.isnan(lab.X).any()
    assert lab.y_frac.min() >= 0 and lab.y_frac.max() <= 1

    ds.save(tmp_path / "ds.npz")
    again = SessionDataset.load(tmp_path / "ds.npz")
    np.testing.assert_array_equal(again.reasons, ds.reasons)
    assert again.screen_wh == SCREEN


# Builds synthetic multi-session labeled frame data with a known yaw-dependent target.
def synthetic_sessions(n_sessions=4, n=600, seed=0):
    from doomscroll.ml.dataset import LabeledFrames

    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_sessions):
        X = rng.normal(size=(n, FEATURE_DIM))
        X[:, YAW_INDEX] = rng.uniform(-30, 30, n)
        fx = 0.5 + 0.3 * X[:, 0] + 0.004 * X[:, YAW_INDEX]
        fy = 0.5 + 0.3 * X[:, 1]
        y = np.clip(np.column_stack([fx, fy]), 0, 1)
        out.append(LabeledFrames(f"s{s}", SCREEN, X, y, np.arange(n) * 33.0))
    return out


# Checks training on pooled synthetic sessions learns and the saved model round-trips.
def test_training_learns_and_saves(tmp_path):
    data = pool(synthetic_sessions())
    assert data.cv_kind == "leave-sessions-out"
    report = evaluate_candidates(data, ["ridge"])
    assert report["ridge"]["mean_px"] < 60
    assert report["ridge"]["by_yaw"]

    model = TrainedGazeModel(fit_final("ridge", data), {"kind": "ridge"})
    model.save(tmp_path / "m.joblib")
    loaded = TrainedGazeModel.load(tmp_path / "m.joblib")
    x = data.X[0]
    np.testing.assert_allclose(loaded.predict_px(x, SCREEN), model.predict_px(x, SCREEN))


# Checks a single long session falls back to time-block cross-validation.
def test_few_sessions_fall_back_to_time_blocks():
    data = pool(synthetic_sessions(n_sessions=1, n=1500))
    assert "time blocks" in data.cv_kind
    assert evaluate_candidates(data, ["ridge"])["ridge"]["mean_px"] < 60


# Checks every training candidate model fits and predicts the expected shape.
def test_all_candidates_fit():
    data = pool(synthetic_sessions(n_sessions=3, n=200))
    for name in CANDIDATES:
        assert fit_final(name, data).predict(data.X[:3]).shape == (3, 2)


# Checks loading a model saved with different feature names raises an error.
def test_model_rejects_changed_features(tmp_path):
    import joblib

    joblib.dump({"estimator": None, "metadata": {}, "feature_names": FEATURE_NAMES[:-1]}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="different features"):
        TrainedGazeModel.load(tmp_path / "old.joblib")


# Checks AffineCorrection removes a known bias and stays safe with too few points.
def test_affine_correction_fixes_bias_and_stays_safe_with_few_points():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.1, 0.9, (9, 2))
    biased = truth * 0.9 + 0.05
    fixed = AffineCorrection.fit(biased, truth, shrink=0.01)
    err_before = np.abs(biased - truth).mean()
    err_after = np.abs(np.array([fixed.apply(p) for p in biased]) - truth).mean()
    assert err_after < err_before / 5
    ident = AffineCorrection.fit(np.empty((0, 2)), np.empty((0, 2)))
    np.testing.assert_allclose(ident.apply(np.array([0.3, 0.7])), [0.3, 0.7])


# Checks extract_features returns the expected shape and known values in order.
def test_extract_features_shape_and_order():
    from doomscroll.gaze.head_pose import HeadPose
    from doomscroll.ml.features import extract_features

    pts = np.random.default_rng(0).uniform(100, 500, (478, 2))
    pose = HeadPose(5.0, -3.0, 1.0, np.zeros((3, 1)), np.array([[10.0], [-20.0], [600.0]]))
    f = extract_features(pts, pose, frame_gray=None)
    assert f.shape == (FEATURE_DIM,)
    assert f[FEATURE_NAMES.index("yaw")] == 5.0
    assert f[FEATURE_NAMES.index("head_tz")] == 600.0


# Checks fit_user_correction's leave-one-out error is much better than before correction.
def test_fit_user_correction_reports_loo():
    from doomscroll.ml.model import fit_user_correction

    rng = np.random.default_rng(2)
    truth = np.array([(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)])
    biased = truth * 0.85 + 0.07 + rng.normal(0, 0.003, truth.shape)
    _, before, after = fit_user_correction(biased, truth, SCREEN, shrink=0.05)
    assert after < before / 3


# Checks leave-person-out pooling never tests on a person seen in training.
def test_leave_person_out_never_tests_on_a_training_person():
    from dataclasses import replace

    from sklearn.model_selection import GroupKFold

    sessions = synthetic_sessions(n_sessions=6, n=200)
    people = ["anni", "anni", "bo", "bo", "cy", "cy"]
    sessions = [replace(s, participant=p) for s, p in zip(sessions, people)]
    data = pool(sessions, group_by="person")
    assert data.cv_kind.startswith("leave-person-out (3 people")
    for train, test in GroupKFold(n_splits=3).split(data.X, data.y, data.groups):
        assert not set(data.participant[train]) & set(data.participant[test])
    report = evaluate_candidates(data, ["ridge"])
    assert [p for p, _, _ in report["ridge"]["by_person"]] == ["anni", "bo", "cy"]
    assert pool(sessions).cv_kind == "leave-sessions-out"


# Checks leave-person-out pooling requires at least two participants.
def test_leave_person_out_needs_two_people():
    from dataclasses import replace

    sessions = [replace(s, participant="anni") for s in synthetic_sessions(n_sessions=3, n=100)]
    with pytest.raises(ValueError, match="at least 2 participants"):
        pool(sessions, group_by="person")
