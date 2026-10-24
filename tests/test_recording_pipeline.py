"""End-to-end test of record -> build dataset -> train, with a fake camera,
fake cursor and fake feature extractor (no webcam / MediaPipe needed)."""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.ml.dataset import SessionDataset, build_session_dataset, read_frame_log
from doomscroll.ml.features import FEATURE_DIM, FEATURE_NAMES, YAW_INDEX
from doomscroll.ml.labels import (
    LabelSettings, activity_mask, click_times, cursor_at, cursor_quality_warnings, make_labels, near_click_mask,
    screen_coverage, speed_mask,
)
from doomscroll.ml.model import AffineCorrection, TrainedGazeModel
from doomscroll.ml.training import CANDIDATES, evaluate_candidates, fit_final, pool
from doomscroll.recording.recorder import Recorder
from doomscroll.recording.screen import nearest_index
from doomscroll.recording.session import SessionMeta, SessionPaths

SCREEN = (1000, 600)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCamera:
    """30 fps camera; each frame encodes its index in pixel (0,0) and advances the clock."""

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


class SweepingCursor:
    """Cursor sweeps left->right across the screen once per second."""

    def __init__(self, clock: FakeClock):
        self.clock = clock

    def position(self):
        return (self.clock.t % 1.0) * SCREEN[0], 300.0

    def button_down(self):
        return (self.clock.t % 1.0) < 0.1  # one short "click" per sweep


def record(tmp_path, duration=4.0, fail_every=0):
    clock = FakeClock()
    paths = SessionPaths(tmp_path / "s1")
    meta = SessionMeta("s1", "tester", *SCREEN)
    rec = Recorder(FakeCamera(clock, fail_every), SweepingCursor(clock), paths, meta, clock=clock)
    return paths, rec.run(duration)


def test_every_video_frame_has_exactly_one_log_row(tmp_path):
    import cv2

    paths, meta = record(tmp_path, fail_every=7)  # failed reads must not desync video and log
    t, xy, down, active = read_frame_log(paths.frames_csv)
    assert down.any() and active.all()
    cap = cv2.VideoCapture(str(paths.webcam_video))
    n_video = 0
    while cap.read()[0]:
        n_video += 1
    assert n_video == len(t) == meta.n_frames
    assert np.all(np.diff(t) > 0)
    assert meta.measured_fps == pytest.approx(30 * 6 / 7, rel=0.05)
    assert paths.read_meta().n_frames == meta.n_frames


def test_session_never_overwritten(tmp_path):
    record(tmp_path)
    with pytest.raises(FileExistsError):
        record(tmp_path)


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


def test_labels_lag_and_idle_filter():
    t = np.arange(0, 3000, 33.3)
    still = np.tile([500.0, 300.0], (len(t), 1))
    assert not activity_mask(t, still).any()
    moving = np.column_stack([np.linspace(0, 1000, len(t)), np.full(len(t), 300.0)])
    assert activity_mask(t, moving).all()
    y, keep = make_labels(t, moving, SCREEN, LabelSettings(lag_ms=0))
    assert y[0, 0] == pytest.approx(0.0) and keep.all()
    y_lag, _ = make_labels(t, moving, SCREEN, LabelSettings(lag_ms=300))
    assert y_lag[10, 0] > y[10, 0]  # label taken from the cursor's FUTURE position
    np.testing.assert_allclose(cursor_at(np.array([50.0]), np.array([0.0, 100.0]), np.array([[0, 0], [10, 20.0]])), [[5, 10]])


def test_nearest_index():
    ts = np.array([0.0, 100.0, 200.0])
    assert [nearest_index(q, ts) for q in (-5, 40, 60, 149, 151, 999)] == [0, 0, 1, 1, 2, 2]


def fake_featurize_factory():
    """Pretend gaze features: the 'iris' tracks cursor x in the frame index, so
    ridge can learn it. Frame content is ignored; we use call order."""
    calls = {"i": 0}

    def featurize(frame, ts_ms):
        i = calls["i"]
        calls["i"] += 1
        if i % 10 == 0:
            return None, "blink"
        return np.zeros(FEATURE_DIM), "ok"

    return featurize


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


def synthetic_sessions(n_sessions=4, n=600, seed=0):
    """Eye feature + head yaw -> screen position, with a yaw-dependent shift
    (the 'head angle changes the mapping' effect) the model must learn."""
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


def test_training_learns_and_saves(tmp_path):
    data = pool(synthetic_sessions())
    assert data.cv_kind == "leave-sessions-out"
    report = evaluate_candidates(data, ["ridge"])
    assert report["ridge"]["mean_px"] < 60  # screen is 1000px wide; chance is ~400
    assert report["ridge"]["by_yaw"]

    model = TrainedGazeModel(fit_final("ridge", data), {"kind": "ridge"})
    model.save(tmp_path / "m.joblib")
    loaded = TrainedGazeModel.load(tmp_path / "m.joblib")
    x = data.X[0]
    np.testing.assert_allclose(loaded.predict_px(x, SCREEN), model.predict_px(x, SCREEN))


def test_few_sessions_fall_back_to_time_blocks():
    data = pool(synthetic_sessions(n_sessions=1, n=1500))
    assert "time blocks" in data.cv_kind
    assert evaluate_candidates(data, ["ridge"])["ridge"]["mean_px"] < 60


def test_all_candidates_fit():
    data = pool(synthetic_sessions(n_sessions=3, n=200))
    for name in CANDIDATES:
        assert fit_final(name, data).predict(data.X[:3]).shape == (3, 2)


def test_model_rejects_changed_features(tmp_path):
    import joblib

    joblib.dump({"estimator": None, "metadata": {}, "feature_names": FEATURE_NAMES[:-1]}, tmp_path / "old.joblib")
    with pytest.raises(ValueError, match="different features"):
        TrainedGazeModel.load(tmp_path / "old.joblib")


def test_affine_correction_fixes_bias_and_stays_safe_with_few_points():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.1, 0.9, (9, 2))
    biased = truth * 0.9 + 0.05  # the shared model is shrunk + shifted for this user
    fixed = AffineCorrection.fit(biased, truth, shrink=0.01)
    err_before = np.abs(biased - truth).mean()
    err_after = np.abs(np.array([fixed.apply(p) for p in biased]) - truth).mean()
    assert err_after < err_before / 5
    ident = AffineCorrection.fit(np.empty((0, 2)), np.empty((0, 2)))  # no data -> identity, no crash
    np.testing.assert_allclose(ident.apply(np.array([0.3, 0.7])), [0.3, 0.7])


def test_extract_features_shape_and_order():
    from doomscroll.gaze.head_pose import HeadPose
    from doomscroll.ml.features import extract_features

    pts = np.random.default_rng(0).uniform(100, 500, (478, 2))
    pose = HeadPose(5.0, -3.0, 1.0, np.zeros((3, 1)), np.array([[10.0], [-20.0], [600.0]]))
    f = extract_features(pts, pose, frame_gray=None)
    assert f.shape == (FEATURE_DIM,)
    assert f[FEATURE_NAMES.index("yaw")] == 5.0
    assert f[FEATURE_NAMES.index("head_tz")] == 600.0


def test_fit_user_correction_reports_loo():
    from doomscroll.ml.model import fit_user_correction

    rng = np.random.default_rng(2)
    truth = np.array([(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)])
    biased = truth * 0.85 + 0.07 + rng.normal(0, 0.003, truth.shape)
    _, before, after = fit_user_correction(biased, truth, SCREEN, shrink=0.05)
    assert after < before / 3


# ---- game-play recording: pause, clicks, fast flicks, locked cursor ----

T = np.arange(0, 6000, 33.3)


def test_pause_flag_is_recorded_and_excluded(tmp_path):
    clock = FakeClock()
    paths = SessionPaths(tmp_path / "g1")
    cam = FakeCamera(clock)
    rec = Recorder(cam, SweepingCursor(clock), paths, SessionMeta("g1", "t", *SCREEN), clock=clock,
                   is_paused=lambda: 1.0 <= clock.t < 2.0)
    rec.run(4.0)
    t, xy, down, active = read_frame_log(paths.frames_csv)
    assert 0.2 < (1 - active.mean()) < 0.3  # ~1 s of 4 s paused, rows still written
    ds = build_session_dataset(paths.dir, fake_featurize_factory())
    labeled = ds.labeled()
    assert not np.isin(labeled.t_ms, t[active == 0]).any()


def test_speed_filter_drops_flicks_but_keeps_tracking():
    slow = np.column_stack([np.linspace(0, 500, len(T)), np.full(len(T), 300.0)])  # ~83 px/s
    fast = np.column_stack([np.linspace(0, 100000, len(T)), np.full(len(T), 300.0)])  # ~16000 px/s
    assert speed_mask(T, slow, 600, 3000).all()
    assert not speed_mask(T, fast, 600, 3000).any()


def test_click_edges_and_window():
    pressed = np.zeros(len(T), dtype=int)
    pressed[100:110] = 1  # one held press: counts once
    clicks = click_times(T, pressed)
    assert len(clicks) == 1 and clicks[0] == pytest.approx(T[100])
    near = near_click_mask(T, pressed, before_ms=500, after_ms=100)
    assert near[100] and near[90] and not near[50]  # before the press yes, long before no
    assert not near[130]  # well after


def test_label_modes():
    still = np.tile([500.0, 300.0], (len(T), 1))
    pressed = np.zeros(len(T), dtype=int)
    pressed[100] = 1
    _, keep_moving = make_labels(T, still, SCREEN, LabelSettings(mode="moving"), pressed=pressed)
    _, keep_clicks = make_labels(T, still, SCREEN, LabelSettings(mode="clicks"), pressed=pressed)
    _, keep_either = make_labels(T, still, SCREEN, LabelSettings(mode="either"), pressed=pressed)
    assert not keep_moving.any()  # parked cursor: not usable by itself
    assert keep_clicks.any() and keep_clicks.sum() < len(T) / 4  # only around the click
    assert keep_either.sum() == keep_clicks.sum()
    _, keep_no_clicks_logged = make_labels(T, still, SCREEN, LabelSettings(mode="clicks"))
    assert not keep_no_clicks_logged.any()
    with pytest.raises(ValueError):
        LabelSettings(mode="nonsense")


def test_inactive_frames_dropped():
    moving = np.column_stack([np.linspace(0, 1000, len(T)), np.full(len(T), 300.0)])
    active = np.ones(len(T), dtype=int)
    active[50:80] = 0
    _, keep = make_labels(T, moving, SCREEN, active=active)
    assert keep[40] and not keep[60:75].any()


def test_locked_cursor_is_flagged():
    locked = np.tile([500.0, 300.0], (len(T), 1))  # FPS game pinned the cursor to one point
    assert any("ONE position" in w for w in cursor_quality_warnings(T, locked, SCREEN))
    roaming = np.column_stack([np.linspace(0, 1000, len(T)), 300 + 100 * np.sin(T / 300)])
    assert cursor_quality_warnings(T, roaming, SCREEN) == []


def test_screen_coverage_shows_gaps():
    left_only = np.column_stack([np.random.default_rng(0).uniform(0, 0.4, 500), np.random.default_rng(1).uniform(0, 1, 500)])
    counts = screen_coverage(left_only, grid=5)
    assert counts.sum() == 500 and (counts[:, 3:] == 0).all() and (counts[:, :2] > 0).all()


def test_old_v1_frame_log_still_loads(tmp_path):
    (tmp_path / "frames.csv").write_text("frame_idx,t_ms,cursor_x,cursor_y\n0,0.0,10,20\n1,33.0,11,21\n")
    t, xy, down, active = read_frame_log(tmp_path / "frames.csv")
    assert len(t) == 2 and not down.any() and active.all()
