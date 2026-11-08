# Tests for saving/listing/pooling/clearing calibration session data on disk.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doomscroll.gaze import session_store
from doomscroll.gaze.calibration import FEATURE_DIM


# Redirects session_store's SESSIONS_DIR to an isolated temp directory.
@pytest.fixture(autouse=True)
def _isolated_sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(session_store, "SESSIONS_DIR", sessions_dir)
    return sessions_dir


# Builds synthetic feature/target/group arrays for n_points calibration points.
def _make_session_arrays(n_points: int, samples_per_point: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    features, targets, groups = [], [], []
    for point_id in range(n_points):
        target = rng.uniform(0, 1000, size=2)
        for _ in range(samples_per_point):
            features.append(rng.normal(size=FEATURE_DIM))
            targets.append(target)
            groups.append(point_id)
    return np.array(features), np.array(targets), np.array(groups)


# Checks listing sessions is empty when nothing has been saved.
def test_list_sessions_empty_when_nothing_saved():
    assert session_store.list_sessions() == []


# Checks a saved session round-trips through list_sessions unchanged.
def test_save_and_list_round_trip_preserves_data():
    features, targets, groups = _make_session_arrays(n_points=3)
    session_store.save_session(features, targets, groups, (1920, 1080))

    sessions = session_store.list_sessions()
    assert len(sessions) == 1
    np.testing.assert_allclose(sessions[0].features, features)
    np.testing.assert_allclose(sessions[0].targets, targets)
    np.testing.assert_allclose(sessions[0].groups, groups)
    assert sessions[0].screen_size == (1920, 1080)


# Checks list_sessions returns the most recently saved session first.
def test_list_sessions_returns_newest_first():
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1920, 1080))

    sessions = session_store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].timestamp >= sessions[1].timestamp
    np.testing.assert_allclose(sessions[0].targets, t2)


# Checks sessions saved in the same second still get distinct file paths.
def test_save_session_disambiguates_same_second_collisions(monkeypatch: pytest.MonkeyPatch):
    fixed_time = 1_700_000_000.0
    monkeypatch.setattr(session_store.time, "time", lambda: fixed_time)

    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    path1 = session_store.save_session(f1, t1, g1, (1920, 1080))
    path2 = session_store.save_session(f2, t2, g2, (1920, 1080))

    assert path1 != path2
    assert path1.exists() and path2.exists()
    assert len(session_store.list_sessions()) == 2


# Checks pooling sessions remaps group ids so they stay globally unique.
def test_pool_sessions_remaps_group_ids_to_stay_globally_unique():
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1920, 1080))

    pooled_features, pooled_targets, pooled_groups, used = session_store.pool_sessions(
        session_store.list_sessions(), (1920, 1080)
    )

    assert len(used) == 2
    assert pooled_features.shape[0] == f1.shape[0] + f2.shape[0]
    assert len(np.unique(pooled_groups)) == 4


# Checks pooling skips sessions recorded at a different screen size.
def test_pool_sessions_skips_sessions_with_mismatched_screen_size():
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1280, 720))

    pooled_features, _, _, used = session_store.pool_sessions(session_store.list_sessions(), (1920, 1080))

    assert len(used) == 1
    assert pooled_features.shape[0] == f1.shape[0]


# Checks pooling respects the max_sessions cap.
def test_pool_sessions_respects_max_sessions_cap():
    for seed in range(5):
        f, t, g = _make_session_arrays(n_points=1, seed=seed)
        session_store.save_session(f, t, g, (1920, 1080))

    _, _, _, used = session_store.pool_sessions(session_store.list_sessions(), (1920, 1080), max_sessions=3)
    assert len(used) == 3


# Checks pooling an empty session list returns empty arrays.
def test_pool_sessions_with_no_matching_sessions_returns_empty():
    features, targets, groups, used = session_store.pool_sessions([], (1920, 1080))
    assert used == []
    assert features.shape == (0,)


# Checks clearing sessions removes all of them and reports the count removed.
def test_clear_sessions_removes_all_and_reports_count():
    for seed in range(3):
        f, t, g = _make_session_arrays(n_points=1, seed=seed)
        session_store.save_session(f, t, g, (1920, 1080))

    assert len(session_store.list_sessions()) == 3
    removed = session_store.clear_sessions()
    assert removed == 3
    assert session_store.list_sessions() == []


# Checks clearing an already-empty store returns zero.
def test_clear_sessions_on_empty_store_returns_zero():
    assert session_store.clear_sessions() == 0
