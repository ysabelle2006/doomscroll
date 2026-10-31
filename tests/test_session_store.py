"""Tests for doomscroll.gaze.session_store: persisting raw calibration
samples across sessions and optionally pooling recent ones together.

All tests redirect SESSIONS_DIR into a pytest tmp_path via monkeypatch, so
nothing here ever touches the real on-disk calibration directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doomscroll.gaze import session_store
from doomscroll.gaze.calibration import FEATURE_DIM


@pytest.fixture(autouse=True)
def _isolated_sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(session_store, "SESSIONS_DIR", sessions_dir)
    return sessions_dir


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


def test_list_sessions_empty_when_nothing_saved():
    assert session_store.list_sessions() == []


def test_save_and_list_round_trip_preserves_data():
    features, targets, groups = _make_session_arrays(n_points=3)
    session_store.save_session(features, targets, groups, (1920, 1080))

    sessions = session_store.list_sessions()
    assert len(sessions) == 1
    np.testing.assert_allclose(sessions[0].features, features)
    np.testing.assert_allclose(sessions[0].targets, targets)
    np.testing.assert_allclose(sessions[0].groups, groups)
    assert sessions[0].screen_size == (1920, 1080)


def test_list_sessions_returns_newest_first():
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1920, 1080))

    sessions = session_store.list_sessions()
    assert len(sessions) == 2
    assert sessions[0].timestamp >= sessions[1].timestamp
    # the most recently saved session (seed=2) should be first
    np.testing.assert_allclose(sessions[0].targets, t2)


def test_save_session_disambiguates_same_second_collisions(monkeypatch: pytest.MonkeyPatch):
    """Two sessions saved within the same wall-clock second must not
    overwrite each other -- save_session should append a numeric suffix."""
    fixed_time = 1_700_000_000.0
    monkeypatch.setattr(session_store.time, "time", lambda: fixed_time)

    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    path1 = session_store.save_session(f1, t1, g1, (1920, 1080))
    path2 = session_store.save_session(f2, t2, g2, (1920, 1080))

    assert path1 != path2
    assert path1.exists() and path2.exists()
    assert len(session_store.list_sessions()) == 2


def test_pool_sessions_remaps_group_ids_to_stay_globally_unique():
    # Both sessions use group ids 0, 1 internally -- pooling must not let
    # point 0 of session A collide with point 0 of session B.
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1920, 1080))

    pooled_features, pooled_targets, pooled_groups, used = session_store.pool_sessions(
        session_store.list_sessions(), (1920, 1080)
    )

    assert len(used) == 2
    assert pooled_features.shape[0] == f1.shape[0] + f2.shape[0]
    # 4 distinct groups total (2 per session), none colliding across sessions
    assert len(np.unique(pooled_groups)) == 4


def test_pool_sessions_skips_sessions_with_mismatched_screen_size():
    f1, t1, g1 = _make_session_arrays(n_points=2, seed=1)
    f2, t2, g2 = _make_session_arrays(n_points=2, seed=2)
    session_store.save_session(f1, t1, g1, (1920, 1080))
    session_store.save_session(f2, t2, g2, (1280, 720))  # different screen -- must be excluded

    pooled_features, _, _, used = session_store.pool_sessions(session_store.list_sessions(), (1920, 1080))

    assert len(used) == 1
    assert pooled_features.shape[0] == f1.shape[0]


def test_pool_sessions_respects_max_sessions_cap():
    for seed in range(5):
        f, t, g = _make_session_arrays(n_points=1, seed=seed)
        session_store.save_session(f, t, g, (1920, 1080))

    _, _, _, used = session_store.pool_sessions(session_store.list_sessions(), (1920, 1080), max_sessions=3)
    assert len(used) == 3


def test_pool_sessions_with_no_matching_sessions_returns_empty():
    features, targets, groups, used = session_store.pool_sessions([], (1920, 1080))
    assert used == []
    assert features.shape == (0,)


def test_clear_sessions_removes_all_and_reports_count():
    for seed in range(3):
        f, t, g = _make_session_arrays(n_points=1, seed=seed)
        session_store.save_session(f, t, g, (1920, 1080))

    assert len(session_store.list_sessions()) == 3
    removed = session_store.clear_sessions()
    assert removed == 3
    assert session_store.list_sessions() == []


def test_clear_sessions_on_empty_store_returns_zero():
    assert session_store.clear_sessions() == 0
