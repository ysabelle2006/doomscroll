"""Tests for scripts/calibrate.py's continuous cursor-tracing mechanics:
_lagged_cursor_position (pursuit-lag compensation), _map_real_to_logical
(the real-vs-logical pixel space fix for the dot/cursor misalignment bug),
the full-screen zigzag guide-path generators, the path-progress/margin
helpers that replace the old time-limit completion, and _MouseTracker
(OpenCV-mouse-callback-based cursor reading, replacing the OS-level
get_cursor_position() approach).

Pure logic, no camera/GUI needed -- same reasoning as
test_calibrate_pose_variant.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate import (
    CLICK_START_RADIUS_PX,
    END_RADIUS_PX,
    MIN_PROGRESS_TO_FINISH,
    PATH_MARGIN_PX,
    _euclidean,
    _face_frame_status,
    _lagged_cursor_position,
    _map_real_to_logical,
    _MouseTracker,
    _next_group_id,
    _path_progress,
    _point_to_segment_distance,
    _vertical_zigzag_path_fractions,
    _zigzag_path_fractions,
)


# --- _lagged_cursor_position -------------------------------------------


def test_lagged_position_before_history_returns_none():
    history = [(1.0, (100.0, 100.0)), (2.0, (200.0, 200.0))]
    assert _lagged_cursor_position(history, target_time=0.5) is None


def test_lagged_position_empty_history_returns_none():
    assert _lagged_cursor_position([], target_time=1.0) is None


def test_lagged_position_after_history_clamps_to_last_entry():
    history = [(1.0, (100.0, 100.0)), (2.0, (200.0, 200.0))]
    assert _lagged_cursor_position(history, target_time=5.0) == (200.0, 200.0)


def test_lagged_position_exact_match_returns_that_sample():
    history = [(1.0, (100.0, 100.0)), (2.0, (200.0, 200.0))]
    assert _lagged_cursor_position(history, target_time=1.0) == (100.0, 100.0)


def test_lagged_position_interpolates_between_bracketing_samples():
    history = [(1.0, (100.0, 0.0)), (2.0, (200.0, 100.0))]
    result = _lagged_cursor_position(history, target_time=1.5)
    assert result == pytest.approx((150.0, 50.0))


def test_lagged_position_three_point_history_picks_correct_bracket():
    history = [(1.0, (0.0, 0.0)), (2.0, (100.0, 0.0)), (3.0, (300.0, 0.0))]
    # Between the 2nd and 3rd sample, not the 1st and 2nd.
    result = _lagged_cursor_position(history, target_time=2.5)
    assert result == pytest.approx((200.0, 0.0))


# --- _map_real_to_logical -------------------------------------------


def test_map_real_to_logical_is_noop_when_display_size_none():
    assert _map_real_to_logical((500.0, 300.0), 1000, 800, None) == (500.0, 300.0)


def test_map_real_to_logical_is_noop_when_sizes_match():
    assert _map_real_to_logical((500.0, 300.0), 1000, 800, (1000, 800)) == (500.0, 300.0)


def test_map_real_to_logical_scales_down_for_a_denser_real_display():
    # Real window renders at 2x the logical size (e.g. a Retina mismatch) --
    # a point at the real display's center should map to the logical center.
    result = _map_real_to_logical((1000.0, 800.0), 1000, 800, (2000, 1600))
    assert result == pytest.approx((500.0, 400.0))


def test_map_real_to_logical_scales_up_for_a_sparser_real_display():
    result = _map_real_to_logical((500.0, 400.0), 1000, 800, (500, 400))
    assert result == pytest.approx((1000.0, 800.0))


# --- guide-path generators (full-screen open zigzag, replacing closed shapes) --


def test_zigzag_path_is_open_not_closed():
    path = _zigzag_path_fractions()
    assert path[0] != path[-1]


def test_zigzag_path_covers_nearly_the_full_screen_fraction_range():
    path = _zigzag_path_fractions()
    xs = [fx for fx, _ in path]
    ys = [fy for _, fy in path]
    assert min(xs) <= 0.05
    assert max(xs) >= 0.95
    assert min(ys) <= 0.05
    assert max(ys) >= 0.95


def test_zigzag_path_points_stay_within_unit_bounds():
    path = _zigzag_path_fractions()
    for fx, fy in path:
        assert 0.0 <= fx <= 1.0
        assert 0.0 <= fy <= 1.0


def test_zigzag_path_alternates_row_direction():
    """Consecutive rows should start from opposite ends (boustrophedon),
    i.e. row i's last point and row i+1's first point should be the same
    x (a vertical connector) rather than jumping back across the screen."""
    path = _zigzag_path_fractions(n_rows=4)
    # 2 points per row (left end, right end) in this simplified generator.
    points_per_row = 2
    for row in range(3):
        last_of_row = path[row * points_per_row + points_per_row - 1]
        first_of_next = path[(row + 1) * points_per_row]
        assert last_of_row[0] == pytest.approx(first_of_next[0])


def test_zigzag_path_respects_custom_ranges():
    path = _zigzag_path_fractions(x_range=(0.15, 0.85), y_range=(0.25, 0.75), n_rows=3)
    xs = [fx for fx, _ in path]
    ys = [fy for _, fy in path]
    assert min(xs) == pytest.approx(0.15)
    assert max(xs) == pytest.approx(0.85)
    assert min(ys) == pytest.approx(0.25)
    assert max(ys) == pytest.approx(0.75)


def test_vertical_zigzag_path_is_open_and_swept_by_column():
    path = _vertical_zigzag_path_fractions()
    assert path[0] != path[-1]
    xs = [fx for fx, _ in path]
    assert xs == sorted(xs)  # columns visited in increasing x order


def test_vertical_zigzag_path_is_spatially_distinct_from_main_zigzag():
    """Validation should not just be a relabeled copy of the main pattern."""
    main_path = _zigzag_path_fractions()
    val_path = _vertical_zigzag_path_fractions()
    assert main_path != val_path


# --- _point_to_segment_distance -------------------------------------------


def test_point_on_segment_has_zero_distance():
    dist, t = _point_to_segment_distance((50.0, 0.0), (0.0, 0.0), (100.0, 0.0))
    assert dist == pytest.approx(0.0)
    assert t == pytest.approx(0.5)


def test_point_off_segment_perpendicular_distance():
    dist, t = _point_to_segment_distance((50.0, 10.0), (0.0, 0.0), (100.0, 0.0))
    assert dist == pytest.approx(10.0)
    assert t == pytest.approx(0.5)


def test_point_beyond_segment_end_clamps_t_to_one():
    dist, t = _point_to_segment_distance((150.0, 0.0), (0.0, 0.0), (100.0, 0.0))
    assert t == pytest.approx(1.0)
    assert dist == pytest.approx(50.0)


def test_point_before_segment_start_clamps_t_to_zero():
    dist, t = _point_to_segment_distance((-50.0, 0.0), (0.0, 0.0), (100.0, 0.0))
    assert t == pytest.approx(0.0)
    assert dist == pytest.approx(50.0)


def test_degenerate_zero_length_segment_returns_distance_to_point():
    dist, t = _point_to_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0))
    assert dist == pytest.approx(5.0)
    assert t == pytest.approx(0.0)


# --- _path_progress -------------------------------------------


def test_path_progress_at_start_is_near_zero():
    path = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
    dist, progress = _path_progress((0.0, 0.0), path)
    assert dist == pytest.approx(0.0)
    assert progress == pytest.approx(0.0)


def test_path_progress_at_end_is_one():
    path = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
    dist, progress = _path_progress((200.0, 0.0), path)
    assert dist == pytest.approx(0.0)
    assert progress == pytest.approx(1.0)


def test_path_progress_at_midpoint_is_half():
    path = [(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)]
    dist, progress = _path_progress((100.0, 0.0), path)
    assert progress == pytest.approx(0.5)


def test_path_progress_distance_off_path():
    path = [(0.0, 0.0), (100.0, 0.0)]
    dist, progress = _path_progress((50.0, 30.0), path)
    assert dist == pytest.approx(30.0)


def test_path_progress_picks_globally_closest_point_not_first_segment():
    """A point near the END of a multi-segment path should report high
    progress, not be miscategorized against an early segment it happens
    to also be geometrically near (e.g. a zigzag doubling back)."""
    path = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    dist, progress = _path_progress((0.0, 100.0), path)
    assert dist == pytest.approx(0.0)
    assert progress == pytest.approx(1.0)


def test_path_progress_margin_threshold_matches_constant_usage():
    """Sanity check that PATH_MARGIN_PX is usable directly against the
    distance _path_progress returns, the way run_continuous_trace() uses it."""
    path = [(0.0, 0.0), (1000.0, 0.0)]
    dist, _ = _path_progress((500.0, PATH_MARGIN_PX + 1.0), path)
    assert dist > PATH_MARGIN_PX


# --- completion condition sanity (mirrors run_continuous_trace()'s reached_end) --


def test_reached_end_requires_both_progress_and_proximity():
    path = [(0.0, 0.0), (1000.0, 0.0)]
    # High progress but far from the literal final point (e.g. overshoot off-path).
    far_point = (950.0, 500.0)
    _, progress = _path_progress(far_point, path)
    assert progress >= MIN_PROGRESS_TO_FINISH
    assert _euclidean(far_point, path[-1]) > END_RADIUS_PX  # should NOT count as reached_end


def test_reached_end_true_when_both_conditions_hold():
    path = [(0.0, 0.0), (1000.0, 0.0)]
    near_end = (995.0, 0.0)
    _, progress = _path_progress(near_end, path)
    assert progress >= MIN_PROGRESS_TO_FINISH
    assert _euclidean(near_end, path[-1]) <= END_RADIUS_PX


def test_reached_end_false_midway_through_path():
    path = [(0.0, 0.0), (1000.0, 0.0)]
    midpoint = (500.0, 0.0)
    _, progress = _path_progress(midpoint, path)
    assert progress < MIN_PROGRESS_TO_FINISH


# --- _MouseTracker -------------------------------------------


def test_mouse_tracker_starts_with_no_position_or_click():
    tracker = _MouseTracker()
    assert tracker.pos is None
    assert tracker.clicked is False
    assert tracker.click_pos is None


def test_mouse_tracker_updates_pos_on_move_without_clicking():
    tracker = _MouseTracker()
    tracker(event=0, x=42, y=99, flags=0, param=None)  # cv2.EVENT_MOUSEMOVE == 0
    assert tracker.pos == (42, 99)
    assert tracker.clicked is False
    assert tracker.click_pos is None


def test_mouse_tracker_records_left_button_down():
    tracker = _MouseTracker()
    tracker(event=1, x=123, y=456, flags=0, param=None)  # cv2.EVENT_LBUTTONDOWN == 1
    assert tracker.clicked is True
    assert tracker.click_pos == (123, 456)
    assert tracker.pos == (123, 456)  # every event, including clicks, updates .pos


def test_mouse_tracker_tracks_pos_continuously_across_many_moves():
    tracker = _MouseTracker()
    for i in range(5):
        tracker(event=0, x=i * 10, y=i * 20, flags=0, param=None)
    assert tracker.pos == (40, 80)
    assert tracker.clicked is False


def test_mouse_tracker_reset_click_clears_click_state_but_not_pos():
    tracker = _MouseTracker()
    tracker(event=1, x=10, y=10, flags=0, param=None)
    tracker.reset_click()
    assert tracker.clicked is False
    assert tracker.click_pos is None
    assert tracker.pos == (10, 10)  # live position is not a "click" concept; stays


# --- click-start-radius sanity (used by wait_for_click) -------------------


def test_click_within_start_radius_of_pattern_start_counts():
    start = (500.0, 500.0)
    near_click = (start[0] + CLICK_START_RADIUS_PX - 1, start[1])
    assert _euclidean(near_click, start) <= CLICK_START_RADIUS_PX


def test_click_outside_start_radius_does_not_count():
    start = (500.0, 500.0)
    far_click = (start[0] + CLICK_START_RADIUS_PX + 50, start[1])
    assert _euclidean(far_click, start) > CLICK_START_RADIUS_PX


# --- _next_group_id -------------------------------------------


def test_next_group_id_with_empty_groups_returns_current():
    assert _next_group_id(5, np.array([], dtype=int)) == 5


def test_next_group_id_one_past_the_max_seen():
    groups = np.array([0, 0, 1, 1, 2])
    assert _next_group_id(0, groups) == 3


def test_next_group_id_chains_across_passes():
    first_pass_groups = np.array([0, 0, 1])
    next_id = _next_group_id(0, first_pass_groups)
    assert next_id == 2
    second_pass_groups = np.array([2, 2, 3, 3])
    next_id = _next_group_id(next_id, second_pass_groups)
    assert next_id == 4


# --- _face_frame_status (unchanged from the previous revision, re-verified here) ---


def test_face_frame_status_centered_face_is_in_frame():
    points_px = np.array([[200, 200], [400, 200], [200, 400], [400, 400]], dtype=np.float64)
    status, in_frame, box = _face_frame_status(points_px, image_width=640, image_height=480)
    assert in_frame is True
    assert status == "in frame"
    assert box is not None
