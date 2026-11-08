# Tests for ContentGazeTracker's dwell-based seen tracking and scroll-readiness checks.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.content.base import ContentBlock
from doomscroll.content.gaze_content import ContentGazeTracker


# Builds a ContentBlock with sensible size defaults for testing.
def _block(text: str, x: int, y: int, w: int = 100, h: int = 20) -> ContentBlock:
    return ContentBlock(text=text, x=x, y=y, width=w, height=h)


# Checks a block isn't seen before its dwell time elapses.
def test_gaze_on_a_block_is_not_seen_before_the_dwell_time_elapses():
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=100)

    assert "hello world" not in tracker.seen_ids


# Checks a block is seen once its dwell time elapses.
def test_gaze_on_a_block_is_seen_once_the_dwell_time_elapses():
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=250)

    assert "hello world" in tracker.seen_ids


# Checks a wider block requires more dwell time than a narrower one by default.
def test_wider_block_requires_more_dwell_than_narrower_block_by_default():
    narrow_tracker = ContentGazeTracker()
    wide_tracker = ContentGazeTracker()
    narrow = _block("ok", x=0, y=0, w=50, h=20)
    wide = _block("a full line of unusually long body text", x=0, y=0, w=1200, h=20)
    gaze = np.array([25.0, 10.0])

    narrow_tracker.update([narrow], gaze, timestamp_ms=0)
    wide_tracker.update([wide], gaze, timestamp_ms=0)
    narrow_tracker.update([narrow], gaze, timestamp_ms=700)
    wide_tracker.update([wide], gaze, timestamp_ms=700)

    assert "ok" in narrow_tracker.seen_ids
    assert "a full line of unusually long body text" not in wide_tracker.seen_ids


# Checks the default dwell time has a floor even for a near-zero-width block.
def test_default_dwell_has_a_floor_even_for_a_near_zero_width_block():
    tracker = ContentGazeTracker()
    tiny = _block("i", x=0, y=0, w=1, h=20)
    gaze = np.array([0.5, 10.0])

    tracker.update([tiny], gaze, timestamp_ms=0)
    tracker.update([tiny], gaze, timestamp_ms=100)
    assert "i" not in tracker.seen_ids

    tracker.update([tiny], gaze, timestamp_ms=200)
    assert "i" in tracker.seen_ids


# Checks the default dwell time is capped for an unusually wide block.
def test_default_dwell_is_capped_for_an_unusually_wide_block():
    tracker = ContentGazeTracker()
    huge = _block("a very long full-width line of text", x=0, y=0, w=5000, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([huge], gaze, timestamp_ms=0)
    tracker.update([huge], gaze, timestamp_ms=1550)
    assert "a very long full-width line of text" in tracker.seen_ids


# Checks an explicit seen_dwell_ms overrides the size-based default.
def test_explicit_seen_dwell_ms_overrides_size_dependence():
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    huge = _block("a very long full-width line of text", x=0, y=0, w=5000, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([huge], gaze, timestamp_ms=0)
    tracker.update([huge], gaze, timestamp_ms=1)

    assert "a very long full-width line of text" in tracker.seen_ids


# Checks a brief gaze blip off the block doesn't reset dwell progress.
def test_brief_off_block_blip_does_not_reset_dwell_progress():
    tracker = ContentGazeTracker(seen_dwell_ms=200, gap_grace_ms=150)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    on_block = np.array([50.0, 10.0])
    off_block = np.array([500.0, 500.0])

    tracker.update([block], on_block, timestamp_ms=0)
    tracker.update([block], off_block, timestamp_ms=50)
    tracker.update([block], on_block, timestamp_ms=100)
    tracker.update([block], on_block, timestamp_ms=250)

    assert "hello world" in tracker.seen_ids


# Checks a sustained departure from the block resets dwell progress.
def test_sustained_departure_resets_dwell_progress():
    tracker = ContentGazeTracker(seen_dwell_ms=200, gap_grace_ms=150)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    on_block = np.array([50.0, 10.0])
    off_block = np.array([500.0, 500.0])

    tracker.update([block], on_block, timestamp_ms=0)
    tracker.update([block], off_block, timestamp_ms=50)
    tracker.update([block], off_block, timestamp_ms=400)
    tracker.update([block], on_block, timestamp_ms=450)
    tracker.update([block], on_block, timestamp_ms=600)

    assert "hello world" not in tracker.seen_ids


# Checks gaze just outside a block still counts as a hit within tolerance.
def test_gaze_near_but_outside_a_block_counts_as_a_hit_within_tolerance():
    tracker = ContentGazeTracker(seen_dwell_ms=200, hit_max_distance_px=35)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    just_outside = np.array([50.0, 30.0])

    tracker.update([block], just_outside, timestamp_ms=0)
    tracker.update([block], just_outside, timestamp_ms=250)

    assert "hello world" in tracker.seen_ids


# Checks gaze too far from any block doesn't count as a hit.
def test_gaze_too_far_from_any_block_does_not_count_as_a_hit():
    tracker = ContentGazeTracker(seen_dwell_ms=0, hit_max_distance_px=35)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    far_away = np.array([200.0, 200.0])

    tracker.update([block], far_away, timestamp_ms=0)
    tracker.update([block], far_away, timestamp_ms=1)

    assert "hello world" not in tracker.seen_ids


# Checks gaze in the gap between two lines is attributed to the nearer one.
def test_gaze_in_the_gap_between_two_lines_is_attributed_to_the_nearer_one():
    tracker = ContentGazeTracker(seen_dwell_ms=0, hit_max_distance_px=35)
    top_line = _block("line one", x=0, y=0, w=100, h=20)
    bottom_line = _block("line two", x=0, y=40, w=100, h=20)
    in_the_gap_closer_to_bottom = np.array([50.0, 32.0])

    tracker.update([top_line, bottom_line], in_the_gap_closer_to_bottom, timestamp_ms=0)
    tracker.update([top_line, bottom_line], in_the_gap_closer_to_bottom, timestamp_ms=1)

    assert "line two" in tracker.seen_ids
    assert "line one" not in tracker.seen_ids


# Checks seen identity is tracked by text and survives the block moving.
def test_seen_identity_is_by_text_and_survives_the_block_moving():
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block_before = _block("paragraph one", x=0, y=300, w=200, h=20)
    gaze = np.array([50.0, 310.0])
    tracker.update([block_before], gaze, timestamp_ms=0)
    tracker.update([block_before], gaze, timestamp_ms=250)
    assert "paragraph one" in tracker.seen_ids

    block_after = _block("paragraph one", x=0, y=100, w=200, h=20)
    elsewhere = np.array([50.0, 500.0])
    tracker.update([block_after], elsewhere, timestamp_ms=500)

    assert "paragraph one" in tracker.seen_ids


# Checks block identity normalizes whitespace and case.
def test_block_id_normalizes_whitespace_and_case():
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    block = _block("  Hello World  ", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])
    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=1)
    assert "hello world" in tracker.seen_ids


# Checks a seen block no longer present in a later scan is pruned from seen_ids.
def test_seen_block_absent_from_a_later_scan_is_pruned():
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    block = _block("transient", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])
    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=1)
    assert "transient" in tracker.seen_ids

    other_block = _block("something else", x=0, y=0, w=100, h=20)
    tracker.update([other_block], np.array([500.0, 500.0]), timestamp_ms=2)
    assert "transient" not in tracker.seen_ids


# Checks has_reached_end is false when the lowest block hasn't been seen yet.
def test_has_reached_end_false_when_lowest_block_not_yet_seen():
    tracker = ContentGazeTracker()
    lowest = _block("last line", x=0, y=900, w=200, h=20)
    gaze_past_it = np.array([50.0, 950.0])
    assert tracker.has_reached_end([lowest], gaze_past_it) is False


# Checks has_reached_end is false when seen but gaze hasn't passed its margin.
def test_has_reached_end_false_when_seen_but_gaze_hasnt_moved_past_it():
    tracker = ContentGazeTracker(seen_dwell_ms=0, end_margin_px=20)
    lowest = _block("last line", x=0, y=900, w=200, h=100)
    gaze_on_it = np.array([50.0, 920.0])
    tracker.update([lowest], gaze_on_it, timestamp_ms=0)
    tracker.update([lowest], gaze_on_it, timestamp_ms=1)
    assert "last line" in tracker.seen_ids

    assert tracker.has_reached_end([lowest], gaze_on_it) is False


# Checks has_reached_end is true once seen and gaze has passed its margin.
def test_has_reached_end_true_when_seen_and_gaze_moved_past_it():
    tracker = ContentGazeTracker(seen_dwell_ms=0, end_margin_px=20)
    lowest = _block("last line", x=0, y=900, w=200, h=100)
    gaze_on_it = np.array([50.0, 920.0])
    tracker.update([lowest], gaze_on_it, timestamp_ms=0)
    tracker.update([lowest], gaze_on_it, timestamp_ms=1)

    gaze_past_it = np.array([50.0, 985.0])
    assert tracker.has_reached_end([lowest], gaze_past_it) is True


# Checks has_reached_end is false when there are no blocks.
def test_has_reached_end_false_on_empty_blocks():
    tracker = ContentGazeTracker()
    assert tracker.has_reached_end([], np.array([0.0, 0.0])) is False


# Checks should_scroll is true when there are no blocks.
def test_should_scroll_true_on_empty_blocks():
    tracker = ContentGazeTracker()
    assert tracker.should_scroll([], np.array([0.0, 500.0]), viewport_bottom=1000.0) is True


# Checks should_scroll is false when unseen content lies ahead of the gaze.
def test_should_scroll_false_when_unseen_content_is_ahead():
    tracker = ContentGazeTracker()
    gaze = np.array([50.0, 100.0])
    unseen_ahead = _block("unread paragraph", x=0, y=200, w=200, h=20)
    assert tracker.should_scroll([unseen_ahead], gaze, viewport_bottom=1000.0) is False


# Checks should_scroll is true when content ahead has already been seen.
def test_should_scroll_true_when_content_ahead_has_already_been_seen():
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    ahead = _block("already read this", x=0, y=200, w=200, h=20)
    tracker.update([ahead], np.array([50.0, 210.0]), timestamp_ms=0)
    tracker.update([ahead], np.array([50.0, 210.0]), timestamp_ms=1)

    gaze_now_above_it = np.array([50.0, 100.0])
    assert tracker.should_scroll([ahead], gaze_now_above_it, viewport_bottom=1000.0) is True


# Checks should_scroll ignores unseen content that's already behind the gaze.
def test_should_scroll_ignores_unseen_content_already_behind_gaze():
    tracker = ContentGazeTracker()
    behind = _block("skipped earlier text", x=0, y=10, w=200, h=20)
    gaze = np.array([50.0, 500.0])
    assert tracker.should_scroll([behind], gaze, viewport_bottom=1000.0) is True


# Checks should_scroll ignores unseen content that's below the viewport.
def test_should_scroll_ignores_unseen_content_below_the_viewport():
    tracker = ContentGazeTracker()
    below_viewport = _block("off-screen text", x=0, y=1200, w=200, h=20)
    gaze = np.array([50.0, 500.0])
    assert tracker.should_scroll([below_viewport], gaze, viewport_bottom=1000.0) is True
