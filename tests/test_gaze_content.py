"""Tests for doomscroll.content.gaze_content.ContentGazeTracker -- the pure
dwell/"seen" tracking and the has_reached_end()/should_scroll() decisions
built on top of it. No camera, no screen capture, no threading: synthetic
ContentBlock lists and gaze points with known, hand-computed answers, same
style as the rest of this project's pure-logic tests.

BackgroundContentScanner (the threaded plumbing in the same module) isn't
covered here -- it has no interesting pure logic of its own (it's a thin
scan-on-an-interval loop around a real ContentScanner), consistent with
this project not testing other live-camera/live-screen paths either.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.content.base import ContentBlock
from doomscroll.content.gaze_content import ContentGazeTracker


def _block(text: str, x: int, y: int, w: int = 100, h: int = 20) -> ContentBlock:
    return ContentBlock(text=text, x=x, y=y, width=w, height=h)


def test_gaze_on_a_block_is_not_seen_before_the_dwell_time_elapses():
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])  # inside the block

    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=100)  # only 100ms elapsed so far

    assert "hello world" not in tracker.seen_ids


def test_gaze_on_a_block_is_seen_once_the_dwell_time_elapses():
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=250)  # past the 200ms threshold

    assert "hello world" in tracker.seen_ids


def test_wider_block_requires_more_dwell_than_narrower_block_by_default():
    """With no explicit seen_dwell_ms override, the required dwell scales
    with the block's own width -- a long line needs more sustained gaze
    before it counts as "read" than a short one."""
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


def test_default_dwell_has_a_floor_even_for_a_near_zero_width_block():
    """The size-dependent dwell still has a floor (SEEN_DWELL_MIN_MS=150) --
    a near-zero-width block shouldn't be markable as "seen" almost
    instantly just because the width term contributes almost nothing.

    A 1px-wide block's required dwell is SEEN_DWELL_MIN_MS + 0.5*1 =
    150.5ms, so 100ms must still be unseen and 200ms must be seen. (This
    test previously used 480ms/520ms against an assumed ~500ms floor,
    which never matched the actual SEEN_DWELL_MIN_MS=150 constant above --
    480ms was already well past the real floor, so the "not yet seen"
    assertion was failing against correct code, not catching a bug.)"""
    tracker = ContentGazeTracker()
    tiny = _block("i", x=0, y=0, w=1, h=20)
    gaze = np.array([0.5, 10.0])

    tracker.update([tiny], gaze, timestamp_ms=0)
    tracker.update([tiny], gaze, timestamp_ms=100)  # below the ~150.5ms floor
    assert "i" not in tracker.seen_ids

    tracker.update([tiny], gaze, timestamp_ms=200)  # past the floor
    assert "i" in tracker.seen_ids


def test_default_dwell_is_capped_for_an_unusually_wide_block():
    """The size-dependent dwell is also capped (SEEN_DWELL_MAX_MS) -- an
    unusually wide block shouldn't require an unreasonably long dwell."""
    tracker = ContentGazeTracker()
    huge = _block("a very long full-width line of text", x=0, y=0, w=5000, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([huge], gaze, timestamp_ms=0)
    # Uncapped, this width would require 500 + 0.8*5000 = 4500ms. The cap
    # (SEEN_DWELL_MAX_MS=1500 by default) means it should already be seen
    # well before that.
    tracker.update([huge], gaze, timestamp_ms=1550)
    assert "a very long full-width line of text" in tracker.seen_ids


def test_explicit_seen_dwell_ms_overrides_size_dependence():
    """Passing a flat seen_dwell_ms opts out of size-dependence entirely --
    even a huge block is marked "seen" on the override's own timing, not
    scaled up for its width."""
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    huge = _block("a very long full-width line of text", x=0, y=0, w=5000, h=20)
    gaze = np.array([50.0, 10.0])

    tracker.update([huge], gaze, timestamp_ms=0)
    tracker.update([huge], gaze, timestamp_ms=1)

    assert "a very long full-width line of text" in tracker.seen_ids


def test_brief_off_block_blip_does_not_reset_dwell_progress():
    """Real gaze tracking is noisy -- a single tick where the reported
    gaze momentarily lands off the block it's really dwelling on
    shouldn't throw away dwell progress already made. Only a divergence
    that itself persists past the grace window counts as a real
    departure (see the next test)."""
    tracker = ContentGazeTracker(seen_dwell_ms=200, gap_grace_ms=150)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    on_block = np.array([50.0, 10.0])
    off_block = np.array([500.0, 500.0])

    tracker.update([block], on_block, timestamp_ms=0)  # dwell starts
    tracker.update([block], off_block, timestamp_ms=50)  # one noisy tick off
    tracker.update([block], on_block, timestamp_ms=100)  # back on, well within grace
    tracker.update([block], on_block, timestamp_ms=250)  # 250ms since the ORIGINAL start

    assert "hello world" in tracker.seen_ids


def test_sustained_departure_resets_dwell_progress():
    """A divergence that itself outlasts the grace window is a real
    departure, not noise -- dwell progress should restart from when gaze
    comes back, not be preserved (otherwise a quick glance + a much later
    quick glance could add up to "seen" even though neither alone was a
    real read)."""
    tracker = ContentGazeTracker(seen_dwell_ms=200, gap_grace_ms=150)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    on_block = np.array([50.0, 10.0])
    off_block = np.array([500.0, 500.0])

    tracker.update([block], on_block, timestamp_ms=0)  # dwell starts
    tracker.update([block], off_block, timestamp_ms=50)  # divergence begins
    tracker.update([block], off_block, timestamp_ms=400)  # still off 350ms later -- past grace, commits
    tracker.update([block], on_block, timestamp_ms=450)  # back on, clock restarts here
    tracker.update([block], on_block, timestamp_ms=600)  # only 150ms since restart

    assert "hello world" not in tracker.seen_ids


def test_gaze_near_but_outside_a_block_counts_as_a_hit_within_tolerance():
    """Real gaze estimates routinely land a little outside a block's
    exact rect (calibration noise, inter-line gaps) -- a point just
    outside the edge, well within HIT_TEST_MAX_DISTANCE_PX, should still
    count as dwelling on that block rather than "on" nothing."""
    tracker = ContentGazeTracker(seen_dwell_ms=200, hit_max_distance_px=35)
    block = _block("hello world", x=0, y=0, w=100, h=20)  # bottom edge at y=20
    just_outside = np.array([50.0, 30.0])  # 10px below the block's bottom edge

    tracker.update([block], just_outside, timestamp_ms=0)
    tracker.update([block], just_outside, timestamp_ms=250)

    assert "hello world" in tracker.seen_ids


def test_gaze_too_far_from_any_block_does_not_count_as_a_hit():
    """Tolerance has a limit -- a point genuinely nowhere near any block
    still counts as "on" nothing, not nearest-block-regardless-of-how-far."""
    tracker = ContentGazeTracker(seen_dwell_ms=0, hit_max_distance_px=35)
    block = _block("hello world", x=0, y=0, w=100, h=20)
    far_away = np.array([200.0, 200.0])  # well beyond the 35px tolerance

    tracker.update([block], far_away, timestamp_ms=0)
    tracker.update([block], far_away, timestamp_ms=1)

    assert "hello world" not in tracker.seen_ids


def test_gaze_in_the_gap_between_two_lines_is_attributed_to_the_nearer_one():
    """Two blocks (e.g. consecutive text lines) with a gap between them --
    a gaze point sitting in that gap should be attributed to whichever
    block it's actually closer to, not left unattributed to either."""
    tracker = ContentGazeTracker(seen_dwell_ms=0, hit_max_distance_px=35)
    top_line = _block("line one", x=0, y=0, w=100, h=20)  # bottom = 20
    bottom_line = _block("line two", x=0, y=40, w=100, h=20)  # top = 40
    # 12px from top_line's bottom edge, 8px from bottom_line's top edge.
    in_the_gap_closer_to_bottom = np.array([50.0, 32.0])

    tracker.update([top_line, bottom_line], in_the_gap_closer_to_bottom, timestamp_ms=0)
    tracker.update([top_line, bottom_line], in_the_gap_closer_to_bottom, timestamp_ms=1)

    assert "line two" in tracker.seen_ids
    assert "line one" not in tracker.seen_ids


def test_seen_identity_is_by_text_and_survives_the_block_moving():
    """After a scroll, the same text typically reappears at a different y
    -- "seen" should still recognize it as the same block it already
    is_gaze_usable'd past, by text content, not by (now-stale) position."""
    tracker = ContentGazeTracker(seen_dwell_ms=200)
    block_before = _block("paragraph one", x=0, y=300, w=200, h=20)
    gaze = np.array([50.0, 310.0])
    tracker.update([block_before], gaze, timestamp_ms=0)
    tracker.update([block_before], gaze, timestamp_ms=250)
    assert "paragraph one" in tracker.seen_ids

    # Simulate a scroll: same text, shifted up, gaze no longer on it.
    block_after = _block("paragraph one", x=0, y=100, w=200, h=20)
    elsewhere = np.array([50.0, 500.0])
    tracker.update([block_after], elsewhere, timestamp_ms=500)

    assert "paragraph one" in tracker.seen_ids


def test_block_id_normalizes_whitespace_and_case():
    tracker = ContentGazeTracker(seen_dwell_ms=0)  # 0ms: seen on first tick
    block = _block("  Hello World  ", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])
    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=1)
    assert "hello world" in tracker.seen_ids


def test_seen_block_absent_from_a_later_scan_is_pruned():
    """Once a block no longer shows up in what's currently on screen (it
    really did scroll away, or OCR just didn't catch it that pass), it
    should stop being remembered as "seen" -- otherwise has_reached_end()/
    should_scroll() could be answered using state about content that isn't
    even on screen anymore."""
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    block = _block("transient", x=0, y=0, w=100, h=20)
    gaze = np.array([50.0, 10.0])
    tracker.update([block], gaze, timestamp_ms=0)
    tracker.update([block], gaze, timestamp_ms=1)
    assert "transient" in tracker.seen_ids

    other_block = _block("something else", x=0, y=0, w=100, h=20)
    tracker.update([other_block], np.array([500.0, 500.0]), timestamp_ms=2)
    assert "transient" not in tracker.seen_ids


def test_has_reached_end_false_when_lowest_block_not_yet_seen():
    tracker = ContentGazeTracker()
    lowest = _block("last line", x=0, y=900, w=200, h=20)
    gaze_past_it = np.array([50.0, 950.0])
    assert tracker.has_reached_end([lowest], gaze_past_it) is False


def test_has_reached_end_false_when_seen_but_gaze_hasnt_moved_past_it():
    tracker = ContentGazeTracker(seen_dwell_ms=0, end_margin_px=20)
    lowest = _block("last line", x=0, y=900, w=200, h=100)  # spans 900-1000
    gaze_on_it = np.array([50.0, 920.0])
    tracker.update([lowest], gaze_on_it, timestamp_ms=0)
    tracker.update([lowest], gaze_on_it, timestamp_ms=1)
    assert "last line" in tracker.seen_ids

    # Still well within the block (920), nowhere near its bottom edge
    # (1000) minus the margin (980).
    assert tracker.has_reached_end([lowest], gaze_on_it) is False


def test_has_reached_end_true_when_seen_and_gaze_moved_past_it():
    tracker = ContentGazeTracker(seen_dwell_ms=0, end_margin_px=20)
    lowest = _block("last line", x=0, y=900, w=200, h=100)  # bottom = 1000
    gaze_on_it = np.array([50.0, 920.0])
    tracker.update([lowest], gaze_on_it, timestamp_ms=0)
    tracker.update([lowest], gaze_on_it, timestamp_ms=1)

    gaze_past_it = np.array([50.0, 985.0])  # >= 1000 - 20 margin
    assert tracker.has_reached_end([lowest], gaze_past_it) is True


def test_has_reached_end_false_on_empty_blocks():
    tracker = ContentGazeTracker()
    assert tracker.has_reached_end([], np.array([0.0, 0.0])) is False


def test_should_scroll_true_on_empty_blocks():
    """Nothing on screen to hold the scroll back."""
    tracker = ContentGazeTracker()
    assert tracker.should_scroll([], np.array([0.0, 500.0]), viewport_bottom=1000.0) is True


def test_should_scroll_false_when_unseen_content_is_ahead():
    tracker = ContentGazeTracker()
    gaze = np.array([50.0, 100.0])
    unseen_ahead = _block("unread paragraph", x=0, y=200, w=200, h=20)
    assert tracker.should_scroll([unseen_ahead], gaze, viewport_bottom=1000.0) is False


def test_should_scroll_true_when_content_ahead_has_already_been_seen():
    tracker = ContentGazeTracker(seen_dwell_ms=0)
    ahead = _block("already read this", x=0, y=200, w=200, h=20)
    tracker.update([ahead], np.array([50.0, 210.0]), timestamp_ms=0)
    tracker.update([ahead], np.array([50.0, 210.0]), timestamp_ms=1)

    gaze_now_above_it = np.array([50.0, 100.0])
    assert tracker.should_scroll([ahead], gaze_now_above_it, viewport_bottom=1000.0) is True


def test_should_scroll_ignores_unseen_content_already_behind_gaze():
    """Content the user has already scrolled/looked past (center above
    the gaze point) shouldn't block scrolling just because it was never
    specifically dwelled on -- only content still AHEAD matters."""
    tracker = ContentGazeTracker()
    behind = _block("skipped earlier text", x=0, y=10, w=200, h=20)
    gaze = np.array([50.0, 500.0])
    assert tracker.should_scroll([behind], gaze, viewport_bottom=1000.0) is True


def test_should_scroll_ignores_unseen_content_below_the_viewport():
    """A block technically "ahead" in y but already below the window's
    visible area isn't something the user could be skipping over."""
    tracker = ContentGazeTracker()
    below_viewport = _block("off-screen text", x=0, y=1200, w=200, h=20)
    gaze = np.array([50.0, 500.0])
    assert tracker.should_scroll([below_viewport], gaze, viewport_bottom=1000.0) is True
