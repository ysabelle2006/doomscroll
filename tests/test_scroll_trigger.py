"""Tests for doomscroll.fixation.scroll_trigger.ScrollTrigger -- the tier
selection (bottom-15% / reached-end / content-ok / nothing), per-tier dwell
timing, cross-tier dwell reset, and cooldown -- all state-machine logic, no
camera/screen/threading involved.

Uses a tiny fake content tracker (not the real ContentGazeTracker, which
has its own dedicated tests in test_gaze_content.py) so these tests can
pin has_reached_end()/should_scroll() to known values directly, in
isolation from whatever makes the real tracker return them.
"""
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.fixation.scroll_trigger import ScrollTrigger, ScrollTriggerConfig

GAZE = np.array([50.0, 50.0])
BLOCKS: list = []
VIEWPORT_BOTTOM = 1000.0


class _FakeTracker:
    def __init__(self, reached_end: bool = False, should_scroll: bool = False):
        self.reached_end = reached_end
        self.should_scroll_value = should_scroll

    def has_reached_end(self, blocks, gaze_point) -> bool:
        return self.reached_end

    def should_scroll(self, blocks, gaze_point, viewport_bottom) -> bool:
        return self.should_scroll_value


def _config(**overrides) -> ScrollTriggerConfig:
    base = dict(bottom_definite_fraction=0.85, fast_dwell_ms=150, content_tier_dwell_ms=1000, cooldown_ms=800)
    base.update(overrides)
    return ScrollTriggerConfig(**base)


def test_nothing_fires_when_no_tier_is_satisfied():
    trigger = ScrollTrigger(_FakeTracker(), _config())
    fired, label = trigger.update(0.5, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert fired is False
    assert label == "not scrolling"


def test_bottom_tier_does_not_fire_before_its_dwell_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=150))
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert fired is False
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=100, content_available=True)
    assert fired is False
    assert "bottom15" in label


def test_bottom_tier_fires_once_its_dwell_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=150))
    trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is True
    assert "bottom15" in label


def test_bottom_tier_takes_priority_over_reached_end_and_content_ok():
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=0))
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert "bottom15" in label


def test_reached_end_tier_fires_using_the_fast_dwell():
    tracker = _FakeTracker(reached_end=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is True
    assert "reached_end" in label


def test_content_ok_tier_fires_using_the_longer_dwell():
    tracker = _FakeTracker(should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(content_tier_dwell_ms=1000))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    # Not yet at 1000ms.
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=500, content_available=True)
    assert fired is False
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1001, content_available=True)
    assert fired is True
    assert "content_ok" in label


def test_reached_end_takes_priority_over_content_ok():
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=0))
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert "reached_end" in label


def test_content_tiers_are_skipped_when_content_unavailable():
    """content_available=False must behave as if the tracker always said
    False, even if it would otherwise say True -- this is the degrade-
    gracefully path for no pyobjc/Vision install."""
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config())
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=False)
    assert fired is False
    assert label == "not scrolling"


def test_switching_tiers_mid_dwell_restarts_the_timer():
    """Tier A partially dwelled, then tier B becomes active instead --
    tier B's dwell should start fresh at the switch, not inherit any of
    tier A's elapsed time."""
    tracker = _FakeTracker(should_scroll=True)  # content_ok available throughout
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150, content_tier_dwell_ms=1000))

    # content_ok tier active for 900ms (not yet enough to fire on its own).
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=900, content_available=True)

    # Now gaze moves to the bottom 15% -- a different tier -- at t=900.
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=900, content_available=True)
    assert fired is False  # just switched, 0ms into bottom15's own dwell

    # 100ms since the switch -- still short of fast_dwell_ms=150.
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1000, content_available=True)
    assert fired is False
    assert "bottom15" in label

    # 150ms since the switch -- now it fires. If the switch hadn't reset
    # the timer (i.e. it had kept content_ok's original t=0 start), this
    # would have fired far earlier than this test's first assertion did.
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1050, content_available=True)
    assert fired is True
    assert "bottom15" in label


def test_reset_dwell_discards_in_progress_progress():
    tracker = _FakeTracker(reached_end=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)

    trigger.reset_dwell()  # e.g. a blink happened

    # Resumes at t=100 -- without the reset, 100ms since t=0 would still be
    # short of 150ms anyway, so use a timestamp that would have fired had
    # the original dwell_start_ms of 0 survived (elapsed since 0 = 200ms),
    # but must NOT fire since the clock should have restarted at the
    # reset.
    fired, _ = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is False


def test_cooldown_blocks_firing_even_when_a_tier_is_satisfied():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=0, cooldown_ms=800))
    trigger.note_scroll_fired(timestamp_ms=0)

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=100, content_available=True)
    assert fired is False
    assert label == "cooldown..."


def test_firing_works_again_once_cooldown_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=0, cooldown_ms=800))
    trigger.note_scroll_fired(timestamp_ms=0)

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=801, content_available=True)
    assert fired is True
