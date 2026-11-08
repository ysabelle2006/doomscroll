# Tests for ScrollTrigger's dwell-tier selection, cooldown and priority rules.
from __future__ import annotations

import numpy as np
import pytest

from doomscroll.fixation.scroll_trigger import ScrollTrigger, ScrollTriggerConfig

# Shared gaze point, empty block list and viewport bottom used across tests.
GAZE = np.array([50.0, 50.0])
BLOCKS: list = []
VIEWPORT_BOTTOM = 1000.0


# Fake ContentGazeTracker with controllable reached_end/should_scroll results.
class _FakeTracker:
    def __init__(self, reached_end: bool = False, should_scroll: bool = False):
        self.reached_end = reached_end
        self.should_scroll_value = should_scroll

    def has_reached_end(self, blocks, gaze_point) -> bool:
        return self.reached_end

    def should_scroll(self, blocks, gaze_point, viewport_bottom) -> bool:
        return self.should_scroll_value


# Builds a ScrollTriggerConfig with sensible defaults, allowing overrides.
def _config(**overrides) -> ScrollTriggerConfig:
    base = dict(bottom_definite_fraction=0.85, fast_dwell_ms=150, content_tier_dwell_ms=1000, cooldown_ms=800)
    base.update(overrides)
    return ScrollTriggerConfig(**base)


# Checks nothing fires when no dwell tier's condition is satisfied.
def test_nothing_fires_when_no_tier_is_satisfied():
    trigger = ScrollTrigger(_FakeTracker(), _config())
    fired, label = trigger.update(0.5, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert fired is False
    assert label == "not scrolling"


# Checks the bottom-of-page tier doesn't fire before its dwell time elapses.
def test_bottom_tier_does_not_fire_before_its_dwell_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=150))
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert fired is False
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=100, content_available=True)
    assert fired is False
    assert "bottom15" in label


# Checks the bottom-of-page tier fires once its dwell time elapses.
def test_bottom_tier_fires_once_its_dwell_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=150))
    trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is True
    assert "bottom15" in label


# Checks the bottom tier takes priority over reached-end and content-ok.
def test_bottom_tier_takes_priority_over_reached_end_and_content_ok():
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=0))
    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert "bottom15" in label


# Checks the reached-end tier fires using the fast dwell time.
def test_reached_end_tier_fires_using_the_fast_dwell():
    tracker = _FakeTracker(reached_end=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is True
    assert "reached_end" in label


# Checks the content-ok tier fires using the longer content dwell time.
def test_content_ok_tier_fires_using_the_longer_dwell():
    tracker = _FakeTracker(should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(content_tier_dwell_ms=1000))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=500, content_available=True)
    assert fired is False
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1001, content_available=True)
    assert fired is True
    assert "content_ok" in label


# Checks reached-end takes priority over content-ok.
def test_reached_end_takes_priority_over_content_ok():
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=0))
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    assert "reached_end" in label


# Checks content tiers are skipped entirely when content is unavailable.
def test_content_tiers_are_skipped_when_content_unavailable():
    tracker = _FakeTracker(reached_end=True, should_scroll=True)
    trigger = ScrollTrigger(tracker, _config())
    fired, label = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=False)
    assert fired is False
    assert label == "not scrolling"


# Checks switching tiers mid-dwell restarts the dwell timer.
def test_switching_tiers_mid_dwell_restarts_the_timer():
    tracker = _FakeTracker(should_scroll=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150, content_tier_dwell_ms=1000))

    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=900, content_available=True)

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=900, content_available=True)
    assert fired is False

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1000, content_available=True)
    assert fired is False
    assert "bottom15" in label

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=1050, content_available=True)
    assert fired is True
    assert "bottom15" in label


# Checks reset_dwell() discards in-progress dwell tracking.
def test_reset_dwell_discards_in_progress_progress():
    tracker = _FakeTracker(reached_end=True)
    trigger = ScrollTrigger(tracker, _config(fast_dwell_ms=150))
    trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=0, content_available=True)

    trigger.reset_dwell()

    fired, _ = trigger.update(0.3, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=200, content_available=True)
    assert fired is False


# Checks the cooldown blocks firing even when a tier's condition is met.
def test_cooldown_blocks_firing_even_when_a_tier_is_satisfied():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=0, cooldown_ms=800))
    trigger.note_scroll_fired(timestamp_ms=0)

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=100, content_available=True)
    assert fired is False
    assert label == "cooldown..."


# Checks firing works again once the cooldown period has elapsed.
def test_firing_works_again_once_cooldown_elapses():
    trigger = ScrollTrigger(_FakeTracker(), _config(fast_dwell_ms=0, cooldown_ms=800))
    trigger.note_scroll_fired(timestamp_ms=0)

    fired, label = trigger.update(0.9, BLOCKS, GAZE, VIEWPORT_BOTTOM, timestamp_ms=801, content_available=True)
    assert fired is True
