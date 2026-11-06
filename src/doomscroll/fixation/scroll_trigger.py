"""The full scroll-or-not decision, combining plain gaze position with the
content-aware signals from content/gaze_content.py:

  1. Gaze in the bottom BOTTOM_DEFINITE_FRACTION of the focused window
     ("bottom 15%" by default) -> definitely scroll. Short dwell (not a
     full 1s) since being this far down is itself a strong, fast signal.
  2. Else, if the content tracker says gaze has read down to the last
     visible block and moved past it -> scroll.
  3. Else, ask the content tracker whether there's any unseen content
     between gaze and the bottom of the window. If not, it's fine to
     scroll (after a longer, more conservative dwell, since this tier
     isn't position-gated at all -- it's the broadest one, so it leans on
     time to avoid firing off a single noisy frame).

If no ContentScanner is available (no pyobjc/Vision install, or a
platform without one -- see content/__init__.py), tiers 2 and 3 are
simply never reached (content_available=False), degrading cleanly to
"only the bottom-of-window check" rather than crashing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.content.base import ContentBlock
from doomscroll.content.gaze_content import ContentGazeTracker

BOTTOM_DEFINITE_FRACTION = 0.85  # bottom 15% of the focused window
FAST_DWELL_MS = 150  # tiers 1 and 2: short, not skipped outright -- see module docstring
CONTENT_TIER_DWELL_MS = 1000  # tier 3: the original, more conservative dwell
COOLDOWN_MS = 800

# Tier labels, named out here instead of inline string literals in update()
# below -- update() compares the current tier against self._dwell_tier
# (a plain string) every tick to decide whether to restart the dwell
# timer, so a typo'd literal in one of the three branches would silently
# create a tier that never matches a previously-stored one, breaking
# dwell continuity with no error anywhere. Referencing these constants
# instead doesn't change behaviour (same string values, checked against
# tests/test_scroll_trigger.py's substring assertions below) -- it just
# makes that typo class of bug a NameError instead of a silent one.
TIER_BOTTOM15 = "bottom15"
TIER_REACHED_END = "reached_end"
TIER_CONTENT_OK = "content_ok"


@dataclass
class ScrollTriggerConfig:
    bottom_definite_fraction: float = BOTTOM_DEFINITE_FRACTION
    fast_dwell_ms: int = FAST_DWELL_MS
    content_tier_dwell_ms: int = CONTENT_TIER_DWELL_MS
    cooldown_ms: int = COOLDOWN_MS


class ScrollTrigger:
    """Stateful: call update() once per gaze-loop tick with the current
    gaze fraction/position, and note_scroll_fired() right after actually
    injecting a scroll. Does NOT itself call ContentGazeTracker.update()
    (the dwell/"seen" bookkeeping) -- callers should do that separately
    with the raw gaze point on every tick, same cadence regardless of
    whether a scroll decision is being made."""

    def __init__(self, tracker: ContentGazeTracker, config: ScrollTriggerConfig | None = None):
        self._tracker = tracker
        self._config = config or ScrollTriggerConfig()
        self._dwell_start_ms: float | None = None
        self._dwell_tier: str | None = None
        self._cooldown_until_ms: float = 0.0

    def update(
        self,
        fy: float,
        blocks: list[ContentBlock],
        gaze_point: np.ndarray,
        viewport_bottom: float,
        timestamp_ms: float,
        content_available: bool,
    ) -> tuple[bool, str]:
        """Returns (should_fire_now, status_label) -- status_label is
        purely for the live debug overlay."""
        if timestamp_ms < self._cooldown_until_ms:
            self._dwell_start_ms = None
            self._dwell_tier = None
            return False, "cooldown..."

        if fy >= self._config.bottom_definite_fraction:
            tier, required_ms = TIER_BOTTOM15, self._config.fast_dwell_ms
        elif content_available and self._tracker.has_reached_end(blocks, gaze_point):
            tier, required_ms = TIER_REACHED_END, self._config.fast_dwell_ms
        elif content_available and self._tracker.should_scroll(blocks, gaze_point, viewport_bottom):
            tier, required_ms = TIER_CONTENT_OK, self._config.content_tier_dwell_ms
        else:
            self._dwell_start_ms = None
            self._dwell_tier = None
            return False, "not scrolling"

        if tier != self._dwell_tier:
            self._dwell_tier = tier
            self._dwell_start_ms = timestamp_ms
        elapsed = timestamp_ms - self._dwell_start_ms
        if elapsed >= required_ms:
            self._dwell_start_ms = None
            self._dwell_tier = None
            return True, f"SCROLL ({tier})"
        return False, f"{tier} {elapsed:.0f}/{required_ms}ms"

    def note_scroll_fired(self, timestamp_ms: float) -> None:
        self._cooldown_until_ms = timestamp_ms + self._config.cooldown_ms

    def reset_dwell(self) -> None:
        """Clears any in-progress dwell without affecting cooldown. Callers
        should call this on ticks where update() isn't called at all --
        paused (looking away), blink, no face, or no usable window bounds
        -- so that e.g. a blink in the middle of dwelling at the bottom
        doesn't let the *next* qualifying frame fire instantly on leftover
        elapsed time from before the interruption."""
        self._dwell_start_ms = None
        self._dwell_tier = None
