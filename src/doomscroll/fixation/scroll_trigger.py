# Decides when to fire an auto-scroll based on gaze dwell tiers and cooldown.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from doomscroll.content.base import ContentBlock
from doomscroll.content.gaze_content import ContentGazeTracker

# Dwell/cooldown thresholds and viewport-bottom fraction for triggering a scroll.
BOTTOM_DEFINITE_FRACTION = 0.85
FAST_DWELL_MS = 150
CONTENT_TIER_DWELL_MS = 1000
COOLDOWN_MS = 800

# Names of the dwell tiers that can trigger a scroll.
TIER_BOTTOM15 = "bottom15"
TIER_REACHED_END = "reached_end"
TIER_CONTENT_OK = "content_ok"


# Configurable thresholds for ScrollTrigger.
@dataclass
class ScrollTriggerConfig:
    bottom_definite_fraction: float = BOTTOM_DEFINITE_FRACTION
    fast_dwell_ms: int = FAST_DWELL_MS
    content_tier_dwell_ms: int = CONTENT_TIER_DWELL_MS
    cooldown_ms: int = COOLDOWN_MS


# Tracks gaze dwell against tiered thresholds and fires scrolls with a cooldown.
class ScrollTrigger:

    # Stores the content tracker/config and resets dwell/cooldown state.
    def __init__(self, tracker: ContentGazeTracker, config: ScrollTriggerConfig | None = None):
        self._tracker = tracker
        self._config = config or ScrollTriggerConfig()
        self._dwell_start_ms: float | None = None
        self._dwell_tier: str | None = None
        self._cooldown_until_ms: float = 0.0

    # Checks dwell time against the current tier; returns whether to scroll and a status string.
    def update(
        self,
        fy: float,
        blocks: list[ContentBlock],
        gaze_point: np.ndarray,
        viewport_bottom: float,
        timestamp_ms: float,
        content_available: bool,
    ) -> tuple[bool, str]:
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

    # Starts the cooldown window after a scroll has fired.
    def note_scroll_fired(self, timestamp_ms: float) -> None:
        self._cooldown_until_ms = timestamp_ms + self._config.cooldown_ms

    # Clears any in-progress dwell tracking.
    def reset_dwell(self) -> None:
        self._dwell_start_ms = None
        self._dwell_tier = None
