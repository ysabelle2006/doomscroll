"""Content-aware scroll decision layer: tracks which on-screen text blocks
the gaze has actually dwelled on, and from that answers the two questions
the live scroll loop needs: has gaze dwelled on the lowest visible block
and then moved past it (has_reached_end), and is there any unseen content
still ahead of gaze within the viewport (should_scroll).

Hit-testing and dwell-tracking are deliberately tolerant of noise, not
exact, because this project's real calibration accuracy (historically
~86-180px mean error in validation runs) is large relative to a typical
OCR text-line block's height (~15-30px): a gaze point that's "really"
inside a block will often be reported a bit outside it -- in the gap
between lines, or briefly on/off a neighboring block for a single noisy
frame. Two things follow from that:

  - _nearest_block() replaces exact rectangle containment with
    nearest-block-within-HIT_TEST_MAX_DISTANCE_PX, so a point just outside
    a block's edge (or sitting in an inter-line gap) still counts as "on"
    the closest block rather than "on" nothing.
  - ContentGazeTracker.update() tolerates a brief single-tick divergence
    away from the block currently being dwelled on (a GAP_GRACE_MS grace
    period) before it commits to switching targets and resetting dwell
    progress -- mirroring the same grace-period idiom AttentionGate uses
    for head-pose pauses -- so one noisy frame doesn't wipe out dwell
    progress that was otherwise real.

Without both of these, a block (in particular the lowest one, which
has_reached_end() depends on) rarely accumulates its required dwell of
uninterrupted exact-containment, so "seen" -- and therefore scrolling --
becomes unreliable in practice even when gaze genuinely did read to the
end of the content.

How long a dwell is actually required before a block counts as "seen" is
itself not a flat constant by default: it scales with the block's own
width (SEEN_DWELL_MIN_MS/_MS_PER_PX/_MAX_MS below), on the theory that a
longer line simply has more text to actually read than a short one, so
treating them identically means either shortchanging long lines (letting
a glance across them count as "read") or making short ones annoyingly
slow. A caller can still pass a flat seen_dwell_ms to ContentGazeTracker
to opt back into a single fixed dwell for every block regardless of size
(this project's own tests mostly do, for simpler/more direct timing
assertions).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from doomscroll.content.base import ContentBlock, ContentScanner
from doomscroll.windowing.base import WindowBounds

RESCAN_INTERVAL_S = 0.4
# Default (size-dependent) "seen" dwell requirement: required_ms = clamp(
# SEEN_DWELL_MIN_MS + SEEN_DWELL_MS_PER_PX * block.width, ..., SEEN_DWELL_MAX_MS).
# The per-px rate is a rough, hand-picked stand-in for reading speed, not
# derived from anything -- a round number chosen to make a full-width
# paragraph line take meaningfully longer than a short one, tune freely.
SEEN_DWELL_MIN_MS = 150  # floor -- applies even to a very narrow block
SEEN_DWELL_MS_PER_PX = 0.8  # extra required dwell per pixel of the block's own width
SEEN_DWELL_MAX_MS = 1500  # cap, so an unusually wide block doesn't require an unreasonable wait
END_OF_CONTENT_MARGIN_PX = 20  # how far past the lowest block's bottom edge gaze must sit
HIT_TEST_MAX_DISTANCE_PX = 35  # how far outside a block's own rect still counts as "on" it
GAP_GRACE_MS = 150  # how long a divergence from the current block must persist before it resets dwell


def _block_id(block: ContentBlock) -> str:
    return block.text.strip().lower()


def _distance_to_block(block: ContentBlock, point: np.ndarray) -> float:
    """0 if point is inside (or on the edge of) block's rect; otherwise the
    straight-line distance from point to the nearest edge/corner of it."""
    px, py = point[0], point[1]
    dx = max(block.x - px, 0.0, px - (block.x + block.width))
    dy = max(block.y - py, 0.0, py - (block.y + block.height))
    return float((dx * dx + dy * dy) ** 0.5)


def _nearest_block(
    blocks: list[ContentBlock], point: np.ndarray, max_distance_px: float
) -> ContentBlock | None:
    """The closest block to point, provided it's within max_distance_px --
    exact containment (distance 0) always wins ties naturally since it's
    the smallest possible distance. Returns None if no block is within
    range (including when there are no blocks at all)."""
    best: ContentBlock | None = None
    best_distance: float | None = None
    for block in blocks:
        distance = _distance_to_block(block, point)
        if distance <= max_distance_px and (best_distance is None or distance < best_distance):
            best = block
            best_distance = distance
    return best


def _lowest_block(blocks: list[ContentBlock]) -> ContentBlock | None:
    if not blocks:
        return None
    return max(blocks, key=lambda b: b.bottom)


class ContentGazeTracker:
    def __init__(
        self,
        seen_dwell_ms: int | None = None,
        seen_dwell_min_ms: float = SEEN_DWELL_MIN_MS,
        seen_dwell_ms_per_px: float = SEEN_DWELL_MS_PER_PX,
        seen_dwell_max_ms: float = SEEN_DWELL_MAX_MS,
        end_margin_px: float = END_OF_CONTENT_MARGIN_PX,
        hit_max_distance_px: float = HIT_TEST_MAX_DISTANCE_PX,
        gap_grace_ms: float = GAP_GRACE_MS,
    ):
        # seen_dwell_ms, if given, is a flat dwell requirement applied to
        # every block regardless of its size -- an explicit opt-out of the
        # default size-dependent behavior (see _required_dwell_ms), handy
        # for tests that want simple, size-independent timing.
        self._seen_dwell_ms = seen_dwell_ms
        self._seen_dwell_min_ms = seen_dwell_min_ms
        self._seen_dwell_ms_per_px = seen_dwell_ms_per_px
        self._seen_dwell_max_ms = seen_dwell_max_ms
        self._end_margin_px = end_margin_px
        self._hit_max_distance_px = hit_max_distance_px
        self._gap_grace_ms = gap_grace_ms
        self._seen_ids: set[str] = set()
        self._current_id: str | None = None
        self._current_since_ms: float | None = None
        # Set while hit_id differs from _current_id but we're still within
        # the grace window -- i.e. a divergence is in progress but not yet
        # committed. None means "no divergence currently in progress".
        self._gap_since_ms: float | None = None

    @property
    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._seen_ids)

    @property
    def end_margin_px(self) -> float:
        return self._end_margin_px

    def _required_dwell_ms(self, block: ContentBlock) -> float:
        """How long gaze must dwell on `block` before it counts as "seen".
        The flat seen_dwell_ms override, if one was given, always wins;
        otherwise this scales with the block's own width, floored and
        capped -- see SEEN_DWELL_MIN_MS/_MS_PER_PX/_MAX_MS."""
        if self._seen_dwell_ms is not None:
            return self._seen_dwell_ms
        return min(self._seen_dwell_max_ms, self._seen_dwell_min_ms + self._seen_dwell_ms_per_px * block.width)

    def update(self, blocks: list[ContentBlock], gaze_point: np.ndarray, timestamp_ms: float) -> None:
        hit = _nearest_block(blocks, gaze_point, self._hit_max_distance_px)
        hit_id = _block_id(hit) if hit is not None else None

        if hit_id == self._current_id:
            # Still tracking the same target (this also covers "on
            # nothing" both before and after, when both are None) -- any
            # divergence that was in progress didn't stick, so drop it.
            self._gap_since_ms = None
            if hit_id is not None and self._current_since_ms is not None:
                if timestamp_ms - self._current_since_ms >= self._required_dwell_ms(hit):
                    self._seen_ids.add(hit_id)
        elif self._current_id is None:
            # No existing dwell progress to protect -- commit immediately,
            # there's nothing a grace period would be guarding here.
            self._current_id = hit_id
            self._current_since_ms = timestamp_ms
            self._gap_since_ms = None
        else:
            # Diverging away from a target we have real progress on.
            # Don't discard that progress over a single noisy tick --
            # only commit the switch once the divergence has itself
            # persisted past the grace period.
            if self._gap_since_ms is None:
                self._gap_since_ms = timestamp_ms
            elif timestamp_ms - self._gap_since_ms >= self._gap_grace_ms:
                self._current_id = hit_id
                self._current_since_ms = timestamp_ms
                self._gap_since_ms = None

        current_ids = {_block_id(b) for b in blocks}
        self._seen_ids &= current_ids

    def has_reached_end(self, blocks: list[ContentBlock], gaze_point: np.ndarray) -> bool:
        lowest = _lowest_block(blocks)
        if lowest is None:
            return False
        if _block_id(lowest) not in self._seen_ids:
            return False
        return bool(gaze_point[1] >= lowest.bottom - self._end_margin_px)

    def should_scroll(self, blocks: list[ContentBlock], gaze_point: np.ndarray, viewport_bottom: float) -> bool:
        for block in blocks:
            if block.center_y <= gaze_point[1]:
                continue
            if block.y >= viewport_bottom:
                continue
            if _block_id(block) not in self._seen_ids:
                return False
        return True


@dataclass
class _ScanState:
    blocks: list[ContentBlock] = field(default_factory=list)
    scanned_at_ms: float = 0.0


class BackgroundContentScanner:
    def __init__(self, scanner: ContentScanner, get_bounds, interval_s: float = RESCAN_INTERVAL_S):
        self._scanner = scanner
        self._get_bounds = get_bounds
        self._interval_s = interval_s
        self._state = _ScanState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="content-scanner")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_blocks(self) -> list[ContentBlock]:
        with self._lock:
            return list(self._state.blocks)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            bounds: WindowBounds | None = self._get_bounds()
            if bounds is not None:
                try:
                    blocks = self._scanner.scan(bounds)
                except Exception:
                    blocks = self._state.blocks
                with self._lock:
                    self._state = _ScanState(blocks=blocks, scanned_at_ms=time.time() * 1000)
            self._stop_event.wait(self._interval_s)
