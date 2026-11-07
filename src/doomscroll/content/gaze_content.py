# Tracks which content blocks the user has looked at, and runs background OCR scans.
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from doomscroll.content.base import ContentBlock, ContentScanner
from doomscroll.windowing.base import WindowBounds

# Timing/distance thresholds for dwell-based "seen" detection and scan rate.
RESCAN_INTERVAL_S = 0.4
SEEN_DWELL_MIN_MS = 150
SEEN_DWELL_MS_PER_PX = 0.5
SEEN_DWELL_MAX_MS = 1500
END_OF_CONTENT_MARGIN_PX = 20
HIT_TEST_MAX_DISTANCE_PX = 35
GAP_GRACE_MS = 500


# Normalizes a block's text into a stable id for tracking.
def _block_id(block: ContentBlock) -> str:
    return block.text.strip().lower()


# Euclidean distance from a point to a block's bounding box (0 if inside).
def _distance_to_block(block: ContentBlock, point: np.ndarray) -> float:
    px, py = point[0], point[1]
    dx = max(block.x - px, 0.0, px - (block.x + block.width))
    dy = max(block.y - py, 0.0, py - (block.y + block.height))
    return float((dx * dx + dy * dy) ** 0.5)


# Finds the closest block to a point within max_distance_px, or None.
def _nearest_block(
    blocks: list[ContentBlock], point: np.ndarray, max_distance_px: float
) -> ContentBlock | None:
    best: ContentBlock | None = None
    best_distance: float | None = None
    for block in blocks:
        distance = _distance_to_block(block, point)
        if distance <= max_distance_px and (best_distance is None or distance < best_distance):
            best = block
            best_distance = distance
    return best


# Returns the block with the lowest bottom edge, or None if empty.
def _lowest_block(blocks: list[ContentBlock]) -> ContentBlock | None:
    if not blocks:
        return None
    return max(blocks, key=lambda b: b.bottom)


# Tracks dwell time on content blocks to decide which have been "seen" and when to scroll.
class ContentGazeTracker:
    # Stores dwell/distance/grace thresholds and resets tracking state.
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
        self._gap_since_ms: float | None = None

    # Ids of blocks marked as seen so far.
    @property
    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._seen_ids)

    # Margin from the viewport bottom counted as having reached the end.
    @property
    def end_margin_px(self) -> float:
        return self._end_margin_px

    # Dwell time required before a block counts as seen, scaled by its width.
    def _required_dwell_ms(self, block: ContentBlock) -> float:
        if self._seen_dwell_ms is not None:
            return self._seen_dwell_ms
        return min(self._seen_dwell_max_ms, self._seen_dwell_min_ms + self._seen_dwell_ms_per_px * block.width)

    # Updates the currently-gazed block and marks it seen once dwell time is met.
    def update(self, blocks: list[ContentBlock], gaze_point: np.ndarray, timestamp_ms: float) -> None:
        hit = _nearest_block(blocks, gaze_point, self._hit_max_distance_px)
        hit_id = _block_id(hit) if hit is not None else None

        if hit_id == self._current_id:
            self._gap_since_ms = None
            if hit_id is not None and self._current_since_ms is not None:
                if timestamp_ms - self._current_since_ms >= self._required_dwell_ms(hit):
                    self._seen_ids.add(hit_id)
        elif self._current_id is None:
            self._current_id = hit_id
            self._current_since_ms = timestamp_ms
            self._gap_since_ms = None
        else:
            if self._gap_since_ms is None:
                self._gap_since_ms = timestamp_ms
            elif timestamp_ms - self._gap_since_ms >= self._gap_grace_ms:
                self._current_id = hit_id
                self._current_since_ms = timestamp_ms
                self._gap_since_ms = None

        current_ids = {_block_id(b) for b in blocks}
        self._seen_ids &= current_ids

    # Whether the gaze has reached the bottom of the lowest seen block.
    def has_reached_end(self, blocks: list[ContentBlock], gaze_point: np.ndarray) -> bool:
        lowest = _lowest_block(blocks)
        if lowest is None:
            return False
        if _block_id(lowest) not in self._seen_ids:
            return False
        return bool(gaze_point[1] >= lowest.bottom - self._end_margin_px)

    # Whether every block above the gaze point within the viewport has been seen.
    def should_scroll(self, blocks: list[ContentBlock], gaze_point: np.ndarray, viewport_bottom: float) -> bool:
        for block in blocks:
            if block.center_y <= gaze_point[1]:
                continue
            if block.y >= viewport_bottom:
                continue
            if _block_id(block) not in self._seen_ids:
                return False
        return True


# Latest scanned content blocks and when they were scanned.
@dataclass
class _ScanState:
    blocks: list[ContentBlock] = field(default_factory=list)
    scanned_at_ms: float = 0.0


# Periodically rescans on-screen content on a background thread.
class BackgroundContentScanner:
    # Stores the scanner/bounds-getter and prepares initial scan state.
    def __init__(
        self,
        scanner: ContentScanner,
        get_bounds,
        interval_s: float = RESCAN_INTERVAL_S,
        get_exclude_window_ids: "Callable[[], frozenset[int]] | None" = None,
    ):
        self._scanner = scanner
        self._get_bounds = get_bounds
        self._interval_s = interval_s
        self._get_exclude_window_ids = get_exclude_window_ids
        self._state = _ScanState()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # Starts the background scanning thread if not already running.
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="content-scanner")
        self._thread.start()

    # Stops the background scanning thread.
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # Returns the most recently scanned content blocks.
    def latest_blocks(self) -> list[ContentBlock]:
        with self._lock:
            return list(self._state.blocks)

    # Loop: rescans content blocks at a fixed interval, keeping old blocks on scan failure.
    def _run(self) -> None:
        while not self._stop_event.is_set():
            bounds: WindowBounds | None = self._get_bounds()
            if bounds is not None:
                exclude_window_ids = (
                    self._get_exclude_window_ids() if self._get_exclude_window_ids is not None else frozenset()
                )
                try:
                    blocks = self._scanner.scan(bounds, exclude_window_ids=exclude_window_ids)
                except Exception:
                    blocks = self._state.blocks
                with self._lock:
                    self._state = _ScanState(blocks=blocks, scanned_at_ms=time.time() * 1000)
            self._stop_event.wait(self._interval_s)
