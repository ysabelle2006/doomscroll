"""macOS synthetic scroll injection via Quartz CGEventCreateScrollWheelEvent.

Not exercised on the Windows dev machine -- validate on macOS before relying
on it, per the project plan's known-risks list (event targeting and
per-app handling of synthetic scroll events varies).
"""
from __future__ import annotations

from doomscroll.scroll.base import ScrollInjector, ScrollUnit

try:
    from Quartz import (
        CGEventCreate,
        CGEventCreateScrollWheelEvent,
        CGEventGetLocation,
        CGEventPost,
        CGWarpMouseCursorPosition,
        kCGHIDEventTap,
        kCGScrollEventUnitLine,
        kCGScrollEventUnitPixel,
    )

    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False


class MacOSScrollInjector(ScrollInjector):
    def __init__(self):
        if not _QUARTZ_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Quartz is required on macOS -- it's a base "
                "dependency there, so `uv sync` / `pip install -e .` should "
                "already have installed it; try re-running that."
            )

    def scroll(
        self,
        dx: int,
        dy: int,
        unit: ScrollUnit = ScrollUnit.LINE,
        target_point: tuple[int, int] | None = None,
    ) -> None:
        # Scroll wheel events are delivered based on cursor position, not
        # window focus, so (as on Windows) briefly warp the cursor to the
        # target and back rather than leave it wherever it was.
        original_point = None
        if target_point is not None:
            original_point = CGEventGetLocation(CGEventCreate(None))
            CGWarpMouseCursorPosition(target_point)

        try:
            cg_unit = kCGScrollEventUnitLine if unit == ScrollUnit.LINE else kCGScrollEventUnitPixel
            # CGEventCreateScrollWheelEvent axis convention: positive vertical
            # delta scrolls up (content moves down); negate so positive dy
            # means "scroll down", matching this project's sign convention.
            event = CGEventCreateScrollWheelEvent(None, cg_unit, 2, -dy, dx)
            CGEventPost(kCGHIDEventTap, event)
        finally:
            if original_point is not None:
                CGWarpMouseCursorPosition(original_point)
