# Windows implementation of ScrollInjector using win32api mouse events.
from __future__ import annotations

import win32api
import win32con

from doomscroll.scroll.base import ScrollInjector, ScrollUnit

# Wheel-delta and pixel-to-line conversion constants for Win32 scroll events.
_WHEEL_DELTA = 120
_PIXELS_PER_LINE = 40


# Sends scroll wheel events via the Win32 API, optionally at a target point.
class WindowsScrollInjector(ScrollInjector):
    # Moves the cursor to target_point (if given), scrolls, then restores it.
    def scroll(
        self,
        dx: int,
        dy: int,
        unit: ScrollUnit = ScrollUnit.LINE,
        target_point: tuple[int, int] | None = None,
    ) -> None:
        original_pos = win32api.GetCursorPos()
        try:
            if target_point is not None:
                win32api.SetCursorPos(target_point)

            if dy != 0:
                amount = dy if unit == ScrollUnit.LINE else dy / _PIXELS_PER_LINE
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, int(-amount * _WHEEL_DELTA), 0)

            if dx != 0:
                amount = dx if unit == ScrollUnit.LINE else dx / _PIXELS_PER_LINE
                win32api.mouse_event(win32con.MOUSEEVENTF_HWHEEL, 0, 0, int(amount * _WHEEL_DELTA), 0)
        finally:
            if target_point is not None:
                win32api.SetCursorPos(original_pos)
