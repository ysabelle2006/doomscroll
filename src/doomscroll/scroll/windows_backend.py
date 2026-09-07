"""Windows synthetic scroll injection via win32api mouse_event.

Wheel events on Windows are delivered to whatever is under the cursor, not
necessarily the "focused" window, so scroll() briefly moves the cursor to the
target point, posts the wheel event, then restores the original cursor
position. This is a known per-app risk noted in the project plan -- validate
against real target apps (browsers, PDF readers) rather than assuming it
always works identically.
"""
from __future__ import annotations

import win32api
import win32con

from doomscroll.scroll.base import ScrollInjector, ScrollUnit

_WHEEL_DELTA = 120  # one notch, per Win32 convention
_PIXELS_PER_LINE = 40  # approximate line height for PIXEL-unit conversion


class WindowsScrollInjector(ScrollInjector):
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
                # Win32 convention: positive wheelDelta scrolls up (content
                # moves down); negate so positive dy means "scroll down".
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, int(-amount * _WHEEL_DELTA), 0)

            if dx != 0:
                amount = dx if unit == ScrollUnit.LINE else dx / _PIXELS_PER_LINE
                win32api.mouse_event(win32con.MOUSEEVENTF_HWHEEL, 0, 0, int(amount * _WHEEL_DELTA), 0)
        finally:
            if target_point is not None:
                win32api.SetCursorPos(original_pos)
