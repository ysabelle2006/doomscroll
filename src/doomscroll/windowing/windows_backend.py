# Windows implementation of WindowBoundsProvider using win32gui.
from __future__ import annotations

import win32gui

from doomscroll.windowing.base import WindowBounds, WindowBoundsProvider


# Reports the foreground window's rect and title via the Win32 API.
class WindowsWindowBoundsProvider(WindowBoundsProvider):
    # Returns the foreground window's bounds, or None if there is none.
    def get_focused_window_bounds(self) -> WindowBounds | None:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        title = win32gui.GetWindowText(hwnd) or "unknown"

        return WindowBounds(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
            app_name=title,
        )
