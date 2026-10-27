"""macOS focused-window-bounds via Quartz. Requires Screen Recording and
Accessibility permissions granted to the running process (System Settings ->
Privacy & Security). Not exercised on the Windows dev machine — validate on
macOS before relying on it."""
from __future__ import annotations

from doomscroll.windowing.base import WindowBounds, WindowBoundsProvider

try:
    from AppKit import NSWorkspace
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionOnScreenOnly,
    )

    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False


class MacOSWindowBoundsProvider(WindowBoundsProvider):
    def __init__(self):
        if not _QUARTZ_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Quartz and pyobjc-framework-Cocoa are required "
                "on macOS. Install the 'macos' extra: pip install .[macos]"
            )

    def get_focused_window_bounds(self) -> WindowBounds | None:
        frontmost_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if frontmost_app is None:
            return None
        frontmost_pid = frontmost_app.processIdentifier()

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )
        for window in window_list:
            if window.get("kCGWindowOwnerPID") != frontmost_pid:
                continue
            if window.get("kCGWindowLayer") != 0:
                continue
            bounds = window.get("kCGWindowBounds")
            if not bounds:
                continue
            return WindowBounds(
                x=int(bounds["X"]),
                y=int(bounds["Y"]),
                width=int(bounds["Width"]),
                height=int(bounds["Height"]),
                app_name=window.get("kCGWindowOwnerName", "unknown"),
            )
        return None
