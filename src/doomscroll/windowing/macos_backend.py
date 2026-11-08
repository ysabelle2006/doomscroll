# macOS implementation of WindowBoundsProvider using AppKit/Quartz.
from __future__ import annotations

import os

from doomscroll.windowing.base import WindowBounds, WindowBoundsProvider

try:
    from AppKit import NSWorkspace
    from Quartz import (
        CGDisplayBounds,
        CGMainDisplayID,
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListOptionOnScreenOnly,
    )

    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False


# Reports the frontmost app window's bounds via AppKit/Quartz.
class MacOSWindowBoundsProvider(WindowBoundsProvider):
    # Raises if the pyobjc AppKit/Quartz frameworks are not installed.
    def __init__(self):
        if not _QUARTZ_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Quartz and pyobjc-framework-Cocoa are required "
                "on macOS -- both are base dependencies there, so `uv sync` / "
                "`pip install -e .` should already have installed them; try "
                "re-running that."
            )

    # Finds the frontmost app's on-screen window, falling back to the main display.
    def get_focused_window_bounds(self) -> WindowBounds | None:
        frontmost_app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if frontmost_app is None:
            return None
        frontmost_pid = frontmost_app.processIdentifier()

        if frontmost_pid == os.getpid():
            return None

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

        main_bounds = CGDisplayBounds(CGMainDisplayID())
        return WindowBounds(
            x=int(main_bounds.origin.x),
            y=int(main_bounds.origin.y),
            width=int(main_bounds.size.width),
            height=int(main_bounds.size.height),
            app_name=str(frontmost_app.localizedName() or "desktop"),
        )
