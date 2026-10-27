from __future__ import annotations

import sys

from doomscroll.windowing.base import WindowBounds, WindowBoundsProvider


def get_window_bounds_provider() -> WindowBoundsProvider:
    if sys.platform == "win32":
        from doomscroll.windowing.windows_backend import WindowsWindowBoundsProvider

        return WindowsWindowBoundsProvider()
    if sys.platform == "darwin":
        from doomscroll.windowing.macos_backend import MacOSWindowBoundsProvider

        return MacOSWindowBoundsProvider()
    raise NotImplementedError(f"No WindowBoundsProvider for platform: {sys.platform}")


__all__ = ["WindowBounds", "WindowBoundsProvider", "get_window_bounds_provider"]
