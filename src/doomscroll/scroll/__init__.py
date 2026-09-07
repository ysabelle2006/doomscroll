from __future__ import annotations

import sys

from doomscroll.scroll.base import ScrollInjector, ScrollUnit


def get_scroll_injector() -> ScrollInjector:
    if sys.platform == "win32":
        from doomscroll.scroll.windows_backend import WindowsScrollInjector

        return WindowsScrollInjector()
    if sys.platform == "darwin":
        from doomscroll.scroll.macos_backend import MacOSScrollInjector

        return MacOSScrollInjector()
    raise NotImplementedError(f"No ScrollInjector for platform: {sys.platform}")


__all__ = ["ScrollInjector", "ScrollUnit", "get_scroll_injector"]
