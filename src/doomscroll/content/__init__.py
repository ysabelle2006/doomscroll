from __future__ import annotations

import sys

from doomscroll.content.base import ContentBlock, ContentScanner


def get_content_scanner() -> ContentScanner:
    if sys.platform == "darwin":
        from doomscroll.content.macos_backend import MacOSVisionContentScanner

        return MacOSVisionContentScanner()
    # No Windows OCR backend yet (would likely use the Windows.Media.Ocr
    # WinRT API, mirroring how windowing/scroll split per-OS) -- callers
    # should catch NotImplementedError and fall back to position-only
    # scrolling, same as a missing pyobjc install on macOS.
    raise NotImplementedError(f"No ContentScanner for platform: {sys.platform}")


__all__ = ["ContentBlock", "ContentScanner", "get_content_scanner"]
