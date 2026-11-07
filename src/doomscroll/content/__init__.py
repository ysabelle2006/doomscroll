# Picks the platform-specific ContentScanner implementation.
from __future__ import annotations

import sys

from doomscroll.content.base import ContentBlock, ContentScanner


# Returns a ContentScanner for the current OS, or raises if unsupported.
def get_content_scanner() -> ContentScanner:
    if sys.platform == "darwin":
        from doomscroll.content.macos_backend import MacOSVisionContentScanner

        return MacOSVisionContentScanner()
    raise NotImplementedError(f"No ContentScanner for platform: {sys.platform}")


__all__ = ["ContentBlock", "ContentScanner", "get_content_scanner"]
