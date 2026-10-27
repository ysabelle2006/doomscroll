"""Common interface for on-screen text/content scanning, per-OS -- the OCR
layer that gives gaze tracking context on *what* the user is looking at,
not just *where* on screen. See gaze_content.py for how recognized blocks
feed into the actual scroll decision."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from doomscroll.windowing.base import WindowBounds


@dataclass(frozen=True)
class ContentBlock:
    """One recognized piece of on-screen text, in absolute screen-pixel
    coordinates -- the same coordinate space WindowBounds and gaze points
    use everywhere else in this project (top-left origin, not normalized,
    not window-relative)."""

    text: str
    x: int
    y: int
    width: int
    height: int

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def bottom(self) -> float:
        return self.y + self.height


class ContentScanner(ABC):
    @abstractmethod
    def scan(self, bounds: WindowBounds) -> list[ContentBlock]:
        """Returns recognized text blocks within `bounds`, in absolute
        screen-pixel coordinates. Screen capture + OCR is too slow to call
        every camera frame -- callers should go through
        gaze_content.BackgroundContentScanner, which re-scans on its own
        interval on a background thread, rather than calling this directly
        from the live gaze loop."""
