# Abstract interface for scanning on-screen text content.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from doomscroll.windowing.base import WindowBounds


# A block of on-screen text with its bounding box.
@dataclass(frozen=True)
class ContentBlock:

    text: str
    x: int
    y: int
    width: int
    height: int

    # Vertical center of the block.
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    # Bottom edge y-coordinate of the block.
    @property
    def bottom(self) -> float:
        return self.y + self.height


# Platform backends implement this to find text blocks on screen.
class ContentScanner(ABC):
    # Scans the window for content blocks, excluding given window ids.
    @abstractmethod
    def scan(self, bounds: WindowBounds, exclude_window_ids: frozenset[int] = frozenset()) -> list[ContentBlock]:
        pass
