"""Common interface for focused-window-bounds retrieval, per-OS."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WindowBounds:
    x: int
    y: int
    width: int
    height: int
    app_name: str


class WindowBoundsProvider(ABC):
    @abstractmethod
    def get_focused_window_bounds(self) -> WindowBounds | None:
        """Bounds of the frontmost/focused window, or None if unavailable."""
