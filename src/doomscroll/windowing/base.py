# Abstract interface for getting the focused window's screen bounds.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# Rectangle and app name of a window on screen.
@dataclass
class WindowBounds:
    x: int
    y: int
    width: int
    height: int
    app_name: str


# Platform backends implement this to report the focused window's bounds.
class WindowBoundsProvider(ABC):
    # Returns the focused window's bounds, or None if unavailable.
    @abstractmethod
    def get_focused_window_bounds(self) -> WindowBounds | None:
        pass
