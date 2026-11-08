# Abstract interface for injecting scroll events into the OS.
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


# Unit a scroll amount is expressed in.
class ScrollUnit(Enum):
    LINE = auto()
    PIXEL = auto()


# Platform backends implement this to send scroll input.
class ScrollInjector(ABC):
    # Scrolls by dx/dy in the given unit, optionally at a target point.
    @abstractmethod
    def scroll(
        self,
        dx: int,
        dy: int,
        unit: ScrollUnit = ScrollUnit.LINE,
        target_point: tuple[int, int] | None = None,
    ) -> None:
        pass
