"""Common interface for synthetic OS-level scroll injection, per-OS."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto


class ScrollUnit(Enum):
    LINE = auto()
    PIXEL = auto()


class ScrollInjector(ABC):
    @abstractmethod
    def scroll(
        self,
        dx: int,
        dy: int,
        unit: ScrollUnit = ScrollUnit.LINE,
        target_point: tuple[int, int] | None = None,
    ) -> None:
        """Inject a synthetic scroll event. Positive dy scrolls down,
        positive dx scrolls right, matching natural reading-direction sign
        convention used elsewhere in this project.

        target_point, if given, is a screen-pixel point the event should be
        targeted at (e.g. the focused window's center) -- both OS backends
        briefly move the cursor there and restore it afterward, since wheel
        events are delivered based on cursor position, not window focus."""
