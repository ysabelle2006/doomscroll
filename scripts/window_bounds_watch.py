"""Phase 1 validation: prints the focused window's bounds on a timer, so you
can switch/resize/full-screen windows and confirm bounds update correctly
(the requirement that this work at full-screen and at partial window sizes).

Run: .venv\\Scripts\\python.exe scripts\\window_bounds_watch.py
Ctrl+C to stop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.dpi import ensure_dpi_aware
from doomscroll.windowing import get_window_bounds_provider


def main() -> None:
    ensure_dpi_aware()
    provider = get_window_bounds_provider()
    print("Watching focused window bounds. Ctrl+C to stop.")
    try:
        while True:
            bounds = provider.get_focused_window_bounds()
            if bounds is None:
                print("No focused window bounds available.")
            else:
                print(
                    f"[{bounds.app_name!r}] x={bounds.x} y={bounds.y} "
                    f"w={bounds.width} h={bounds.height}"
                )
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
