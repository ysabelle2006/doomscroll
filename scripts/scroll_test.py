"""Phase 1 validation: keypress-triggered synthetic scroll against whatever
app is currently focused. This is the highest-risk integration point in the
plan (event targeting, per-app handling of synthetic input) -- run this with
a real target app (browser, PDF reader, etc.) focused and confirm it
actually scrolls before building anything on top of it.

Run: .venv\\Scripts\\python.exe scripts\\scroll_test.py
Then click into a browser/PDF/other scrollable app to focus it, and press:
  d = scroll down   u = scroll up   l = scroll left   r = scroll right
  q = quit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if sys.platform != "win32":
    print("This script's keypress handling uses msvcrt and is Windows-only. "
          "On macOS, drive doomscroll.scroll.get_scroll_injector() from your "
          "own test harness instead.")
    sys.exit(1)

import msvcrt

from doomscroll.dpi import ensure_dpi_aware
from doomscroll.scroll import ScrollUnit, get_scroll_injector
from doomscroll.windowing import get_window_bounds_provider

STEP = 6  # lines per keypress


def main() -> None:
    ensure_dpi_aware()
    injector = get_scroll_injector()
    bounds_provider = get_window_bounds_provider()

    print(__doc__)
    print("Ready. Waiting for keypresses...")

    while True:
        key = msvcrt.getch().decode(errors="ignore").lower()
        if key == "q":
            break

        bounds = bounds_provider.get_focused_window_bounds()
        target_point = None
        if bounds is not None:
            target_point = (bounds.x + bounds.width // 2, bounds.y + bounds.height // 2)
            print(f"Targeting {bounds.app_name!r} at {target_point}")

        if key == "d":
            injector.scroll(0, STEP, ScrollUnit.LINE, target_point=target_point)
        elif key == "u":
            injector.scroll(0, -STEP, ScrollUnit.LINE, target_point=target_point)
        elif key == "l":
            injector.scroll(-STEP, 0, ScrollUnit.LINE, target_point=target_point)
        elif key == "r":
            injector.scroll(STEP, 0, ScrollUnit.LINE, target_point=target_point)
        else:
            continue

        print(f"Sent scroll for key {key!r}")


if __name__ == "__main__":
    main()
