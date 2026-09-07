"""Ensure per-monitor DPI awareness on Windows so every screen/window
coordinate API used in this project (tkinter, win32gui, cv2 fullscreen
window placement, cursor positioning for scroll targeting) agrees on one
true physical-pixel coordinate space.

Without this, a scaled display reports a virtualized, downscaled "screen"
to a DPI-unaware process (confirmed on the dev machine: 1600x1000 reported
vs. 3200x2000 actual physical resolution at 200% scaling) -- which silently
corrupts calibration, since the calibration targets and the runtime
window-bounds-based scroll mapping can end up expressed in different scaled
coordinate spaces. Must be called before any window/screen query (tkinter,
win32gui, cv2 window creation) happens in the process.
"""
from __future__ import annotations

import sys


def ensure_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows
        except Exception:
            pass


def get_screen_size() -> tuple[int, int]:
    """True physical screen size -- call ensure_dpi_aware() first, or this
    returns the same virtualized/scaled-down size described above."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h
