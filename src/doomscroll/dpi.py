# Windows DPI-awareness setup and cross-platform screen size lookup.
from __future__ import annotations

import sys


# Marks the process as DPI-aware on Windows so coordinates aren't scaled.
def ensure_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# Returns the main display's pixel size, via Quartz on macOS or Tk elsewhere.
def get_screen_size() -> tuple[int, int]:
    if sys.platform == "darwin":
        try:
            import Quartz

            display_id = Quartz.CGMainDisplayID()
            w = Quartz.CGDisplayPixelsWide(display_id)
            h = Quartz.CGDisplayPixelsHigh(display_id)
            if w and h:
                return int(w), int(h)
        except Exception:
            pass

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h
