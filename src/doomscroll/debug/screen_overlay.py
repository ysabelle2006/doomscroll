"""A live, click-through, transparent screen overlay (macOS-only) that
draws the OCR boxes + current gaze dot directly on top of the real
content you're reading, updating every tick -- instead of a separate
window showing a static screenshot, which is what this replaced.

Deliberately excluded from feeding back into the pipeline it's debugging,
two different ways:

  - setSharingType_(NSWindowSharingNone) tells the window server to leave
    this window out of any screen capture it composites for other
    callers -- including the mss-based screenshot content/macos_backend.py
    takes for OCR. Without this, the overlay's own boxes would show up in
    the next OCR scan, which would then try to recognize text inside
    them, feeding the debug view back into the thing it's supposed to be
    observing.
  - It's click-through (setIgnoresMouseEvents_(True)) and never becomes
    the key/main window or activates its own app, so it can never cause
    this process to become the "frontmost application" -- which is
    exactly what windowing/macos_backend.py's get_focused_window_bounds()
    keys off of (frontmost app -> that app's on-screen windows). So it
    can never get mistaken for the window you're actually looking at,
    the way a clickable debug window could.

Not exercised on the Linux dev machine this was written on -- same
caveat as this project's other macOS-only backends (content/macos_backend.py,
windowing/macos_backend.py): the AppKit API surface here (NSWindow/NSView
construction by hand, window levels, the sharing-type exclusion, the
Quartz<->AppKit coordinate flip) is unverified until it actually runs on
macOS. Every call into this from phase2_gaze_scroll.py is wrapped in a
try/except there, so if something about the window itself misbehaves
(doesn't appear, doesn't move/update, throws), that's isolated to this
module and degrades to "no overlay" rather than taking down gaze tracking.
"""
from __future__ import annotations

from doomscroll.content.base import ContentBlock
from doomscroll.windowing.base import WindowBounds

try:
    import objc
    from AppKit import (
        NSApplication,
        NSBezierPath,
        NSColor,
        NSMakeRect,
        NSScreen,
        NSView,
        NSWindow,
        NSWindowSharingNone,
    )

    _OVERLAY_AVAILABLE = True
except ImportError:
    _OVERLAY_AVAILABLE = False

# NSBackingStoreBuffered's documented value -- hardcoded rather than
# imported. These old AppKit integer constants are stable, unchanged-for-
# decades API, but whether a given pyobjc version happens to bind the
# *name* at the top level of AppKit has been inconsistent across versions
# in the wild; the raw value isn't.
_NS_BACKING_STORE_BUFFERED = 2
# NSScreenSaverWindowLevel's documented value -- high enough to float
# above ordinary application windows. Same hardcoding rationale as above.
_NS_SCREEN_SAVER_WINDOW_LEVEL = 1000


if _OVERLAY_AVAILABLE:

    class _OverlayView(NSView):
        """Owns the actual drawing. State is pushed in from outside via
        plain attribute assignment (see ScreenOverlay.update) rather than
        through drawRect_'s own arguments, which only ever gets the dirty
        rect, not application state."""

        def initWithFrame_(self, frame):
            self = objc.super(_OverlayView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.ov_blocks: list[ContentBlock] = []
            self.ov_gaze_xy: tuple[float, float] | None = None
            self.ov_seen_ids: frozenset[str] = frozenset()
            self.ov_end_margin_px: float = 20.0
            # This window's own top-left corner, in the same absolute
            # screen-pixel space `ov_blocks`/`ov_gaze_xy` are given in --
            # every draw call below subtracts this back out to get
            # view-local coordinates.
            self.ov_origin_xy: tuple[int, int] = (0, 0)
            return self

        def isFlipped(self) -> bool:
            # Top-left origin, y increasing downward -- matches this
            # project's screen-pixel convention everywhere else
            # (WindowBounds, ContentBlock, gaze points), so none of the
            # drawing below needs its own flip math.
            return True

        def isOpaque(self) -> bool:
            return False

        def drawRect_(self, rect) -> None:
            NSColor.clearColor().set()
            NSBezierPath.fillRect_(rect)

            blocks = self.ov_blocks
            if not blocks:
                return
            ox, oy = self.ov_origin_xy
            lowest = max(blocks, key=lambda b: b.bottom)

            for block in blocks:
                x0, y0 = block.x - ox, block.y - oy
                box = NSMakeRect(x0, y0, block.width, block.height)
                path = NSBezierPath.bezierPathWithRect_(box)
                if block is lowest:
                    color, width = NSColor.colorWithRed_green_blue_alpha_(0.0, 0.47, 1.0, 1.0), 2.5
                elif block.text.strip().lower() in self.ov_seen_ids:
                    color, width = NSColor.colorWithRed_green_blue_alpha_(0.0, 0.8, 0.0, 1.0), 1.5
                else:
                    color, width = NSColor.colorWithRed_green_blue_alpha_(1.0, 0.0, 0.0, 1.0), 1.5
                path.setLineWidth_(width)
                color.set()
                path.stroke()

            threshold_y = lowest.bottom - oy - self.ov_end_margin_px
            line = NSBezierPath.bezierPath()
            line.moveToPoint_((0, threshold_y))
            line.lineToPoint_((self.bounds().size.width, threshold_y))
            line.setLineWidth_(1.0)
            NSColor.colorWithRed_green_blue_alpha_(1.0, 1.0, 0.0, 1.0).set()
            line.stroke()

            if self.ov_gaze_xy is not None:
                gx, gy = self.ov_gaze_xy[0] - ox, self.ov_gaze_xy[1] - oy
                r = 8
                dot = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(gx - r, gy - r, r * 2, r * 2))
                NSColor.colorWithRed_green_blue_alpha_(1.0, 0.0, 1.0, 0.9).set()
                dot.fill()
                dot.setLineWidth_(1.0)
                NSColor.whiteColor().set()
                dot.stroke()

else:
    _OverlayView = None  # Referenced only when _OVERLAY_AVAILABLE is True.


class ScreenOverlay:
    """Lazily creates a single transparent overlay window sized to the
    focused window's bounds, and repositions/redraws it on demand.
    Construction alone draws nothing -- call update() each tick with the
    current blocks/bounds/gaze point; the window is created on first use
    and moved/resized whenever bounds change (window switch, or resize)."""

    def __init__(self):
        if not _OVERLAY_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Cocoa is required on macOS for the live content "
                "overlay. Install the 'macos' extra: pip install .[macos]"
            )
        self._window: NSWindow | None = None
        self._view: "_OverlayView | None" = None

    def _ensure_window(self, bounds: WindowBounds) -> None:
        NSApplication.sharedApplication()
        screen = NSScreen.mainScreen()
        # Quartz/CGWindow bounds (what `bounds` is, same as everywhere else
        # in this project) are top-left-origin, y-down, in the global
        # display coordinate space. AppKit's own window-frame coordinates
        # are bottom-left-origin, y-up, in that same global space. Flipping
        # against the *main* screen's height is the standard conversion for
        # this -- it can misplace things on an unusual multi-monitor setup
        # where the main screen isn't the one being read from, but that's
        # untested here, same as the rest of this module.
        screen_h = screen.frame().size.height if screen is not None else bounds.y + bounds.height
        appkit_y = screen_h - bounds.y - bounds.height
        frame = NSMakeRect(bounds.x, appkit_y, bounds.width, bounds.height)

        if self._window is None:
            self._view = _OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, bounds.width, bounds.height))
            self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                frame, 0, _NS_BACKING_STORE_BUFFERED, False
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(NSColor.clearColor())
            self._window.setLevel_(_NS_SCREEN_SAVER_WINDOW_LEVEL)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setSharingType_(NSWindowSharingNone)
            self._window.setContentView_(self._view)
            self._window.orderFrontRegardless()
        else:
            self._window.setFrame_display_(frame, False)
            self._view.setFrame_(NSMakeRect(0, 0, bounds.width, bounds.height))

    def update(
        self,
        blocks: list[ContentBlock],
        bounds: WindowBounds,
        gaze_point_abs,
        seen_ids: frozenset[str],
        end_margin_px: float,
    ) -> None:
        self._ensure_window(bounds)
        self._view.ov_blocks = blocks
        self._view.ov_gaze_xy = (float(gaze_point_abs[0]), float(gaze_point_abs[1]))
        self._view.ov_seen_ids = seen_ids
        self._view.ov_end_margin_px = end_margin_px
        self._view.ov_origin_xy = (bounds.x, bounds.y)
        self._view.setNeedsDisplay_(True)
        self._window.displayIfNeeded()

    def hide(self) -> None:
        if self._window is not None:
            self._window.orderOut_(None)

    def close(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None
            self._view = None
