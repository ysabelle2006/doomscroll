# Transparent click-through macOS overlay window drawing content blocks and gaze dot.
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
    )

    _OVERLAY_AVAILABLE = True
except ImportError:
    _OVERLAY_AVAILABLE = False

# AppKit window-level/collection-behavior/backing-store constants for the overlay.
_NS_BACKING_STORE_BUFFERED = 2
_NS_SCREEN_SAVER_WINDOW_LEVEL = 1000
_NS_WINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES = 1 << 0
_NS_WINDOW_COLLECTION_BEHAVIOR_FULL_SCREEN_AUXILIARY = 1 << 8
_NS_WINDOW_COLLECTION_BEHAVIOR_IGNORES_CYCLE = 1 << 6


if _OVERLAY_AVAILABLE:

    # NSView that draws content block outlines, the end-of-content line, and gaze dot.
    class _OverlayView(NSView):

        # Initializes overlay drawing state (blocks, gaze point, seen ids, origin).
        def initWithFrame_(self, frame):
            self = objc.super(_OverlayView, self).initWithFrame_(frame)
            if self is None:
                return None
            self.ov_blocks: list[ContentBlock] = []
            self.ov_gaze_xy: tuple[float, float] | None = None
            self.ov_seen_ids: frozenset[str] = frozenset()
            self.ov_end_margin_px: float = 20.0
            self.ov_origin_xy: tuple[int, int] = (0, 0)
            return self

        # Flips coordinate system so y increases downward.
        def isFlipped(self) -> bool:
            return True

        # Overlay is transparent, not opaque.
        def isOpaque(self) -> bool:
            return False

        # Draws block outlines (color by seen/lowest status), the end line, and the gaze dot.
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
    _OverlayView = None


# Manages the transparent overlay window and content view on macOS.
class ScreenOverlay:

    # Raises if the pyobjc Cocoa framework is not installed.
    def __init__(self):
        if not _OVERLAY_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Cocoa is required on macOS for the live content "
                "overlay -- it's a base dependency there, so `uv sync` / "
                "`pip install -e .` should already have installed it; try "
                "re-running that."
            )
        self._window: NSWindow | None = None
        self._view: "_OverlayView | None" = None

    # Creates or repositions the overlay window to match the given window bounds.
    def _ensure_window(self, bounds: WindowBounds) -> None:
        NSApplication.sharedApplication()
        screen = NSScreen.mainScreen()
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
            self._window.setCollectionBehavior_(
                _NS_WINDOW_COLLECTION_BEHAVIOR_CAN_JOIN_ALL_SPACES
                | _NS_WINDOW_COLLECTION_BEHAVIOR_FULL_SCREEN_AUXILIARY
                | _NS_WINDOW_COLLECTION_BEHAVIOR_IGNORES_CYCLE
            )
            self._window.setContentView_(self._view)
            self._window.orderFrontRegardless()
        else:
            self._window.setFrame_display_(frame, False)
            self._view.setFrame_(NSMakeRect(0, 0, bounds.width, bounds.height))

    # Updates the overlay's drawing state and triggers a redraw.
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

    # Returns the overlay window's id, or None if not created.
    def window_id(self) -> int | None:
        return self._window.windowNumber() if self._window is not None else None

    # Hides the overlay window.
    def hide(self) -> None:
        if self._window is not None:
            self._window.orderOut_(None)

    # Closes the overlay window and releases it.
    def close(self) -> None:
        if self._window is not None:
            self._window.close()
            self._window = None
            self._view = None
