# macOS implementation of ContentScanner using Quartz screen capture and Vision OCR.
from __future__ import annotations

from doomscroll.content.base import ContentBlock, ContentScanner
from doomscroll.windowing.base import WindowBounds

try:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        CGWindowListCreateImage,
        CGWindowListCreateImageFromArray,
        kCGNullWindowID,
        kCGWindowImageDefault,
        kCGWindowListOptionOnScreenOnly,
    )
    from Vision import (
        VNImageRequestHandler,
        VNRecognizeTextRequest,
        VNRequestTextRecognitionLevelAccurate,
    )

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False


# Converts a Vision normalized bounding box to screen pixel coordinates.
def _vision_bbox_to_screen_px(
    bbox: tuple[float, float, float, float], bounds: WindowBounds
) -> tuple[int, int, int, int]:
    vx, vy, vw, vh = bbox
    px_x = bounds.x + vx * bounds.width
    px_w = vw * bounds.width
    px_h = vh * bounds.height
    px_y = bounds.y + (1.0 - vy - vh) * bounds.height
    px_y -= 0.5
    return int(round(px_x)), int(round(px_y)), int(round(px_w)), int(round(px_h))


# Scans the screen for text blocks using Vision OCR on a Quartz screenshot.
class MacOSVisionContentScanner(ContentScanner):
    # Raises if the pyobjc Vision/Quartz frameworks are not installed.
    def __init__(self):
        if not _VISION_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Vision and pyobjc-framework-Quartz are required on "
                "macOS for content-aware scrolling -- both are base dependencies "
                "there, so `uv sync` / `pip install -e .` should already have "
                "installed them; try re-running that."
            )

    # Captures the window region (excluding given windows) and OCRs it into text blocks.
    def scan(self, bounds: WindowBounds, exclude_window_ids: frozenset[int] = frozenset()) -> list[ContentBlock]:
        if bounds.width <= 0 or bounds.height <= 0:
            return []

        screen_rect = ((bounds.x, bounds.y), (bounds.width, bounds.height))

        if exclude_window_ids:
            window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            window_ids = [
                w["kCGWindowNumber"] for w in window_list if w.get("kCGWindowNumber") not in exclude_window_ids
            ]
            cg_image = CGWindowListCreateImageFromArray(screen_rect, window_ids, kCGWindowImageDefault)
        else:
            cg_image = CGWindowListCreateImage(
                screen_rect, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, kCGWindowImageDefault
            )
        if cg_image is None:
            return []

        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)

        success, _error = handler.performRequests_error_([request], None)
        if not success:
            return []

        blocks: list[ContentBlock] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            text = str(candidates[0].string())
            if not text.strip():
                continue
            box = observation.boundingBox()
            x, y, w, h = _vision_bbox_to_screen_px(
                (box.origin.x, box.origin.y, box.size.width, box.size.height), bounds
            )
            blocks.append(ContentBlock(text=text, x=x, y=y, width=w, height=h))
        return blocks
