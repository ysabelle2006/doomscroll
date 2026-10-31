"""macOS on-screen text scanning: mss for screen capture (already a project
dependency -- it handles Retina/scaled-display capture correctly on its
own) feeding into the Vision framework (VNRecognizeTextRequest, via
pyobjc-framework-Vision) for OCR.

Not exercised on the Linux dev machine -- pyobjc/Vision
aren't installable there at all, so unlike windowing/scroll's macOS
backends (which at least import-guard cleanly), this one has had zero
real execution before reaching macOS. Treat the exact Vision API surface
(request class names, the recognition-level constant, performRequests_error_
signature) as unverified until it's actually run there; _vision_bbox_to_screen_px
below is the one piece of real logic and is covered by ordinary unit tests
since it's plain arithmetic.

Vision returns each recognized text line's bounding box as FRACTIONAL
coordinates (0..1) relative to the input image, bottom-left origin (y
increases upward). Converting that fraction back to absolute coordinates
using the WINDOW's own bounds (not the captured image's raw pixel size) is
deliberately scale-invariant -- a Retina 2x capture and a 1x capture of the
same window produce identical ContentBlock output, with no special-casing
needed here for display scaling.
"""
from __future__ import annotations

from doomscroll.content.base import ContentBlock, ContentScanner
from doomscroll.windowing.base import WindowBounds

try:
    import mss
    import mss.tools
    from Foundation import NSData
    from Vision import (
        VNImageRequestHandler,
        VNRecognizeTextRequest,
        VNRequestTextRecognitionLevelAccurate,
    )

    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False


def _vision_bbox_to_screen_px(
    bbox: tuple[float, float, float, float], bounds: WindowBounds
) -> tuple[int, int, int, int]:
    """bbox: (x, y, width, height), all fractions in [0, 1], in Vision's
    bottom-left-origin convention (y increases upward from the bottom of
    the scanned image). Returns (x, y, width, height) in absolute
    screen-pixel coordinates, top-left origin -- the convention
    WindowBounds/gaze points use everywhere else in this project.

    Deliberately uses only `bounds`' own width/height, never the captured
    image's actual pixel dimensions -- see module docstring."""
    vx, vy, vw, vh = bbox
    px_x = bounds.x + vx * bounds.width
    px_w = vw * bounds.width
    px_h = vh * bounds.height
    # Flip vertical origin: Vision's y=0 is the BOTTOM of the image;
    # bounds.y is the TOP of the window.
    px_y = bounds.y + (1.0 - vy - vh) * bounds.height
    # Small empirical correction (-0.5px), tuned by hand against a real
    # screen -- not derived from anything in the flip math above, which is
    # otherwise exact. Re-tune if it ever looks off on a different display.
    px_y -= 0.5
    return int(round(px_x)), int(round(px_y)), int(round(px_w)), int(round(px_h))


class MacOSVisionContentScanner(ContentScanner):
    def __init__(self):
        if not _VISION_AVAILABLE:
            raise RuntimeError(
                "pyobjc-framework-Vision (and mss) are required on macOS for "
                "content-aware scrolling -- pyobjc-framework-Vision is a base "
                "dependency there, so `uv sync` / `pip install -e .` should "
                "already have installed it; try re-running that."
            )
        self._sct = mss.mss()

    def scan(self, bounds: WindowBounds) -> list[ContentBlock]:
        if bounds.width <= 0 or bounds.height <= 0:
            return []

        monitor = {"left": bounds.x, "top": bounds.y, "width": bounds.width, "height": bounds.height}
        shot = self._sct.grab(monitor)

        # Hand Vision compressed image bytes (PNG) rather than building a
        # CGImage from the raw BGRA buffer by hand -- simpler and avoids
        # getting the colorspace/bytes-per-row bookkeeping wrong for a
        # one-shot scan that only runs a couple times a second.
        #
        # mss.tools.to_png() expects RGB (3 bytes/pixel) data -- shot.rgb,
        # not shot.bgra (4 bytes/pixel, raw capture order). Every example
        # in mss's own docs passes .rgb; passing .bgra here produced a
        # structurally corrupted PNG (wrong bytes-per-pixel entirely, not
        # just swapped channels) that Vision could decode without raising
        # but found zero legible text in -- exactly the "0 blocks, no
        # error" symptom this shipped with. Confirmed via a real Mac run:
        # an independent diagnostic capture of the same region (bypassing
        # to_png) looked like a completely normal screenshot, proving the
        # capture/permissions were fine and narrowing the bug to this line.
        png_bytes = mss.tools.to_png(shot.rgb, shot.size)
        ns_data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))

        handler = VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
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
