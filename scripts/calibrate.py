"""Phase 2: gaze calibration, plus a head-pose variation pass.

Two calibration PROTOCOLS now coexist, selected by --features (see
_run_grid_protocol()/_run_trace_protocol() and main()):

  - --features handcrafted (default) uses the original point-by-point
    dot-grid protocol (run_sequence()/collect_samples_for_point()): you
    look at each dot in turn, head still and forward, for a fixed capture
    window per point. This is what calibration used before the continuous
    cursor-trace protocol below existed, and it's back as the default
    because the trace protocol was never validated against it head-to-head
    for the handcrafted backend -- see the revert that reintroduced this
    docstring's grid section for the full reasoning. It needs nothing
    beyond mouse clicks to start each pass; no mouse TRACKING during
    capture.
  - --features l2cs uses the continuous cursor-tracing protocol
    (run_continuous_trace()) described below -- kept for l2cs specifically
    because that's what it was built and tested against.

Both protocols share the same wait_for_front_ready()/wait_for_pose_ready()
readiness screens, the same six-variant head-pose pass, and the same
fit/validate/save tail end in main() -- only the actual capture mechanic
differs.

Before anything starts, a live readiness screen (wait_for_front_ready())
shows the camera feed -- with a literal box drawn around your detected
face so you can SEE you're in frame, not just read a number -- plus your
current yaw/pitch/distance, so you have a concrete reference point for
"this is normally how far/centered I sit" before committing to it for the
whole session.

--- Dot-grid protocol (--features handcrafted) ---

Main pass: a 4x4 grid (16 points, see TRAIN_GRID_FRACTIONS). Keep your head
facing forward, roughly toward the center of the screen, and use ONLY your
eyes to look at each dot as it appears -- don't turn your head toward it.
Each point gets a brief settle window then a fixed capture window (SETTLE_S
/ CAPTURE_S). A held-out 5-point set (VALIDATION_FRACTIONS) is used for the
validation pass afterward.

--- Continuous cursor-trace protocol (--features l2cs) ---

Main pass: a single-stroke zigzag line, drawn with directional arrows,
sweeps edge-to-edge across the whole screen (see _zigzag_path_fractions())
-- covering far more of it, and far more continuously, than a handful of
discrete dots ever could. A green dot marks the one unambiguous START and
a white ring marks the one unambiguous END. Once you click near the start
(wait_for_click()), you trace the line with your own mouse cursor --
looking at your cursor the whole time -- and gaze samples are collected
CONTINUOUSLY the entire time you're tracing. There is no time limit: the
pass only ends once you've actually reached the end of the line (see
run_continuous_trace()'s progress tracking), not after some arbitrary
duration. Straying more than PATH_MARGIN_PX from the line shows a
"you've strayed" message (still never a hard block -- it's feedback, not
an error), and moving faster than MAX_TRACE_SPEED_PX_S shows a "slow down"
message and skips capturing until you ease off.

Why this isn't naive cursor-pursuit calibration, and what's done about it
-- logging "wherever the cursor currently is, right now" as the training
label for every captured frame would be naive: a synthetic comparison
(backed by tests/synthetic_calibration.py's modeled pursuit lag/gain)
found that doing it that way is dramatically WORSE than the old still-point
protocol -- 5-10x worse under realistic assumptions -- because smooth
pursuit eye movement lags a moving target it's tracking by a reliable,
well-documented amount (~100-150ms), and that lag becomes label noise no
amount of model fitting can remove. So every captured sample here is
labeled with the cursor's position from LAG_COMPENSATION_S *before* that
frame (see _lagged_cursor_position()), not its position at that exact
instant -- shifting the label back by the same amount the eye is expected
to be lagging by removes most of that bias. A cursor moving faster than
MAX_TRACE_SPEED_PX_S is skipped outright rather than trusted: faster motion
means proportionally more undershoot (gaze trailing the target in
*position*, not just time, the "pursuit gain < 1" part of the same model)
that a fixed time-shift alone doesn't fully correct for, plus more motion
blur -- and this is also the live "you're going too fast" feedback shown
on screen.

Where the live cursor position comes from, and why -- a reported bug had
the drawn target and the real mouse cursor visibly not lining up on
screen. The fix: this file no longer asks the OS for the cursor position
at all (the earlier approach, via capture/cursor.py's Quartz/win32 call,
turned out not to reliably agree with OpenCV's own rendering coordinate
space). Instead, _MouseTracker reads the cursor's position straight from
OpenCV's own mouse-move callback (cv2.setMouseCallback) on this exact
window -- which is, by construction, in the exact same pixel space as
whatever was last drawn into that window, so there's no second coordinate
system to reconcile. The one remaining conversion, _map_real_to_logical(),
handles only the display_size-vs-logical-point scale (see
_detect_display_size()), which is a much narrower, already-validated gap.

Head-pose variation pass: immediately after, you'll be asked to hold your
head in a few different natural positions -- lean LEFT, lean RIGHT, lean
CLOSER (forward), sit BACK (backward), tilt UP, tilt DOWN -- while tracing
a shorter zigzag pattern at each. The main pass alone gives the model
almost no variation in head rotation or position to learn from (you're
asked to stay forward throughout), so at runtime, real head movement lands
outside anything the model ever saw and the mapping degrades badly. This
pass gives the same kind of continuously-collected data under genuinely
different head positions, which is what lets the model learn to correct
for head motion instead of just extrapolating blindly.

Each pose-variant transition is self-paced: it shows the live camera feed
plus a readout (yaw/pitch/camera-distance, whichever that variant actually
depends on) and waits for you to press SPACE once you're actually holding
the position and still visibly in frame -- see wait_for_pose_ready().
People take a different amount of time to find and hold an off-center pose;
a fixed timer either wastes time once you're already there, or starts
capturing before someone who needed longer has actually arrived.

A continuous validation pass follows (same tracing mechanic, swept by
column instead of by row so it's visually and spatially distinct from the
training pattern), reporting mean pixel error.

Raw per-session samples are always saved afterward (see session_store.py),
so later runs can optionally pool recent sessions together with --pool.
--list-history / --clear-history manage that stored history.

Press ESC at any time to abort.

Run: .venv\\Scripts\\python.exe scripts\\calibrate.py
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.gaze.calibration import FEATURE_DIMS, GazeCalibrator, build_feature_vector, profile_path_for_backend
from doomscroll.gaze.feature_backend import FEATURE_BACKEND_NAMES, GazeVectorExtractor, make_gaze_extractor
from doomscroll.gaze.gaze_vector import is_gaze_usable
from doomscroll.gaze.head_pose import HeadPoseEstimator, HeadPose
from doomscroll.gaze.session_store import clear_sessions, list_sessions, pool_sessions, save_session
from doomscroll.landmarks.mediapipe_backend import MediaPipeLandmarkBackend

WINDOW_NAME = "doomscroll calibration"
POINT_RADIUS = 18
CURSOR_MARKER_RADIUS = 10
ARROW_TIP_PX = 16.0

# Deliberately stricter than gaze_vector.DEFAULT_MIN_EYE_ASPECT_RATIO (which
# live tracking uses). Live tracking needs recall -- rejecting a genuine
# downward glance as a "blink" freezes the cursor, so it's tuned loose. But
# calibration has the opposite priority: precision. There's no cost to
# being picky here (continuous tracing gives plenty of frames to choose
# from), and every cost to not being: a frame where the eye is
# narrowed-but-not-closed still measures a compressed iris offset (the
# eyelid is cropping how far the iris visibly travels), which teaches the
# model "this small offset means a target way down the screen" -- then that
# same small-offset reading recurs for genuinely centered gaze live, and the
# model (especially the local-correction stage, which is sensitive to a
# handful of mislabeled neighbors) pulls it toward "down" too.
CALIBRATION_MIN_EAR = 0.12

# --- Dot-grid protocol mechanics (--features handcrafted) -----------------

STILL_INSTRUCTION = "keep your head still & facing forward -- move only your eyes to the dot"

SETTLE_S = 0.6
CAPTURE_S = 1.4

# 4x4 grid (16 points) for denser spatial coverage -- more independent
# calibration events for RidgeCV's per-region regularization search to work
# with, and better coverage near the edges where the dwell-based scroll
# trigger actually needs to fire.
_GRID_STEPS = [0.08, 0.36, 0.64, 0.92]
TRAIN_GRID_FRACTIONS = [(fx, fy) for fy in _GRID_STEPS for fx in _GRID_STEPS]

# Held out, deliberately off the training grid.
VALIDATION_FRACTIONS = [(0.22, 0.22), (0.78, 0.22), (0.5, 0.5), (0.22, 0.78), (0.78, 0.78)]

# Reduced 5-point set (edge midpoints + center) repeated once per head-pose
# variant below. Deliberately distinct from both TRAIN_GRID_FRACTIONS and
# VALIDATION_FRACTIONS so this pass doesn't overlap either -- its purpose is
# pose variation, not additional spatial coverage.
POSE_VARIATION_FRACTIONS = [(0.08, 0.5), (0.92, 0.5), (0.5, 0.08), (0.5, 0.92), (0.5, 0.5)]

# --- Continuous cursor-tracing mechanics (--features l2cs) ----------------
#
# See the module docstring for the pursuit-lag problem this is designed
# around (naive cursor-pursuit labeling is measurably worse -- see
# tests/synthetic_calibration.py's modeled pursuit lag/gain).

# Shift each captured sample's label back in time by this much, to
# approximate "where the cursor was when the eye -- which is still
# catching up -- was actually looking", rather than "where the cursor is
# right now". Matches tests/synthetic_calibration.py's PURSUIT_LATENCY_S
# default, so this isn't an independently-guessed number -- it's the same
# lag assumption already validated against that synthetic model.
LAG_COMPENSATION_S = 0.12

# How much recent cursor-position history to keep around, so a lookup for
# "the cursor's position LAG_COMPENSATION_S ago" always has something to
# interpolate from. Comfortably more than LAG_COMPENSATION_S itself.
CURSOR_HISTORY_MAX_S = 0.6

# Above this cursor speed (px/s), skip capturing a sample entirely rather
# than trust it -- this is the real "you're going too fast" threshold
# behind the on-screen warning. Faster motion means more pursuit
# undershoot (not just lag) than the fixed LAG_COMPENSATION_S shift
# corrects for, and more motion blur degrading the landmark read.
MAX_TRACE_SPEED_PX_S = 2600.0

# Contiguous time window treated as one group for GazeCalibrator's
# group-aware cross-validation (see calibration.py's docstring on why that
# matters) -- matches tests/synthetic_calibration.py's
# PURSUIT_GROUP_WINDOW_S for the same reason LAG_COMPENSATION_S does.
GROUP_WINDOW_S = 1.0

# How far (logical screen px) the cursor may stray from the guide line
# before a "you've strayed" message shows. Informational only -- straying
# doesn't stop samples being captured (your actual cursor position is
# still a valid label regardless of how close it is to the intended line;
# see run_continuous_trace()), it just means you're not covering the
# intended line as well as you could be.
PATH_MARGIN_PX = 90.0

# How close (logical screen px) the cursor must get to the path's FINAL
# point to count as "reached the end".
END_RADIUS_PX = 50.0

# The cursor's cumulative progress along the path (by arc length, via
# _path_progress()) must have reached at least this fraction at some point
# before "reached the end" is honored -- stops someone from finishing by
# jumping straight to the end point without having actually traced the
# line first.
MIN_PROGRESS_TO_FINISH = 0.9

# Safety net only, not a normal "time limit": if a pass somehow never
# reaches its end (e.g. mouse tracking stops updating), stop anyway after
# this long rather than hang indefinitely.
MAX_PASS_FAILSAFE_S = 300.0

# How close (real, on-screen px -- see _MouseTracker) a click needs to
# land to the pattern's start point to count as "starting the trace"
# there, in wait_for_click(). A soft requirement, not a hard validation
# error -- click too far away and the screen just keeps waiting.
CLICK_START_RADIUS_PX = 70.0

# How far yaw/pitch may sit from dead-on-0 and still count as "straight" on
# the pre-main-pass readiness screen (wait_for_front_ready()) -- purely
# informational (like the pose-variant ranges, it's never enforced; SPACE
# always proceeds), just loose enough that normal, comfortable sitting
# posture reads as "straight" rather than nagging over a degree or two.
FRONT_YAW_PITCH_TOLERANCE_DEG = 8.0
FRONT_READY_INSTRUCTION = (
    "Sit facing forward, centered in frame, at whatever distance feels natural "
    "-- this is where the main pass will track your eyes from."
)

# "see that their head is in frame before starting": padding mirrors
# l2cs_backend.py's own face_crop_from_points() convention, so the box
# drawn here is representative of what the l2cs appearance-model crop
# actually uses, not just a cosmetic guess.
FACE_CROP_PAD_FRAC = 0.25
FACE_FRAME_EDGE_MARGIN_FRAC = 0.04

_display_size: tuple[int, int] | None = None  # set once in main(), see _detect_display_size()


@dataclass
class PoseVariant:
    label: str
    instruction: str
    yaw_range: tuple[float, float] | None = None
    pitch_range: tuple[float, float] | None = None
    z_delta_range: tuple[float, float] | None = None


# Natural head-position variants: rotation (lean left/right turns the head
# slightly) and translation (closer/back changes distance and, on most desk
# setups, also height/angle a little) -- both of which the model previously
# never saw any variation in at all. Kept to "slight"/"natural" amounts
# deliberately: the point is to teach the model to tolerate incidental
# real-world head motion, not to calibrate for extreme poses it will rarely
# see in normal use. Together these six are what the project plan refers to
# as left/right/up/down/forward/backward: lean LEFT/RIGHT = left/right,
# tilt UP/DOWN = up/down, lean CLOSER/sit BACK = forward/backward.
#
# yaw_range/pitch_range/z_delta_range are shown live by wait_for_pose_ready()
# as a rough visual guide only (green = "roughly there") -- not enforced;
# SPACE always proceeds regardless of color, since a person's comfortable
# "slight" lean varies and these are guides, not a pose to force exactly.
# z_delta_range is relative to whatever camera distance you're at the
# moment that variant's wait screen appears (there's no single fixed
# "target distance" across different desks/chairs), negative = closer.
POSE_VARIANTS = [
    PoseVariant(
        "lean LEFT",
        "Lean your head slightly to the LEFT and hold that position -- keep tracking with your eyes only.",
        yaw_range=(-30.0, -10.0),
    ),
    PoseVariant(
        "lean RIGHT",
        "Lean your head slightly to the RIGHT and hold that position -- keep tracking with your eyes only.",
        yaw_range=(10.0, 30.0),
    ),
    PoseVariant(
        "lean CLOSER",
        "Lean slightly CLOSER to the screen and hold that position -- keep tracking with your eyes only.",
        z_delta_range=(-200.0, -80.0),
    ),
    PoseVariant(
        "sit BACK",
        "Sit slightly further BACK from the screen and hold that position -- keep tracking with your eyes only.",
        z_delta_range=(50.0, 130.0),
    ),
    PoseVariant(
        "tilt UP",
        "Tilt your head slightly UP and hold that position -- keep tracking with your eyes only, don't drop your eyes to compensate.",
        pitch_range=(6.0, 20.0),
    ),
    PoseVariant(
        "tilt DOWN",
        "Tilt your head slightly DOWN (not too far) and hold that position -- keep tracking with your eyes only, don't raise your eyes to compensate.",
        pitch_range=(-20.0, -6.0),
    ),
]


# --- Guide-pattern shapes ---------------------------------------------------
#
# Each returns an OPEN, single-stroke polyline of (fx, fy) screen fractions
# -- one unambiguous start, one unambiguous end, drawn with arrows showing
# the direction of travel (see _draw_path_with_arrows()). Every point
# *along* each segment gets traced continuously as you drag your mouse
# across it, not just the listed vertices -- so a handful of vertices
# still produces dense, full-coverage sampling.


def _zigzag_path_fractions(
    x_range: tuple[float, float] = (0.03, 0.97),
    y_range: tuple[float, float] = (0.03, 0.97),
    n_rows: int = 7,
) -> list[tuple[float, float]]:
    """A single continuous left-right-left-right sweep (boustrophedon)
    covering the given fractional region edge-to-edge -- an open path
    (not a closed loop) with one start (top-left) and one end (depends on
    whether n_rows is odd/even), so run_continuous_trace() has a single,
    well-defined finish line."""
    rows_y = np.linspace(y_range[0], y_range[1], n_rows)
    path: list[tuple[float, float]] = []
    for i, fy in enumerate(rows_y):
        xs = (x_range[0], x_range[1]) if i % 2 == 0 else (x_range[1], x_range[0])
        for fx in xs:
            path.append((float(fx), float(fy)))
    return path


def _vertical_zigzag_path_fractions(
    x_range: tuple[float, float] = (0.06, 0.94),
    y_range: tuple[float, float] = (0.06, 0.94),
    n_cols: int = 6,
) -> list[tuple[float, float]]:
    """Same idea as _zigzag_path_fractions() but swept top-to-bottom by
    column instead of left-to-right by row -- used for validation so its
    shape and sweep direction are visually and spatially distinct from the
    main training pattern, not a relabeled copy of it."""
    cols_x = np.linspace(x_range[0], x_range[1], n_cols)
    path: list[tuple[float, float]] = []
    for i, fx in enumerate(cols_x):
        ys = (y_range[0], y_range[1]) if i % 2 == 0 else (y_range[1], y_range[0])
        for fy in ys:
            path.append((float(fx), float(fy)))
    return path


def _detect_display_size(screen_w: int, screen_h: int) -> tuple[int, int]:
    """Ask OpenCV what pixel size it actually gave the fullscreen window,
    instead of trusting our own screen-size query (get_screen_size()) to
    exactly match it. Switching that query to Quartz (see dpi.py) was meant
    to fix the grey strip left uncovered along one edge of the window, but a
    gap was still observed afterwards -- so rather than keep guessing at the
    exact cause (Retina backing-store scaling, multi-monitor placement, or
    something else in how the Cocoa highgui backend sizes its content view),
    ask OpenCV directly and scale every rendered frame to fill *that* size.
    Falls back to (screen_w, screen_h) -- today's status quo -- if the query
    isn't supported or the window hasn't settled yet."""
    blank = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    # present twice: some backends only finish resizing the content view to
    # the fullscreen frame after the window has already been shown once
    for _ in range(2):
        cv2.imshow(WINDOW_NAME, blank)
        cv2.waitKey(150)
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW_NAME)
    except Exception:
        w, h = 0, 0
    if w > 0 and h > 0:
        return w, h
    return screen_w, screen_h


def _map_real_to_logical(
    point: tuple[float, float], screen_w: int, screen_h: int, display_size: tuple[int, int] | None
) -> tuple[float, float]:
    """Converts a point from the REAL on-screen pixel space the rendered
    window actually occupies (display_size, as measured by
    _detect_display_size() -- also the space _MouseTracker's coordinates
    are already in, since that's the array shape handed to cv2.imshow)
    into the LOGICAL screen_w x screen_h space everything else in this
    file (patterns, drawn targets) is drawn and reasoned about in. A no-op
    when the two spaces already match (display_size is None, or equal to
    (screen_w, screen_h) -- the common case on a non-scaled display)."""
    if display_size is None or display_size == (screen_w, screen_h):
        return point
    scale_x = screen_w / display_size[0]
    scale_y = screen_h / display_size[1]
    return (point[0] * scale_x, point[1] * scale_y)


def _present(frame: np.ndarray) -> None:
    """Shows `frame` (drawn in the logical, DPI-aware screen_w x screen_h
    coordinate space that calibration math and the runtime gaze->scroll
    mapping both rely on) scaled to fill the real detected window size, so
    the two can differ without leaving any of the window uncovered."""
    if _display_size is not None and _display_size != (frame.shape[1], frame.shape[0]):
        frame = cv2.resize(frame, _display_size, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(WINDOW_NAME, frame)


def _camera_background(frame: np.ndarray | None, screen_w: int, screen_h: int) -> np.ndarray:
    """Builds the live-preview background for a wait screen: the current
    webcam frame stretched to fill the logical screen_w x screen_h canvas,
    so a person getting into position can see whether their head is still
    actually in frame -- rather than a plain black screen that gives no
    feedback on framing at all. Falls back to black if no frame is
    available yet (camera still warming up / a dropped read)."""
    if frame is None:
        return np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    return cv2.resize(frame, (screen_w, screen_h))


def _darken_band(canvas: np.ndarray, y0: int, y1: int, alpha: float = 0.55) -> None:
    """Dims a horizontal band of `canvas` in place, behind where overlay
    text is about to be drawn. Needed once the background became a live
    camera feed instead of solid black -- a bright/varied face/background
    can otherwise wash out the instruction and readout text."""
    y0, y1 = max(0, y0), min(canvas.shape[0], y1)
    if y1 <= y0:
        return
    canvas[y0:y1] = (canvas[y0:y1].astype(np.float32) * (1 - alpha)).astype(np.uint8)


def _wrap_line(text: str, font: int, font_scale: float, thickness: int, max_width: int) -> list[str]:
    """Greedily wraps `text` on spaces so every rendered line measures at
    most max_width pixels (via cv2.getTextSize). Without this, a long
    instruction string is drawn as a single line and silently runs off the
    right edge of the screen instead of wrapping -- which is exactly what
    happened with the longer pose-variant instructions (e.g. "tilt UP"/
    "tilt DOWN")."""
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _face_frame_status(
    points_px: np.ndarray | None, image_width: int, image_height: int
) -> tuple[str, bool, tuple[float, float, float, float] | None]:
    """Pure logic behind the "is your head in frame" check shown on both
    wait screens -- split out so it's unit-testable without a camera/GUI,
    same as _pose_variant_readout()/_front_readout(). Returns (status text,
    whether the (padded) face box sits comfortably inside the raw camera
    frame, that box in raw-frame pixel coords or None)."""
    if points_px is None:
        return "no face detected -- move into the camera's view", False, None

    x_min, y_min = points_px.min(axis=0)
    x_max, y_max = points_px.max(axis=0)
    pad_x = (x_max - x_min) * FACE_CROP_PAD_FRAC
    pad_y = (y_max - y_min) * FACE_CROP_PAD_FRAC
    box = (x_min - pad_x, y_min - pad_y, x_max + pad_x, y_max + pad_y)

    margin_x = image_width * FACE_FRAME_EDGE_MARGIN_FRAC
    margin_y = image_height * FACE_FRAME_EDGE_MARGIN_FRAC
    touching_edge = (
        box[0] < margin_x or box[1] < margin_y or box[2] > image_width - margin_x or box[3] > image_height - margin_y
    )
    if touching_edge:
        return "face is close to the edge of frame -- center yourself", False, box
    return "in frame", True, box


def _draw_face_box(
    canvas: np.ndarray,
    box: tuple[float, float, float, float] | None,
    frame_w: int,
    frame_h: int,
    screen_w: int,
    screen_h: int,
    in_frame: bool,
) -> None:
    """Draws `box` (raw camera-frame pixel coords, from _face_frame_status())
    scaled onto `canvas`, which is the same frame resized to screen_w x
    screen_h by _camera_background() -- so the rectangle lines up with the
    face actually visible in the live preview."""
    if box is None or frame_w <= 0 or frame_h <= 0:
        return
    scale_x, scale_y = screen_w / frame_w, screen_h / frame_h
    x_min, y_min, x_max, y_max = box
    pt1 = (int(x_min * scale_x), int(y_min * scale_y))
    pt2 = (int(x_max * scale_x), int(y_max * scale_y))
    color = (0, 220, 0) if in_frame else (0, 0, 255)
    cv2.rectangle(canvas, pt1, pt2, color, 2)


def _pose_variant_readout(head_pose: HeadPose | None, variant: PoseVariant, baseline_z: float | None) -> tuple[str, bool, float | None]:
    """Returns (readout text, whether it looks roughly in range, updated
    baseline_z). baseline_z is only set/used for z_delta_range variants --
    it's captured from the first successful reading each time
    wait_for_pose_ready() is called, since there's no fixed target distance
    independent of the user's own desk setup."""
    if head_pose is None:
        return "face not detected", False, baseline_z
    if variant.yaw_range is not None:
        lo, hi = variant.yaw_range
        in_range = lo <= head_pose.yaw_deg <= hi
        return f"yaw={head_pose.yaw_deg:+.1f} deg  (roughly {lo:+.0f} to {hi:+.0f})", in_range, baseline_z
    if variant.pitch_range is not None:
        lo, hi = variant.pitch_range
        in_range = lo <= head_pose.pitch_deg <= hi
        return f"pitch={head_pose.pitch_deg:+.1f} deg  (roughly {lo:+.0f} to {hi:+.0f})", in_range, baseline_z
    if variant.z_delta_range is not None:
        current_z = float(head_pose.translation_vector[2, 0])
        if baseline_z is None:
            baseline_z = current_z
        delta_z = current_z - baseline_z
        lo, hi = variant.z_delta_range
        in_range = lo <= delta_z <= hi
        return f"distance change={delta_z:+.0f}mm  (roughly {lo:+.0f} to {hi:+.0f}; negative=closer)", in_range, baseline_z
    return "", True, baseline_z


def wait_for_pose_ready(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    variant: PoseVariant,
) -> None:
    """Self-paced replacement for a fixed-duration show_transition_message():
    shows the instruction plus a live readout of whatever this variant's
    pose actually depends on (yaw, pitch, or camera distance), plus a box
    around your face so you can see you're still in frame while holding an
    off-center pose, and waits for SPACE before capture starts. The
    readout's color is feedback only, not a gate: SPACE always proceeds."""
    font, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 2, 60
    instruction_lines = _wrap_line(variant.instruction, font, 0.9, thickness, screen_w - 2 * margin)
    baseline_z: float | None = None

    while True:
        ok, frame = cam.read()
        head_pose = None
        face_status, in_frame, face_box = "no face detected -- move into the camera's view", False, None
        if ok and frame is not None:
            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)
            if result is not None:
                points_px = result.points_px()
                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                face_status, in_frame, face_box = _face_frame_status(points_px, result.image_width, result.image_height)

        readout, in_range, baseline_z = _pose_variant_readout(head_pose, variant, baseline_z)

        canvas = _camera_background(frame if ok else None, screen_w, screen_h)
        if ok and frame is not None:
            _draw_face_box(canvas, face_box, frame.shape[1], frame.shape[0], screen_w, screen_h, in_frame)

        line_height = 40
        start_y = screen_h // 2 - 100 - (line_height * (len(instruction_lines) - 1)) // 2
        _darken_band(canvas, start_y - 40, screen_h // 2 + 150)
        for i, line in enumerate(instruction_lines):
            (text_w, _), _ = cv2.getTextSize(line, font, 0.9, thickness)
            x = screen_w // 2 - text_w // 2
            cv2.putText(canvas, line, (x, start_y + i * line_height), font, 0.9, (0, 220, 255), thickness)

        face_color = (0, 220, 0) if in_frame else (0, 0, 255)
        (face_w, _), _ = cv2.getTextSize(face_status, font, 0.7, 2)
        cv2.putText(canvas, face_status, (screen_w // 2 - face_w // 2, screen_h // 2 + 30), font, 0.7, face_color, 2)

        if readout:
            color = (0, 0, 255) if readout == "face not detected" else ((0, 220, 0) if in_range else (0, 165, 255))
            (text_w, _), _ = cv2.getTextSize(readout, font, 0.7, 2)
            cv2.putText(canvas, readout, (screen_w // 2 - text_w // 2, screen_h // 2 + 65), font, 0.7, color, 2)

        hint = "Press SPACE when ready (ESC to abort)"
        (hint_w, _), _ = cv2.getTextSize(hint, font, 0.7, 2)
        cv2.putText(canvas, hint, (screen_w // 2 - hint_w // 2, screen_h // 2 + 110), font, 0.7, (150, 150, 150), 2)

        _present(canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:  # SPACE
            return


def _front_readout(head_pose: HeadPose | None) -> tuple[str, bool]:
    """Pure logic behind wait_for_front_ready()'s live readout -- split out
    so it's unit-testable without a camera/GUI, same as
    _pose_variant_readout(). Returns (readout text, whether yaw AND pitch
    both look roughly straight)."""
    if head_pose is None:
        return "face not detected", False
    yaw_deg, pitch_deg = head_pose.yaw_deg, head_pose.pitch_deg
    distance_mm = float(head_pose.translation_vector[2, 0])
    straight = abs(yaw_deg) <= FRONT_YAW_PITCH_TOLERANCE_DEG and abs(pitch_deg) <= FRONT_YAW_PITCH_TOLERANCE_DEG
    status = "straight" if straight else "not quite straight"
    return f"yaw={yaw_deg:+.1f} deg  pitch={pitch_deg:+.1f} deg  distance={distance_mm:.0f}mm  ({status})", straight


def wait_for_front_ready(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    instruction: str = FRONT_READY_INSTRUCTION,
) -> None:
    """Live-camera readiness screen -- the same self-paced pattern as
    wait_for_pose_ready(), but reporting "are you in frame at all" and
    yaw/pitch/distance together."""
    font, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 2, 60
    instruction_lines = _wrap_line(instruction, font, 0.9, thickness, screen_w - 2 * margin)

    while True:
        ok, frame = cam.read()
        head_pose = None
        face_status, in_frame, face_box = "no face detected -- move into the camera's view", False, None
        if ok and frame is not None:
            timestamp_ms = cam.frame_timestamp_ms()
            result = backend.detect(frame, timestamp_ms)
            if result is not None:
                points_px = result.points_px()
                head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                face_status, in_frame, face_box = _face_frame_status(points_px, result.image_width, result.image_height)

        canvas = _camera_background(frame if ok else None, screen_w, screen_h)
        if ok and frame is not None:
            _draw_face_box(canvas, face_box, frame.shape[1], frame.shape[0], screen_w, screen_h, in_frame)

        line_height = 40
        start_y = screen_h // 2 - 100 - (line_height * (len(instruction_lines) - 1)) // 2
        _darken_band(canvas, start_y - 40, screen_h // 2 + 150)
        for i, line in enumerate(instruction_lines):
            (text_w, _), _ = cv2.getTextSize(line, font, 0.9, thickness)
            x = screen_w // 2 - text_w // 2
            cv2.putText(canvas, line, (x, start_y + i * line_height), font, 0.9, (0, 220, 255), thickness)

        face_color = (0, 220, 0) if in_frame else (0, 0, 255)
        (face_w, _), _ = cv2.getTextSize(face_status, font, 0.7, 2)
        cv2.putText(canvas, face_status, (screen_w // 2 - face_w // 2, screen_h // 2 + 30), font, 0.7, face_color, 2)

        readout, straight = _front_readout(head_pose)
        color = (0, 0, 255) if readout == "face not detected" else ((0, 220, 0) if straight else (0, 165, 255))
        (text_w, _), _ = cv2.getTextSize(readout, font, 0.7, 2)
        cv2.putText(canvas, readout, (screen_w // 2 - text_w // 2, screen_h // 2 + 65), font, 0.7, color, 2)

        hint = "Press SPACE when ready (ESC to abort)"
        (hint_w, _), _ = cv2.getTextSize(hint, font, 0.7, 2)
        cv2.putText(canvas, hint, (screen_w // 2 - hint_w // 2, screen_h // 2 + 110), font, 0.7, (150, 150, 150), 2)

        _present(canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:  # SPACE
            return


# --- Dot-grid pass (--features handcrafted) --------------------------------


def draw_target(screen_w: int, screen_h: int, point_px: tuple[int, int], instruction: str = STILL_INSTRUCTION) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.circle(frame, point_px, POINT_RADIUS, (0, 220, 255), -1)
    cv2.circle(frame, point_px, POINT_RADIUS, (255, 255, 255), 2)

    # Wrap the bottom status text so long instructions (the pose-variant
    # strings, in particular) stay on screen instead of running off the
    # right edge -- see _wrap_line().
    font, font_scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 20
    lines = _wrap_line(instruction, font, font_scale, thickness, screen_w - 2 * margin)
    line_height = 26
    base_y = screen_h - 30  # baseline the single-line version used to sit on
    start_y = base_y - line_height * (len(lines) - 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (margin, start_y + i * line_height), font, font_scale, (150, 150, 150), thickness)
    return frame


def collect_samples_for_point(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    point_px: tuple[int, int],
    gaze_extractor: GazeVectorExtractor,
    instruction: str = STILL_INSTRUCTION,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    # Diagnostic counters -- when a point captures few/zero samples, these
    # say *why* (no face in frame vs. blink-rejected vs. head-pose solve
    # rejected) instead of leaving it a silent mystery. A run of "no_face"
    # almost always means the pose asked for pushed the face out of the
    # webcam's detection cone (e.g. laptop webcams already look down at you
    # from above the screen, so tilting the chin down further is the
    # direction most likely to do this).
    frames_seen = 0
    no_face = 0
    blink_rejected = 0
    pose_rejected = 0

    _present(draw_target(screen_w, screen_h, point_px, instruction))
    cv2.waitKey(1)

    settle_deadline = time.time() + SETTLE_S
    while time.time() < settle_deadline:
        cam.read()  # drain frames so capture below starts fresh
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    capture_deadline = time.time() + CAPTURE_S
    while time.time() < capture_deadline:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        frames_seen += 1
        timestamp_ms = cam.frame_timestamp_ms()
        result = backend.detect(frame, timestamp_ms)
        if result is None:
            no_face += 1
            continue
        points_px = result.points_px()
        if not is_gaze_usable(points_px, min_ear=CALIBRATION_MIN_EAR):
            blink_rejected += 1
            continue  # mid-blink (or too-narrowed-to-trust) frame -- skip it rather than risk a
            # degraded iris reading becoming training data (see CALIBRATION_MIN_EAR above)
        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
        if head_pose is None:
            pose_rejected += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaze_vector = gaze_extractor.extract(frame, gray, points_px)
        if gaze_vector is None:
            no_face += 1  # e.g. l2cs: landmarks were usable but the face crop wasn't
            continue
        feature = build_feature_vector(
            gaze_vector,
            (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg),
            tuple(head_pose.translation_vector.flatten()),
        )
        features.append(feature)
        targets.append(np.array(point_px, dtype=np.float64))

        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    if len(features) < 10:
        print(
            f"    !! only {len(features)}/{frames_seen} frames usable "
            f"(no_face={no_face}, blink={blink_rejected}, pose_rejected={pose_rejected})"
        )
        if no_face > frames_seen * 0.5:
            print("       -- face was mostly not detected here; this pose may be pushing your face out of the webcam's view")
        print(
            "       -- discarding this point's samples: too few to trust. A handful of frames that barely "
            "squeaked past the blink check tend to be half-closed-eye frames with a degraded iris reading "
            "(often reading as a near-neutral/centered offset regardless of the true target), and the local "
            "correction stage is sensitive enough to a few such mislabeled samples that keeping them risks "
            "biasing nearby live predictions (e.g. true center gaze getting pulled toward this point's target)."
        )
        return [], []

    return features, targets


def run_sequence(
    cam,
    backend,
    pose_estimator,
    screen_w,
    screen_h,
    fractions,
    gaze_extractor: GazeVectorExtractor,
    instruction: str = STILL_INSTRUCTION,
    group_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_groups: list[int] = []
    for i, (fx, fy) in enumerate(fractions):
        point_px = (int(fx * screen_w), int(fy * screen_h))
        print(f"  Point {i + 1}/{len(fractions)} at {point_px}...")
        features, targets = collect_samples_for_point(
            cam, backend, pose_estimator, screen_w, screen_h, point_px, gaze_extractor, instruction
        )
        print(f"    captured {len(features)} samples")
        all_features.extend(features)
        all_targets.extend(targets)
        all_groups.extend([group_offset + i] * len(features))  # tag every sample with which point it came from

    # np.array([]) on an empty list collapses to shape (0,) rather than
    # (0, feature_dim)/(0, 2) -- harmless on its own, but np.concatenate()
    # against the running (N, feature_dim) accumulator in main() then fails
    # with a dimension mismatch. This happens for real whenever every single
    # point in a sequence captures zero samples (e.g. a pose variant that
    # loses face tracking throughout) -- not just a theoretical edge case.
    # Force the correct 2D shape explicitly so an all-empty sequence is a
    # harmless no-op contribution instead of a crash.
    if not all_features:
        return np.empty((0, FEATURE_DIMS[gaze_extractor.name])), np.empty((0, 2)), np.empty((0,), dtype=int)
    return np.array(all_features), np.array(all_targets), np.array(all_groups)


# --- Mouse tracking + click-to-start -----------------------------------


class _MouseTracker:
    """Tracks the live mouse position AND click events via
    cv2.setMouseCallback() on WINDOW_NAME, registered once in main().

    Reporting position this way -- rather than through a separate OS-level
    cursor-position API -- guarantees the coordinates are in the exact
    same pixel space as whatever was last shown with cv2.imshow (the same
    thing _map_real_to_logical() converts out of), since that's literally
    what OpenCV measures mouse events against. An earlier version of this
    file read the cursor via the OS directly (capture/cursor.py) and
    converted it on the assumption that it shared OpenCV's coordinate
    space -- that assumption wasn't reliable enough in practice (a
    reported dot/cursor misalignment persisted), so this sidesteps the
    question entirely by never asking the OS at all."""

    def __init__(self) -> None:
        self.pos: tuple[int, int] | None = None
        self.clicked = False
        self.click_pos: tuple[int, int] | None = None

    def reset_click(self) -> None:
        self.clicked = False
        self.click_pos = None

    def __call__(self, event: int, x: int, y: int, flags: int, param) -> None:
        self.pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.click_pos = (x, y)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _draw_path_with_arrows(
    frame: np.ndarray, path_px: list[tuple[float, float]], color: tuple[int, int, int] = (90, 90, 90), thickness: int = 2
) -> None:
    """Draws path_px as a single connected line, with a small arrowhead at
    the end of every segment showing the direction of travel -- so it's
    visually obvious which way to go at every turn, not just at the very
    start."""
    for i in range(len(path_px) - 1):
        a = (int(path_px[i][0]), int(path_px[i][1]))
        b = (int(path_px[i + 1][0]), int(path_px[i + 1][1]))
        seg_len = max(_euclidean(a, b), 1.0)
        tip_length = min(0.5, ARROW_TIP_PX / seg_len)
        cv2.arrowedLine(frame, a, b, color, thickness, line_type=cv2.LINE_AA, tipLength=tip_length)


def _draw_start_end_markers(frame: np.ndarray, path_px: list[tuple[float, float]]) -> None:
    start_px, end_px = path_px[0], path_px[-1]
    cv2.circle(frame, (int(start_px[0]), int(start_px[1])), POINT_RADIUS, (0, 220, 0), -1)
    cv2.putText(frame, "START", (int(start_px[0]) - 32, int(start_px[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, (int(end_px[0]), int(end_px[1])), POINT_RADIUS, (255, 255, 255), 2)
    cv2.putText(frame, "END", (int(end_px[0]) - 22, int(end_px[1]) - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def wait_for_click(
    cam: WebcamSource,
    screen_w: int,
    screen_h: int,
    display_size: tuple[int, int] | None,
    path_px: list[tuple[float, float]],
    mouse_tracker: _MouseTracker,
    label: str,
) -> None:
    """Shows the full guide pattern (with arrows and start/end markers) for
    the upcoming trace and waits for a single click near its start point
    before run_continuous_trace() begins collecting. Clicking far from the
    start point is just ignored (never a hard error) -- move closer and
    click again."""
    mouse_tracker.reset_click()
    start_px = path_px[0]
    font, font_scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 20

    while True:
        cam.read()  # keep draining the camera so the feed doesn't buffer stale frames once tracing starts
        frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        _draw_path_with_arrows(frame, path_px)
        _draw_start_end_markers(frame, path_px)
        cv2.circle(frame, (int(start_px[0]), int(start_px[1])), POINT_RADIUS + 6, (255, 255, 255), 1)

        instruction = f"Move your cursor to the green dot, then click to start tracing the {label}"
        lines = _wrap_line(instruction, font, font_scale, thickness, screen_w - 2 * margin)
        line_height = 26
        base_y = screen_h - 30
        start_y = base_y - line_height * (len(lines) - 1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (margin, start_y + i * line_height), font, font_scale, (150, 150, 150), thickness)

        _present(frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt

        if mouse_tracker.clicked:
            click_logical = _map_real_to_logical(mouse_tracker.click_pos, screen_w, screen_h, display_size)
            if _euclidean(click_logical, start_px) <= CLICK_START_RADIUS_PX:
                return
            mouse_tracker.reset_click()


# --- Continuous cursor-tracing pass (replaces the old discrete point-by-point capture) ---


def _lagged_cursor_position(
    history: list[tuple[float, tuple[float, float]]], target_time: float
) -> tuple[float, float] | None:
    """Looks up (via linear interpolation) where the cursor was at
    target_time, given `history` as a time-ordered list of (timestamp,
    (x, y)) samples. Returns None if target_time falls before the earliest
    sample -- not enough history yet to answer honestly, so the caller
    should skip labeling a sample this frame rather than guess."""
    if not history or target_time < history[0][0]:
        return None
    if target_time >= history[-1][0]:
        return history[-1][1]
    for (t0, p0), (t1, p1) in zip(history, history[1:]):
        if t0 <= target_time <= t1:
            if t1 == t0:
                return p0
            frac = (target_time - t0) / (t1 - t0)
            return (p0[0] + frac * (p1[0] - p0[0]), p0[1] + frac * (p1[1] - p0[1]))
    return history[-1][1]  # defensive fallback; the loop above should always find a bracketing pair


def _point_to_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Returns (distance from p to the segment a->b, the fraction t in
    [0, 1] along that segment of the closest point) -- the standard
    point-to-segment projection, clamped to the segment's own extent."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return _euclidean(p, a), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return _euclidean(p, closest), t


def _path_progress(point: tuple[float, float], path_px: list[tuple[float, float]]) -> tuple[float, float]:
    """Returns (distance from point to the nearest point ON the polyline
    path_px, progress fraction in [0, 1]) -- progress is the cumulative
    arc-length of that nearest point, divided by the path's total length,
    i.e. "how far along the line is the closest thing to where you are".
    Used both for the margin ("you've strayed") check and for tracking
    whether the pass has genuinely reached its end, not just that the
    cursor happens to be near the last point."""
    seg_lengths = [_euclidean(path_px[i], path_px[i + 1]) for i in range(len(path_px) - 1)]
    total_length = sum(seg_lengths) or 1.0

    best_dist = float("inf")
    best_progress_len = 0.0
    cum = 0.0
    for i, seg_len in enumerate(seg_lengths):
        dist, t = _point_to_segment_distance(point, path_px[i], path_px[i + 1])
        if dist < best_dist:
            best_dist = dist
            best_progress_len = cum + t * seg_len
        cum += seg_len
    return best_dist, best_progress_len / total_length


_PHASE_CURSOR_COLORS: dict[str, tuple[int, int, int]] = {
    "capturing": (0, 220, 0),
    "too_fast": (0, 0, 255),
    "off_path": (0, 165, 255),
}
_DEFAULT_CURSOR_COLOR = (255, 255, 255)


def _draw_continuous_trace_frame(
    screen_w: int,
    screen_h: int,
    path_px: list[tuple[float, float]],
    cursor_px: tuple[float, float],
    phase: str,
    message: str,
    progress: float,
) -> np.ndarray:
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    _draw_path_with_arrows(frame, path_px)
    _draw_start_end_markers(frame, path_px)

    cursor_color = _PHASE_CURSOR_COLORS.get(phase, _DEFAULT_CURSOR_COLOR)
    cv2.circle(frame, (int(cursor_px[0]), int(cursor_px[1])), CURSOR_MARKER_RADIUS, cursor_color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{min(progress, 1.0) * 100:.0f}%", (20, 40), font, 0.7, (150, 150, 150), 2)

    if message:
        (text_w, _), _ = cv2.getTextSize(message, font, 0.7, 2)
        cv2.putText(frame, message, (screen_w // 2 - text_w // 2, 70), font, 0.7, (0, 0, 255), 2)

    instr = "Trace the line with your cursor, looking at it the whole time"
    (iw, _), _ = cv2.getTextSize(instr, font, 0.6, 1)
    cv2.putText(frame, instr, (screen_w // 2 - iw // 2, screen_h - 30), font, 0.6, (150, 150, 150), 1)
    return frame


def run_continuous_trace(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    display_size: tuple[int, int] | None,
    path_px: list[tuple[float, float]],
    mouse_tracker: _MouseTracker,
    gaze_extractor: GazeVectorExtractor,
    group_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuously collects (gaze feature, target) samples while you trace
    path_px with your own mouse cursor -- no stopping required, and no
    time limit: the pass only ends once the cursor has both (a) covered at
    least MIN_PROGRESS_TO_FINISH of the path and (b) reached within
    END_RADIUS_PX of its final point (see _path_progress()). See the
    module docstring and _lagged_cursor_position() for why the recorded
    TARGET for each captured frame is the cursor's position from
    LAG_COMPENSATION_S *before* that frame, not its position at that exact
    instant."""
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[int] = []
    no_face = blink_rejected = pose_rejected = too_fast = not_enough_history = 0

    cursor_history: list[tuple[float, tuple[float, float]]] = []
    start_t = time.time()
    failsafe_deadline = start_t + MAX_PASS_FAILSAFE_S
    prev_cursor: tuple[float, float] | None = None
    prev_t: float | None = None
    max_progress = 0.0

    while True:
        now = time.time()
        ok, frame = cam.read()

        raw_pos = mouse_tracker.pos if mouse_tracker.pos is not None else path_px[0]
        cursor_logical = _map_real_to_logical(raw_pos, screen_w, screen_h, display_size)
        cursor_logical = (
            float(np.clip(cursor_logical[0], 0, screen_w)),
            float(np.clip(cursor_logical[1], 0, screen_h)),
        )

        cursor_history.append((now, cursor_logical))
        cutoff = now - CURSOR_HISTORY_MAX_S
        while cursor_history and cursor_history[0][0] < cutoff:
            cursor_history.pop(0)

        if prev_cursor is not None and prev_t is not None and now > prev_t:
            velocity = _euclidean(cursor_logical, prev_cursor) / (now - prev_t)
        else:
            velocity = 0.0
        prev_cursor, prev_t = cursor_logical, now

        distance_to_path, progress = _path_progress(cursor_logical, path_px)
        max_progress = max(max_progress, progress)

        too_fast_now = velocity > MAX_TRACE_SPEED_PX_S
        off_path = distance_to_path > PATH_MARGIN_PX
        phase, message = "tracing", ""
        if too_fast_now:
            phase, message = "too_fast", "Slow down a little"
        elif off_path:
            phase, message = "off_path", "You've strayed from the path -- move back towards the line"

        if too_fast_now:
            too_fast += 1
        else:
            points_px = None
            head_pose = None
            if ok and frame is not None:
                result = backend.detect(frame, cam.frame_timestamp_ms())
                if result is None:
                    no_face += 1
                else:
                    points_px = result.points_px()
                    if not is_gaze_usable(points_px, min_ear=CALIBRATION_MIN_EAR):
                        blink_rejected += 1
                    else:
                        head_pose = pose_estimator.estimate(points_px, result.image_width, result.image_height)
                        if head_pose is None:
                            pose_rejected += 1
            else:
                no_face += 1

            if head_pose is not None:
                lagged_target = _lagged_cursor_position(cursor_history, now - LAG_COMPENSATION_S)
                if lagged_target is None:
                    not_enough_history += 1
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gaze_vector = gaze_extractor.extract(frame, gray, points_px)
                    if gaze_vector is None:
                        no_face += 1  # e.g. l2cs: landmarks were usable but the face crop wasn't
                    else:
                        feature = build_feature_vector(
                            gaze_vector,
                            (head_pose.yaw_deg, head_pose.pitch_deg, head_pose.roll_deg),
                            tuple(head_pose.translation_vector.flatten()),
                        )
                        features.append(feature)
                        targets.append(np.array(lagged_target, dtype=np.float64))
                        groups.append(group_offset + int((now - start_t) // GROUP_WINDOW_S))
                        phase = "capturing"

        canvas = _draw_continuous_trace_frame(screen_w, screen_h, path_px, cursor_logical, phase, message, max_progress)
        _present(canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt

        reached_end = max_progress >= MIN_PROGRESS_TO_FINISH and _euclidean(cursor_logical, path_px[-1]) <= END_RADIUS_PX
        if reached_end:
            break
        if now > failsafe_deadline:
            print(f"    !! pass didn't reach the end within {MAX_PASS_FAILSAFE_S:.0f}s -- stopping with whatever was captured")
            break

    if no_face or blink_rejected or pose_rejected or too_fast or not_enough_history:
        print(
            f"    ({no_face} no-face, {blink_rejected} blink, {pose_rejected} pose-reject, "
            f"{too_fast} too-fast, {not_enough_history} not-enough-history frames skipped; "
            f"{len(features)} samples captured)"
        )
    else:
        print(f"    {len(features)} samples captured")

    if not features:
        return np.empty((0, FEATURE_DIMS[gaze_extractor.name])), np.empty((0, 2)), np.empty((0,), dtype=int)
    return np.array(features), np.array(targets), np.array(groups)


def _next_group_id(current: int, groups: np.ndarray) -> int:
    """Picks up group numbering where the last pass left off -- same
    "one past the max seen so far" idiom session_store.py's pool_sessions()
    already uses to keep group ids globally unique."""
    if len(groups) == 0:
        return current
    return int(groups.max()) + 1


# --- Protocol dispatch -------------------------------------------------
#
# Which of the two capture mechanics above actually runs is chosen in
# main() by --features: handcrafted -> _run_grid_training()/
# _run_grid_validation() (dot-by-dot, see module docstring), l2cs ->
# _run_trace_training()/_run_trace_validation() (continuous cursor-trace).
# Split into training/validation halves (rather than one do-everything
# function per protocol) so main() can save_session() the training data to
# disk *before* the validation pass runs -- same as both protocols always
# did -- so aborting (ESC) during validation doesn't lose a session that
# was otherwise already safely captured.


def _run_grid_training(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wait_for_front_ready(cam, backend, pose_estimator, screen_w, screen_h)
    print("Keep your head still & forward -- move only your eyes to each dot.")
    print("Training points:")
    all_features, all_targets, all_groups = run_sequence(
        cam, backend, pose_estimator, screen_w, screen_h, TRAIN_GRID_FRACTIONS, gaze_extractor, STILL_INSTRUCTION
    )
    next_group_id = len(TRAIN_GRID_FRACTIONS)

    print("\nHead-pose variation pass -- this teaches the model to keep tracking")
    print("correctly after you shift position, instead of only ever seeing you")
    print("sit still and forward.")
    for variant in POSE_VARIANTS:
        wait_for_pose_ready(cam, backend, pose_estimator, screen_w, screen_h, variant)
        print(f"  Variant: {variant.label}")
        features, targets, groups = run_sequence(
            cam,
            backend,
            pose_estimator,
            screen_w,
            screen_h,
            POSE_VARIATION_FRACTIONS,
            gaze_extractor,
            variant.instruction,
            group_offset=next_group_id,
        )
        next_group_id += len(POSE_VARIATION_FRACTIONS)
        if features.shape[0] == 0:
            print(
                f"    !! captured nothing usable for '{variant.label}' -- this variant contributes "
                "no data to the model. The pose was likely too extreme (face lost by the "
                "webcam) or held too briefly. Consider re-running with a gentler version of "
                "this pose."
            )
        all_features = np.concatenate([all_features, features])
        all_targets = np.concatenate([all_targets, targets])
        all_groups = np.concatenate([all_groups, groups])

    return all_features, all_targets, all_groups


def _run_grid_validation(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    gaze_extractor: GazeVectorExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    print("Validation points:")
    # Reuses wait_for_front_ready() (live camera + yaw/pitch/distance
    # readout + SPACE to continue) rather than a blank fixed-duration
    # message, so you can actually see you're back in frame and facing
    # forward before validation starts capturing.
    wait_for_front_ready(
        cam,
        backend,
        pose_estimator,
        screen_w,
        screen_h,
        instruction="Return your head to a normal, centered position -- keep tracking dots with your eyes only.",
    )
    val_features, val_targets, _val_groups = run_sequence(
        cam, backend, pose_estimator, screen_w, screen_h, VALIDATION_FRACTIONS, gaze_extractor
    )
    return val_features, val_targets


def _run_trace_training(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    mouse_tracker: _MouseTracker,
    gaze_extractor: GazeVectorExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    wait_for_front_ready(cam, backend, pose_estimator, screen_w, screen_h)

    main_path = [(fx * screen_w, fy * screen_h) for fx, fy in _zigzag_path_fractions()]
    print(
        "Click the green dot to begin, then trace the line -- there's no time limit, "
        "just trace until you reach the END marker."
    )
    wait_for_click(cam, screen_w, screen_h, _display_size, main_path, mouse_tracker, "line")
    all_features, all_targets, all_groups = run_continuous_trace(
        cam, backend, pose_estimator, screen_w, screen_h, _display_size, main_path, mouse_tracker, gaze_extractor,
    )
    next_group_id = _next_group_id(0, all_groups)

    print("\nHead-pose variation pass -- this teaches the model to keep tracking")
    print("correctly after you shift position, instead of only ever seeing you")
    print("sit still and forward.")
    pose_variant_path = [
        (fx * screen_w, fy * screen_h)
        for fx, fy in _zigzag_path_fractions(x_range=(0.15, 0.85), y_range=(0.25, 0.75), n_rows=3)
    ]
    for variant in POSE_VARIANTS:
        wait_for_pose_ready(cam, backend, pose_estimator, screen_w, screen_h, variant)
        print(f"  Variant: {variant.label}")
        wait_for_click(cam, screen_w, screen_h, _display_size, pose_variant_path, mouse_tracker, "line")
        features, targets, groups = run_continuous_trace(
            cam, backend, pose_estimator, screen_w, screen_h, _display_size, pose_variant_path, mouse_tracker,
            gaze_extractor, group_offset=next_group_id,
        )
        next_group_id = _next_group_id(next_group_id, groups)
        if features.shape[0] == 0:
            print(
                f"    !! captured nothing usable for '{variant.label}' -- this variant contributes "
                "no data to the model. The pose was likely too extreme (face lost by the "
                "webcam) or held too briefly. Consider re-running with a gentler version of "
                "this pose."
            )
        all_features = np.concatenate([all_features, features])
        all_targets = np.concatenate([all_targets, targets])
        all_groups = np.concatenate([all_groups, groups])

    return all_features, all_targets, all_groups, next_group_id


def _run_trace_validation(
    cam: WebcamSource,
    backend: MediaPipeLandmarkBackend,
    pose_estimator: HeadPoseEstimator,
    screen_w: int,
    screen_h: int,
    mouse_tracker: _MouseTracker,
    gaze_extractor: GazeVectorExtractor,
    next_group_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    wait_for_front_ready(
        cam,
        backend,
        pose_estimator,
        screen_w,
        screen_h,
        instruction="Return your head to a normal, centered position -- keep tracing with your eyes only.",
    )
    validation_path = [(fx * screen_w, fy * screen_h) for fx, fy in _vertical_zigzag_path_fractions()]
    wait_for_click(cam, screen_w, screen_h, _display_size, validation_path, mouse_tracker, "line")
    val_features, val_targets, _val_groups = run_continuous_trace(
        cam, backend, pose_estimator, screen_w, screen_h, _display_size, validation_path, mouse_tracker,
        gaze_extractor, group_offset=next_group_id,
    )
    return val_features, val_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="doomscroll gaze calibration")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Also fit against this session pooled with recent stored sessions (see session_store.py "
        "for why this is opt-in rather than the default), and keep whichever fit actually validates "
        "better. Prints both the session-only and pooled validation error so you can compare.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Delete all stored calibration sessions and exit. Do this after any real setup "
        "change (new lighting, seating, camera position).",
    )
    parser.add_argument(
        "--list-history",
        action="store_true",
        help="List stored calibration sessions and exit.",
    )
    parser.add_argument(
        "--features",
        choices=FEATURE_BACKEND_NAMES,
        default="handcrafted",
        help="Which gaze-vector pipeline to capture with (default: handcrafted). 'l2cs' uses "
        "L2CS-Net's pretrained appearance-based (yaw, pitch) estimate instead of the original "
        "hand-crafted eye-in-head offset -- an offline comparison motivated adding it (see the "
        "README's Gaze feature backends section), and src/doomscroll/gaze/l2cs_backend.py for what "
        "it needs installed (`uv sync --extra l2cs`). Whatever you pick here, run scripts/phase2_gaze_scroll.py "
        "with the SAME --features value -- a saved profile is tagged with the backend it was "
        "captured with and the live script will refuse to use it otherwise.",
    )
    args = parser.parse_args()

    if args.clear_history:
        removed = clear_sessions()
        print(f"Cleared {removed} stored calibration session(s).")
        return

    if args.list_history:
        sessions = list_sessions()
        if not sessions:
            print("No stored calibration sessions.")
        for s in sessions:
            print(
                f"  {s.label}  {len(s.features):5d} samples  screen={s.screen_size[0]}x{s.screen_size[1]}  "
                f"features={s.feature_backend}"
            )
        return

    ensure_dpi_aware()
    screen_w, screen_h = get_screen_size()
    print(f"Screen size: {screen_w}x{screen_h}")

    backend = MediaPipeLandmarkBackend()
    pose_estimator = HeadPoseEstimator()
    # Built up front, before the camera even opens, so a missing torch/L2CS-Net
    # install (or missing weights) fails fast with a clear message rather than
    # after you've already gone through the face-in-frame check.
    print(f"Feature backend: {args.features}")
    gaze_extractor = make_gaze_extractor(args.features)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    mouse_tracker = _MouseTracker()
    cv2.setMouseCallback(WINDOW_NAME, mouse_tracker)

    global _display_size
    _display_size = _detect_display_size(screen_w, screen_h)
    if _display_size != (screen_w, screen_h):
        print(f"Note: fullscreen window renders at {_display_size[0]}x{_display_size[1]} "
              f"vs logical {screen_w}x{screen_h}; scaling frames (and click readings) to agree with it.")

    try:
        with WebcamSource(0) as cam:
            print("Calibration starting. ESC to abort.")

            # Which capture mechanic actually runs depends on --features --
            # see the module docstring and the "Protocol dispatch" section
            # above. next_group_id is only meaningful for the trace
            # protocol (so its validation pass's group ids don't collide
            # with training's); the grid protocol's validation doesn't need
            # it, matching how it always worked.
            if args.features == "l2cs":
                all_features, all_targets, all_groups, next_group_id = _run_trace_training(
                    cam, backend, pose_estimator, screen_w, screen_h, mouse_tracker, gaze_extractor
                )
            else:
                all_features, all_targets, all_groups = _run_grid_training(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor
                )
                next_group_id = 0

            # Always persist this session's raw samples, regardless of
            # --pool: storing them costs little and makes pooling available
            # on a later run, whereas discarding them now is irreversible.
            # Deliberately BEFORE the validation pass below, so aborting
            # (ESC) during validation doesn't lose training data that was
            # otherwise already safely captured.
            save_session(all_features, all_targets, all_groups, (screen_w, screen_h), feature_backend=args.features)
            print(f"\nSession saved ({len(all_features)} samples).")

            def fit_and_report(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, label: str) -> tuple[GazeCalibrator, float | None]:
                calibrator = GazeCalibrator(feature_backend=args.features)
                calibrator.fit(features, targets, groups=groups)
                n_points = len(np.unique(groups))
                print(f"\n[{label}] fitted on {len(features)} samples across {n_points} time-window groups.")
                print(f"[{label}] RidgeCV selected alpha (x, y): {calibrator.selected_alpha}")

                # Gain/compression check: a heavily-regularized fit can
                # minimize validation error while still shrinking the
                # model's dynamic range. Compare predicted spread to actual
                # target spread directly, over the training data.
                train_preds = np.array([calibrator.predict(f) for f in features])
                target_range = targets.max(axis=0) - targets.min(axis=0)
                pred_range = train_preds.max(axis=0) - train_preds.min(axis=0)
                gain = pred_range / np.where(target_range == 0, 1, target_range)
                print(f"[{label}] gain (predicted range / actual range), x,y: {gain} -- 1.0 is ideal, well below 1.0 means compressed")

                err = None
                if len(val_features) > 0:
                    err = calibrator.validate(val_features, val_targets)
                    print(
                        f"[{label}] validation mean error: {err:.1f}px "
                        f"({100 * err / screen_diag:.1f}% of screen diagonal)"
                    )
                return calibrator, err

            # Without this, validation starts immediately after the last
            # pose variant (tilt DOWN/the last grid point) with only the
            # small grey status text changing -- easy to miss. Both
            # protocols' validation helpers reuse wait_for_front_ready()
            # (live camera + yaw/pitch/distance readout + SPACE to
            # continue) rather than a blank message, so you can actually
            # see you're back in frame and facing forward before
            # validation starts.
            if args.features == "l2cs":
                val_features, val_targets = _run_trace_validation(
                    cam, backend, pose_estimator, screen_w, screen_h, mouse_tracker, gaze_extractor, next_group_id
                )
            else:
                val_features, val_targets = _run_grid_validation(
                    cam, backend, pose_estimator, screen_w, screen_h, gaze_extractor
                )
            screen_diag = (screen_w**2 + screen_h**2) ** 0.5

            session_calibrator, session_err = fit_and_report(
                all_features, all_targets, all_groups, "this session only"
            )
            best_calibrator, best_err, best_label = session_calibrator, session_err, "this session only"

            if args.pool:
                pooled_features, pooled_targets, pooled_groups, used = pool_sessions(
                    list_sessions(), (screen_w, screen_h), feature_backend=args.features
                )
                if len(used) <= 1:
                    print(
                        "\nNothing to pool yet (this is the only stored session at this screen size "
                        f"and --features {args.features!r})."
                    )
                else:
                    print(f"\nPooling {len(used)} sessions: " + ", ".join(s.label for s in used))
                    pooled_calibrator, pooled_err = fit_and_report(
                        pooled_features, pooled_targets, pooled_groups, "pooled"
                    )
                    # Keep whichever actually validates better -- pooling is
                    # not assumed to help (see session_store.py docstring).
                    if pooled_err is not None and session_err is not None and pooled_err < session_err:
                        best_calibrator, best_err, best_label = pooled_calibrator, pooled_err, "pooled"
                    elif pooled_err is not None and session_err is not None:
                        print(f"\nPooled fit did not improve on this session alone -- keeping '{best_label}'.")

            if best_err is not None and best_err / screen_diag > 0.08:
                print("\nAccuracy looks poor -- consider recalibrating with better/more even lighting.")

            if len(val_features) == 0:
                print("\nNo validation samples captured (face not detected during validation).")

            # Per-backend path (not the old shared DEFAULT_PROFILE_PATH) --
            # see calibration.py's profile_path_for_backend() docstring:
            # calibrating with one --features value must never silently
            # overwrite the other's saved profile.
            profile_path = profile_path_for_backend(args.features)
            best_calibrator.save(profile_path)
            print(f"Calibration profile saved to {profile_path} (using: {best_label}).")
    except KeyboardInterrupt:
        print("Calibration aborted.")
    finally:
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
