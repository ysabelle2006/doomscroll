"""Pure (camera-free) trajectory math for the continuous anchor+raster
calibration protocol (--protocol trajectory in scripts/calibrate.py).

Kept separate from the capture loop for the same reason the rest of this
package separates pure math from camera-touching code: the path geometry,
keyframe timing, turn detection, target interpolation and the
velocity/comfort math below are all unit-testable without a webcam, which
the capture loop around them is not.

See scripts/calibrate.py's module docstring for how this protocol is
wired in alongside the dot-grid one.

Design:
  - the path is a continuous, programmed (not free-roaming) sweep that
    visits the whole screen, guaranteeing even coverage within a fixed
    time budget.
  - human smooth pursuit degrades into catch-up saccades above roughly
    COMFORTABLE_PURSUIT_DEG_S (see below) of angular velocity, which is
    both unpleasant to follow and a source of label noise (the fixed-lag
    label model assumes continuous pursuit, which breaks down during a
    genuine saccade). Anchor-to-anchor transitions are paced to this
    comfortable angular velocity -- build_keyframes() derives one
    constant angular velocity for the WHOLE path (anchors and raster
    segments alike) from total_duration_ms and the path's total
    visual-angle length, rather than a dedicated anchor-only speed
    constant. Every anchor arrival also gets an explicit settle window
    (ANCHOR_SETTLE_MS) during which frames are discarded regardless of
    the generic turn-angle heuristic, on top of the longer dwell that
    follows it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Anchors: 4 corners + center + bottom-center, in an order chosen so each
# hop changes direction from the last one -- visiting order matters, not
# just which points are included. Visited first, each with a short dwell,
# before the raster sweep covers the rest of the screen.
ANCHOR_FRACTIONS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),    # center (start)
    (0.08, 0.08),  # top-left
    (0.92, 0.08),  # top-right
    (0.92, 0.92),  # bottom-right
    (0.08, 0.92),  # bottom-left
    (0.5, 0.92),   # bottom-center
    (0.5, 0.5),    # center (back, before the raster)
)

# Raster sweep: boustrophedon (alternating direction each row) rows, for
# full-screen coverage -- see module docstring. This is what a handful of
# static dots can't give.
#
# Column count and range are chosen to land exactly 2 columns in each of the
# diagnostics' 5 coverage cells (cell width 0.2; columns at 0.05, 0.15, 0.25,
# ..., 0.95). A 7-column grid that doesn't line up with the diagnostic grid
# looked uneven in practice even before any discarding, because some cells
# got 1 raster column crossing them and others got 2.
RASTER_ROW_Y_FRACTIONS: tuple[float, ...] = (0.15, 0.35, 0.55, 0.75, 0.92)
RASTER_COLS = 10
RASTER_X_RANGE: tuple[float, float] = (0.05, 0.95)

# Anchor dwell is split into an unconditionally-discarded SETTLE window
# (reaction time + the catch-up saccade landing on the new anchor -- long
# jumps need more of this than a small raster zigzag does, see
# ANCHOR_SETTLE_MS) followed by the remaining dwell, which is trusted as a
# clean, static-target label. Total raised from the original 500ms so a
# useful trusted window (~350ms) still remains after the settle cut.
ANCHOR_DWELL_MS = 800.0
# Measured (via simulate_pursuit_gaze) against a bounded-velocity pursuit
# model: once transitions are paced under COMFORTABLE_PURSUIT_DEG_S (see
# build_keyframes()'s constant-velocity pacing below), widening this window
# from 250ms to 450-600ms
# gave NO further reduction in simulated label error -- the dwell region is
# already low-noise once pacing is fixed, so a longer cut mostly just
# discards good, clean samples. Set to the low end of the requested
# 300-500ms range: a safety margin for real reaction-time variance that
# this simplified simulation doesn't capture, without the larger sample
# cost a bigger window measured as pure waste.
ANCHOR_SETTLE_MS = 300.0

# How long the dot stops moving, standing still, right before any sharp
# direction change along the path -- not just at an anchor (which already
# pauses via ANCHOR_DWELL_MS above), but at every raster-internal turn too
# (e.g. a row reversal, or swapping from a horizontal sweep to a vertical
# one): continuing straight through those at full speed, with only a
# post-hoc discard window over the resulting samples (see turn_discard_mask
# / TURN_DISCARD_MS in scripts/calibrate.py), felt abrupt to actually
# follow. A real stationary pause here gives the eye an explicit moment to
# arrive and settle before the new direction starts, mirroring the anchor
# dwell's own settle-then-trust pattern (see build_keyframes) instead of
# asking the eye to redirect while still in motion.
TURN_PAUSE_MS = 500.0

# Smooth pursuit's commonly-cited comfort ceiling: below this, people track
# a moving target with their eyes alone (low gain loss); above it, pursuit
# increasingly degrades into catch-up saccades, which is both unpleasant to
# follow and breaks the fixed-lag label model's continuous-pursuit
# assumption (see module docstring).
COMFORTABLE_PURSUIT_DEG_S = 30.0
# There is deliberately no separate "anchor transition speed" constant.
# build_keyframes() paces every segment of the path -- anchor transitions
# and raster segments alike -- at ONE constant angular velocity, derived
# from total_duration_ms and the path's total visual-angle length, so the
# whole thing moves at a single, predictable speed throughout. Tune the
# speed by changing total_duration_ms (a longer budget over the same path
# means a slower, more comfortable pace).

TURN_ANGLE_THRESHOLD_DEG = 50.0  # direction changes sharper than this count as a "turn"

# Assumptions used ONLY to pace the calibration animation in physical visual
# terms (never for label geometry, which stays in screen fractions) -- a
# reasonable 13-15" laptop panel at a normal viewing distance. Real screens
# vary; pass screen_w_cm/view_dist_cm explicitly for a closer match on a
# very different setup (e.g. a large external monitor).
ASSUMED_SCREEN_W_PX = 1920
ASSUMED_SCREEN_H_PX = 1080
ASSUMED_SCREEN_W_CM = 34.5
ASSUMED_VIEW_DIST_CM = 60.0


def visual_angle_deg(
    dx_frac: float,
    dy_frac: float,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> float:
    """The visual angle (degrees) subtended by a screen-fraction displacement
    (dx_frac, dy_frac), at a fixed viewing distance. Used to pace/measure
    target motion in human terms -- see module docstring."""
    cm_per_px = screen_w_cm / screen_w_px
    dx_cm = dx_frac * screen_w_px * cm_per_px
    dy_cm = dy_frac * screen_h_px * cm_per_px
    dist_cm = float(np.hypot(dx_cm, dy_cm))
    return float(np.degrees(np.arctan2(dist_cm, view_dist_cm)))


@dataclass(frozen=True)
class Keyframes:
    """Piecewise-linear target path: t_ms strictly increasing, x/y in screen
    fractions. turn_t_ms holds the times of sharp direction changes along
    the path (e.g. raster row reversals); pause_arrival_t_ms holds the
    times the path arrives at any point where it actually stops moving for
    a while -- an anchor (ANCHOR_DWELL_MS) or a sharp turn (TURN_PAUSE_MS,
    below) alike -- pre-pause, so turn_discard_mask() can cut the same kind
    of settle window after either one (see ANCHOR_SETTLE_MS's docstring
    note and run_trajectory_capture())."""

    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    turn_t_ms: np.ndarray
    pause_arrival_t_ms: np.ndarray = field(default_factory=lambda: np.array([]))

    def target_at(self, query_t_ms):
        """Interpolated target fraction(s) at arbitrary time(s). Returns a
        (2,) array for a scalar query, or (N, 2) for an array-like query."""
        q = np.atleast_1d(np.asarray(query_t_ms, dtype=np.float64))
        out = np.column_stack([np.interp(q, self.t_ms, self.x), np.interp(q, self.t_ms, self.y)])
        return out[0] if np.ndim(query_t_ms) == 0 else out

    @property
    def duration_ms(self) -> float:
        return float(self.t_ms[-1])


@dataclass(frozen=True)
class VelocityStats:
    """Per-segment target speed (time-weighted over the whole path), in
    degrees/second of visual angle -- see measure_velocity."""

    segment_dt_ms: np.ndarray
    segment_deg_s: np.ndarray
    mean_deg_s: float
    p90_deg_s: float
    max_deg_s: float

    def segments_over(self, threshold_deg_s: float) -> np.ndarray:
        """Boolean mask of (moving, i.e. dt>0) segments exceeding threshold_deg_s."""
        moving = self.segment_dt_ms > 1e-6
        over = np.zeros_like(moving)
        over[moving] = self.segment_deg_s[moving] > threshold_deg_s
        return over


def measure_velocity(
    keyframes: Keyframes,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> VelocityStats:
    """Measures the target's own angular velocity along the path -- not a
    simulation of the EYE (see simulate_pursuit_gaze for that), just "how
    fast does the dot itself move", segment by segment. Speed is constant
    within a segment (piecewise-linear motion), so this is exact, not a
    finite-difference approximation."""
    dt_ms = np.diff(keyframes.t_ms)
    dx = np.diff(keyframes.x)
    dy = np.diff(keyframes.y)
    moving = dt_ms > 1e-6
    deg = np.zeros_like(dt_ms)
    deg[moving] = [
        visual_angle_deg(dx[i], dy[i], screen_w_px, screen_h_px, screen_w_cm, view_dist_cm)
        for i in np.flatnonzero(moving)
    ]
    deg_s = np.zeros_like(dt_ms)
    deg_s[moving] = deg[moving] / (dt_ms[moving] / 1000.0)

    w = dt_ms[moving]
    v = deg_s[moving]
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum = np.cumsum(w_sorted)
    cum = cum / cum[-1] if cum[-1] > 0 else cum
    p90_idx = min(np.searchsorted(cum, 0.9), len(v_sorted) - 1)

    return VelocityStats(
        segment_dt_ms=dt_ms,
        segment_deg_s=deg_s,
        mean_deg_s=float(np.average(v, weights=w)) if len(v) else 0.0,
        p90_deg_s=float(v_sorted[p90_idx]) if len(v_sorted) else 0.0,
        max_deg_s=float(deg_s.max()) if len(deg_s) else 0.0,
    )


def build_waypoints() -> list[tuple[float, float]]:
    """Anchors, then the raster sweep, as a flat list of (x_frac, y_frac).
    The anchors are the first len(ANCHOR_FRACTIONS) entries -- build_keyframes
    relies on that ordering to know which waypoints get a dwell."""
    waypoints = list(ANCHOR_FRACTIONS)
    for i, y in enumerate(RASTER_ROW_Y_FRACTIONS):
        xs = np.linspace(RASTER_X_RANGE[0], RASTER_X_RANGE[1], RASTER_COLS)
        if i % 2 == 1:
            xs = xs[::-1]
        waypoints.extend((float(x), float(y)) for x in xs)
    return waypoints


def build_keyframes(
    total_duration_ms: float,
    waypoints: list[tuple[float, float]] | None = None,
    n_anchors: int | None = None,
    anchor_dwell_ms: float = ANCHOR_DWELL_MS,
    turn_pause_ms: float = TURN_PAUSE_MS,
    turn_angle_threshold_deg: float = TURN_ANGLE_THRESHOLD_DEG,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> Keyframes:
    """Lay the waypoints out in time, moving the target at ONE constant
    angular velocity (degrees/second of visual angle) for the entire path
    -- anchor-to-anchor transitions and raster segments alike. Each
    segment's travel time is simply proportional to its own visual-angle
    length; the shared constant speed is whatever value makes the whole
    path (travel time plus anchor dwell plus turn pauses) fit
    total_duration_ms. Each anchor arrival gets `anchor_dwell_ms` of
    standstill; each OTHER sharp direction change (a raster row reversal,
    swapping from a horizontal sweep to a vertical one, etc. -- see
    is_turn_waypoint below) gets a shorter `turn_pause_ms` standstill
    instead, so the dot always actually stops before changing direction
    rather than turning through it at full speed. An anchor that also
    happens to be a sharp turn only gets the one (longer) anchor pause,
    never both.

    Pacing every segment off the same constant speed means there's no
    fixed-cost component that could collapse any one segment's share of
    total_duration_ms, and no anchor/raster distinction is needed here at
    all (n_anchors still controls which waypoints get a dwell, below)."""
    waypoints = waypoints if waypoints is not None else build_waypoints()
    n_anchors = n_anchors if n_anchors is not None else len(ANCHOR_FRACTIONS)
    pts = np.array(waypoints, dtype=np.float64)
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 waypoints to build a path")

    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len_safe = np.where(seg_len < 1e-9, 1e-9, seg_len)  # guard a (near-)repeated waypoint

    seg_deg = np.array(
        [
            visual_angle_deg(seg_vec[i, 0], seg_vec[i, 1], screen_w_px, screen_h_px, screen_w_cm, view_dist_cm)
            for i in range(n - 1)
        ]
    )
    seg_deg_safe = np.where(seg_deg < 1e-9, 1e-9, seg_deg)  # guard a (near-)repeated waypoint, same as seg_len_safe

    # Which interior waypoints are a sharp direction change -- computed up
    # front, from direction vectors alone, so the main timing loop below
    # can decide per-waypoint whether to insert a turn pause (TURN_PAUSE_MS)
    # without needing a second pass. An anchor arrival is deliberately
    # excluded here even if its own incoming/outgoing turn is sharp: it
    # already gets the longer ANCHOR_DWELL_MS pause, and stacking a second,
    # shorter pause on top of that would just be redundant standstill time.
    dirs = seg_vec / seg_len_safe[:, None]
    is_turn_waypoint = np.zeros(n, dtype=bool)
    for j in range(1, n - 1):
        cos_angle = np.clip(np.dot(dirs[j - 1], dirs[j]), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg >= turn_angle_threshold_deg:
            is_turn_waypoint[j] = True
    n_turn_pauses = sum(
        1 for j in range(1, n - 1) if is_turn_waypoint[j] and not (1 <= j <= n_anchors - 1)
    )

    dwell_total_ms = anchor_dwell_ms * max(n_anchors - 1, 0)
    dwell_total_ms = min(dwell_total_ms, total_duration_ms * 0.5)  # never let dwell eat the whole budget
    turn_pause_total_ms = turn_pause_ms * n_turn_pauses
    turn_pause_total_ms = min(turn_pause_total_ms, total_duration_ms * 0.25)  # same kind of safety cap

    # Every segment's share of the travel budget is proportional to its own
    # visual-angle length -- i.e. degrees-per-ms, and therefore degrees-per-
    # second, is the SAME constant for every segment in the path.
    travel_budget_ms = max(total_duration_ms - dwell_total_ms - turn_pause_total_ms, 1.0)
    travel_ms = seg_deg_safe * (travel_budget_ms / seg_deg_safe.sum())

    t_ms = [0.0]
    xs = [pts[0, 0]]
    ys = [pts[0, 1]]
    arrival_t = [0.0]  # arrival_t[j] = time the path first reaches waypoint j, pre-pause
    pause_arrivals = []  # any waypoint the path actually stops at -- anchor dwell or turn pause
    t = 0.0
    for i in range(n - 1):
        t += travel_ms[i]
        j = i + 1
        arrival_t.append(t)
        t_ms.append(t)
        xs.append(pts[j, 0])
        ys.append(pts[j, 1])
        if 1 <= j <= n_anchors - 1:
            pause_arrivals.append(t)
            t += anchor_dwell_ms
            t_ms.append(t)
            xs.append(pts[j, 0])
            ys.append(pts[j, 1])
        elif is_turn_waypoint[j]:
            pause_arrivals.append(t)
            t += turn_pause_ms
            t_ms.append(t)
            xs.append(pts[j, 0])
            ys.append(pts[j, 1])

    turn_times = [arrival_t[j] for j in range(1, n - 1) if is_turn_waypoint[j]]

    return Keyframes(
        t_ms=np.array(t_ms, dtype=np.float64),
        x=np.array(xs, dtype=np.float64),
        y=np.array(ys, dtype=np.float64),
        turn_t_ms=np.array(turn_times, dtype=np.float64),
        pause_arrival_t_ms=np.array(pause_arrivals, dtype=np.float64),
    )


def turn_discard_mask(label_t_ms: np.ndarray, turn_t_ms: np.ndarray, discard_window_ms: float) -> np.ndarray:
    """True where label_t_ms falls within discard_window_ms AFTER some
    marked time (a sharp direction change, or a pause arrival -- see
    Keyframes.turn_t_ms / pause_arrival_t_ms). These are frames where the
    label is a target position the eye hasn't caught up with yet (reaction
    time plus the catch-up saccade) rather than a genuine fixation. Used
    with two different windows and time arrays -- a shorter one for raster
    direction changes (turn_t_ms, which mostly overlaps an actual turn
    pause too, see TURN_PAUSE_MS -- the two windows are redundant there,
    not conflicting), a longer settle window for anywhere the path actually
    stops (pause_arrival_t_ms: anchors AND turns alike) -- see
    scripts/calibrate.py's run_trajectory_capture()."""
    if len(turn_t_ms) == 0:
        return np.zeros(len(label_t_ms), dtype=bool)
    discard = np.zeros(len(label_t_ms), dtype=bool)
    for t_turn in turn_t_ms:
        discard |= (label_t_ms >= t_turn) & (label_t_ms < t_turn + discard_window_ms)
    return discard


def simulate_pursuit_gaze(
    keyframes: Keyframes,
    query_t_ms: np.ndarray,
    reaction_ms: float = 200.0,
    max_speed_deg_s: float = COMFORTABLE_PURSUIT_DEG_S,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> np.ndarray:
    """A deliberately simple bounded-velocity model of where a human eye
    would actually be (screen fractions) while following this path, for
    estimating how much label noise a trajectory design introduces --
    NOT a real oculomotor model. The simulated gaze does nothing for the
    first `reaction_ms` after the target starts a new segment (saccadic/
    pursuit reaction latency), then moves directly toward the target's
    CURRENT position at up to `max_speed_deg_s` -- i.e. it tracks target
    motion under the comfort ceiling exactly, but falls behind (and stays
    behind) whenever the target outruns that cap, which is exactly the
    failure mode a too-fast transition introduces. Comparing this against
    the fixed-lag label (target position at t - lag_ms) on a fine time grid
    estimates how much of that label is actually wrong -- see
    run_trajectory_capture()'s lag correction."""
    dt_ms = 1000.0 / 60.0  # simulate at 60Hz internally regardless of query_t_ms's own spacing
    t = 0.0
    gaze = np.array(keyframes.target_at(0.0), dtype=np.float64)
    sim_t, sim_gaze = [0.0], [gaze.copy()]
    end_t = max(float(query_t_ms.max()), keyframes.duration_ms)
    cm_per_px = screen_w_cm / screen_w_px
    while t < end_t:
        t = min(t + dt_ms, end_t)
        target = keyframes.target_at(t - min(reaction_ms, t))  # reacts to where the target WAS reaction_ms ago
        delta = target - gaze
        dx_cm = delta[0] * screen_w_px * cm_per_px
        dy_cm = delta[1] * screen_h_px * cm_per_px
        remaining_deg = float(np.degrees(np.arctan2(np.hypot(dx_cm, dy_cm), view_dist_cm)))
        max_step_deg = max_speed_deg_s * (dt_ms / 1000.0)
        if remaining_deg <= max_step_deg or remaining_deg < 1e-9:
            gaze = target
        else:
            gaze = gaze + delta * (max_step_deg / remaining_deg)
        sim_t.append(t)
        sim_gaze.append(gaze.copy())

    sim_t_arr = np.array(sim_t)
    sim_gaze_arr = np.array(sim_gaze)
    qx = np.interp(query_t_ms, sim_t_arr, sim_gaze_arr[:, 0])
    qy = np.interp(query_t_ms, sim_t_arr, sim_gaze_arr[:, 1])
    return np.column_stack([qx, qy])
