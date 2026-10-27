"""Pure (camera-free) trajectory math for the continuous anchor+raster
calibration path used by `scripts/quick_calibrate.py --full`.

Kept separate from the capture loop for the same reason `ml/labels.py` is
kept pure numpy ("no camera" -- see that module's docstring): the path
geometry, keyframe timing, turn detection, target interpolation and the
velocity/comfort math below are all unit-testable without a webcam, which
the capture loop around them is not.

Design background (see quick_calibrate.py / calibration.py for the sibling
single-angle and multi-angle approaches, and the written-up findings this
pipeline is built from):
  - more dots alone does not help a small calibration fit -- so this adds
    *coverage* (a continuous path visiting the whole screen), not more
    static points.
  - uncontrolled 15s moving-target calibration performed badly because
    screen coverage was poor -- this path is deliberately programmed (not
    free-roaming) specifically to guarantee even coverage in a fixed
    20-30s budget.
  - 30-60s of continuous, good-coverage collection matches or beats static-
    dot calibration -- coverage matters more than point count.

A later real run showed an even coverage heatmap isn't enough on its own:
making anchor-to-anchor transitions artificially fast (to stop them biasing
the coverage heatmap, see history below) made the dot move at up to ~68
deg/s -- well past the ~30 deg/s range past which human smooth pursuit
degrades into catch-up saccades -- which is both unpleasant to follow and a
source of label noise (the fixed-lag label model assumes continuous
pursuit, which breaks down during a genuine saccade). Anchor-to-anchor
transitions are now PACED to a comfortable target angular velocity
(ANCHOR_TRANSITION_DEG_S) instead of raced through, and every anchor
arrival gets an explicit settle window (ANCHOR_SETTLE_MS) during which
frames are discarded regardless of the generic turn-angle heuristic, on top
of the longer dwell that follows it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Anchors: 4 corners + center + bottom-center, in an order chosen so each
# hop changes direction from the last one (see "varying direction and
# speed" below -- visiting order matters, not just which points are
# included). Visited first, each with a short dwell, before the raster
# sweep covers the rest of the screen.
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
# ANCHOR_TRANSITION_DEG_S), widening this window from 250ms to 450-600ms
# gave NO further reduction in simulated label error -- the dwell region is
# already low-noise once pacing is fixed, so a longer cut mostly just
# discards good, clean samples. Set to the low end of the requested
# 300-500ms range: a safety margin for real reaction-time variance that
# this simplified simulation doesn't capture, without the larger sample
# cost a bigger window measured as pure waste.
ANCHOR_SETTLE_MS = 300.0

# Per-segment speed multiplier, cycled across the raster's OWN segments only
# (intra-row sweeps and the short row-to-row reversals -- see
# _raster_segment_multipliers), so consecutive stretches of the sweep don't
# all move at the same speed ("varying ... speed" in the design brief).
# >1 = slower (more ms for the same distance), <1 = faster. Kept inside a
# range that stays under COMFORTABLE_PURSUIT_DEG_S for a typical raster
# segment -- see measure_velocity.
SPEED_MULTIPLIER_CYCLE: tuple[float, ...] = (1.0, 0.55, 1.6, 0.8)

# Smooth pursuit's commonly-cited comfort ceiling: below this, people track
# a moving target with their eyes alone (low gain loss); above it, pursuit
# increasingly degrades into catch-up saccades, which is both unpleasant to
# follow and breaks the fixed-lag label model's continuous-pursuit
# assumption (see module docstring).
COMFORTABLE_PURSUIT_DEG_S = 30.0
# Target angular velocity for anchor-to-anchor transitions (and the hop
# leaving the last anchor to enter the raster) -- paced well under the
# comfort ceiling, since these are long, often whole-screen jumps, not a
# short raster correction. Each such hop's duration is computed directly
# from this target (see build_keyframes), not from a length-proportional
# multiplier share of a fixed time budget.
ANCHOR_TRANSITION_DEG_S = 18.0

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
    the path (e.g. raster row reversals); anchor_arrival_t_ms holds the
    times the path arrives at each anchor, pre-dwell -- both feed
    direction-change/settle discard logic (see turn_discard_mask)."""

    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    turn_t_ms: np.ndarray
    anchor_arrival_t_ms: np.ndarray = field(default_factory=lambda: np.array([]))

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


def _raster_segment_multipliers(raster_rows: int, raster_cols: int, speed_cycle: tuple[float, ...]) -> np.ndarray:
    """Per-segment speed multiplier for the raster's OWN segments: both the
    intra-row sweeps and the short row-to-row reversals, which now get the
    *same* continuously-varying treatment (unlike anchor-to-anchor transit,
    which is paced separately by target angular velocity -- see
    build_keyframes). An earlier version gave row-transitions a dedicated
    fast or slow multiplier; both were measured worse (fast made the dot
    jump too quickly to follow at the one moment it's reversing direction;
    slow made it linger and reintroduced a coverage spike), so a row
    transition is now just an ordinary raster segment like any other."""
    n_raster_segments = raster_rows * raster_cols - 1  # edges among the raster's OWN points (excludes the entry hop)
    return np.array([speed_cycle[i % len(speed_cycle)] for i in range(n_raster_segments)])


def build_keyframes(
    total_duration_ms: float,
    waypoints: list[tuple[float, float]] | None = None,
    n_anchors: int | None = None,
    anchor_dwell_ms: float = ANCHOR_DWELL_MS,
    speed_multipliers: tuple[float, ...] = SPEED_MULTIPLIER_CYCLE,
    anchor_transition_deg_s: float = ANCHOR_TRANSITION_DEG_S,
    turn_angle_threshold_deg: float = TURN_ANGLE_THRESHOLD_DEG,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> Keyframes:
    """Lay the waypoints out in time.

    For the default anchor+raster path, anchor-to-anchor transit (plus the
    hop leaving the last anchor to enter the raster) gets an ABSOLUTE
    duration computed from anchor_transition_deg_s -- a fixed, comfortable
    pacing, independent of the overall time budget -- while the raster's
    own segments split whatever time budget remains, proportional to
    on-screen length times a cycling speed multiplier (varying speed without
    letting any one segment or row dominate the frame count -- see
    _raster_segment_multipliers). Each anchor arrival also gets
    `anchor_dwell_ms` of standstill.

    For a custom path (waypoints passed explicitly), there's no anchor/
    raster distinction to exploit, so every segment falls back to the
    simpler length-times-cycling-multiplier split of the whole travel
    budget."""
    using_default_raster = waypoints is None
    waypoints = waypoints if waypoints is not None else build_waypoints()
    n_anchors = n_anchors if n_anchors is not None else len(ANCHOR_FRACTIONS)
    pts = np.array(waypoints, dtype=np.float64)
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 waypoints to build a path")

    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len_safe = np.where(seg_len < 1e-9, 1e-9, seg_len)  # guard a (near-)repeated waypoint

    dwell_total_ms = anchor_dwell_ms * max(n_anchors - 1, 0)
    dwell_total_ms = min(dwell_total_ms, total_duration_ms * 0.5)  # never let dwell eat the whole budget

    if using_default_raster:
        n_transit_segments = n_anchors  # (n_anchors - 1) anchor-to-anchor hops + 1 entry-to-raster hop
        transit_ms = np.array(
            [
                visual_angle_deg(seg_vec[i, 0], seg_vec[i, 1], screen_w_px, screen_h_px, screen_w_cm, view_dist_cm)
                / anchor_transition_deg_s
                * 1000.0
                for i in range(n_transit_segments)
            ]
        )
        raster_budget_ms = max(total_duration_ms - dwell_total_ms - transit_ms.sum(), 1.0)
        raster_multipliers = _raster_segment_multipliers(len(RASTER_ROW_Y_FRACTIONS), RASTER_COLS, speed_multipliers)
        raster_raw_ms = seg_len_safe[n_transit_segments:] * raster_multipliers
        raster_ms = raster_raw_ms * (raster_budget_ms / raster_raw_ms.sum())
        travel_ms = np.concatenate([transit_ms, raster_ms])
    else:
        multipliers = np.array([speed_multipliers[i % len(speed_multipliers)] for i in range(n - 1)])
        raw_travel_ms = seg_len_safe * multipliers
        travel_budget_ms = max(total_duration_ms - dwell_total_ms, 1.0)
        travel_ms = raw_travel_ms * (travel_budget_ms / raw_travel_ms.sum())

    t_ms = [0.0]
    xs = [pts[0, 0]]
    ys = [pts[0, 1]]
    arrival_t = [0.0]  # arrival_t[j] = time the path first reaches waypoint j, pre-dwell
    anchor_arrivals = []
    t = 0.0
    for i in range(n - 1):
        t += travel_ms[i]
        arrival_t.append(t)
        t_ms.append(t)
        xs.append(pts[i + 1, 0])
        ys.append(pts[i + 1, 1])
        if 1 <= (i + 1) <= n_anchors - 1:
            anchor_arrivals.append(t)
            t += anchor_dwell_ms
            t_ms.append(t)
            xs.append(pts[i + 1, 0])
            ys.append(pts[i + 1, 1])

    dirs = seg_vec / seg_len_safe[:, None]
    turn_times = []
    for j in range(1, n - 1):
        cos_angle = np.clip(np.dot(dirs[j - 1], dirs[j]), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg >= turn_angle_threshold_deg:
            turn_times.append(arrival_t[j])

    return Keyframes(
        t_ms=np.array(t_ms, dtype=np.float64),
        x=np.array(xs, dtype=np.float64),
        y=np.array(ys, dtype=np.float64),
        turn_t_ms=np.array(turn_times, dtype=np.float64),
        anchor_arrival_t_ms=np.array(anchor_arrivals, dtype=np.float64),
    )


def turn_discard_mask(label_t_ms: np.ndarray, turn_t_ms: np.ndarray, discard_window_ms: float) -> np.ndarray:
    """True where label_t_ms falls within discard_window_ms AFTER some
    marked time (a sharp direction change, or an anchor arrival -- see
    Keyframes.turn_t_ms / anchor_arrival_t_ms). These are frames where the
    label is a target position the eye hasn't caught up with yet (reaction
    time plus the catch-up saccade) rather than a genuine fixation. Used
    with two different windows and time arrays -- a shorter one for raster
    direction changes, a longer settle window for anchor arrivals -- see
    quick_calibrate.py."""
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
    quick_calibrate.py / the --full design notes for how this was used to
    choose ANCHOR_TRANSITION_DEG_S."""
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
