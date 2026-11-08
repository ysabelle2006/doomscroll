# Builds the moving-dot path (anchors + raster sweep) for trajectory
# calibration, and simulates/measures its pursuit speed.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Screen positions the dot dwells on at the path's start/corners/end.
ANCHOR_FRACTIONS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.08, 0.08),
    (0.92, 0.08),
    (0.92, 0.92),
    (0.08, 0.92),
    (0.5, 0.92),
    (0.5, 0.5),
)

# Row/column layout of the raster sweep that fills in the rest of the path.
RASTER_ROW_Y_FRACTIONS: tuple[float, ...] = (0.15, 0.35, 0.55, 0.75, 0.92)
RASTER_COLS = 10
RASTER_X_RANGE: tuple[float, float] = (0.05, 0.95)

ANCHOR_DWELL_MS = 800.0
ANCHOR_SETTLE_MS = 300.0

TURN_PAUSE_MS = 500.0

COMFORTABLE_PURSUIT_DEG_S = 30.0

TURN_ANGLE_THRESHOLD_DEG = 50.0

# Fallback screen size/viewing distance used to convert pixel motion to visual angle.
ASSUMED_SCREEN_W_PX = 1920
ASSUMED_SCREEN_H_PX = 1080
ASSUMED_SCREEN_W_CM = 34.5
ASSUMED_VIEW_DIST_CM = 60.0


# Converts a pixel displacement to visual angle in degrees.
def visual_angle_deg(
    dx_frac: float,
    dy_frac: float,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> float:
    cm_per_px = screen_w_cm / screen_w_px
    dx_cm = dx_frac * screen_w_px * cm_per_px
    dy_cm = dy_frac * screen_h_px * cm_per_px
    dist_cm = float(np.hypot(dx_cm, dy_cm))
    return float(np.degrees(np.arctan2(dist_cm, view_dist_cm)))


# A piecewise-linear path: timestamps plus x/y targets, with turn/pause markers.
@dataclass(frozen=True)
class Keyframes:

    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    turn_t_ms: np.ndarray
    pause_arrival_t_ms: np.ndarray = field(default_factory=lambda: np.array([]))

    # Interpolates the path's (x, y) target at the given time(s).
    def target_at(self, query_t_ms):
        q = np.atleast_1d(np.asarray(query_t_ms, dtype=np.float64))
        out = np.column_stack([np.interp(q, self.t_ms, self.x), np.interp(q, self.t_ms, self.y)])
        return out[0] if np.ndim(query_t_ms) == 0 else out

    @property
    def duration_ms(self) -> float:
        return float(self.t_ms[-1])


# Per-segment angular speed stats for a built path.
@dataclass(frozen=True)
class VelocityStats:

    segment_dt_ms: np.ndarray
    segment_deg_s: np.ndarray
    mean_deg_s: float
    p90_deg_s: float
    max_deg_s: float

    # Returns which segments exceed the given speed threshold.
    def segments_over(self, threshold_deg_s: float) -> np.ndarray:
        moving = self.segment_dt_ms > 1e-6
        over = np.zeros_like(moving)
        over[moving] = self.segment_deg_s[moving] > threshold_deg_s
        return over


# Computes angular speed (deg/s) for each segment of a path.
def measure_velocity(
    keyframes: Keyframes,
    screen_w_px: int = ASSUMED_SCREEN_W_PX,
    screen_h_px: int = ASSUMED_SCREEN_H_PX,
    screen_w_cm: float = ASSUMED_SCREEN_W_CM,
    view_dist_cm: float = ASSUMED_VIEW_DIST_CM,
) -> VelocityStats:
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


# Builds the full waypoint list: anchors, then a raster sweep covering the screen.
def build_waypoints() -> list[tuple[float, float]]:
    waypoints = list(ANCHOR_FRACTIONS)
    for i, y in enumerate(RASTER_ROW_Y_FRACTIONS):
        xs = np.linspace(RASTER_X_RANGE[0], RASTER_X_RANGE[1], RASTER_COLS)
        if i % 2 == 1:
            xs = xs[::-1]
        waypoints.extend((float(x), float(y)) for x in xs)
    return waypoints


# Converts waypoints into timed keyframes, budgeting dwell/turn-pause time within total_duration_ms.
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
    waypoints = waypoints if waypoints is not None else build_waypoints()
    n_anchors = n_anchors if n_anchors is not None else len(ANCHOR_FRACTIONS)
    pts = np.array(waypoints, dtype=np.float64)
    n = len(pts)
    if n < 2:
        raise ValueError("Need at least 2 waypoints to build a path")

    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len_safe = np.where(seg_len < 1e-9, 1e-9, seg_len)

    seg_deg = np.array(
        [
            visual_angle_deg(seg_vec[i, 0], seg_vec[i, 1], screen_w_px, screen_h_px, screen_w_cm, view_dist_cm)
            for i in range(n - 1)
        ]
    )
    seg_deg_safe = np.where(seg_deg < 1e-9, 1e-9, seg_deg)

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
    dwell_total_ms = min(dwell_total_ms, total_duration_ms * 0.5)
    turn_pause_total_ms = turn_pause_ms * n_turn_pauses
    turn_pause_total_ms = min(turn_pause_total_ms, total_duration_ms * 0.25)

    travel_budget_ms = max(total_duration_ms - dwell_total_ms - turn_pause_total_ms, 1.0)
    travel_ms = seg_deg_safe * (travel_budget_ms / seg_deg_safe.sum())

    t_ms = [0.0]
    xs = [pts[0, 0]]
    ys = [pts[0, 1]]
    arrival_t = [0.0]
    pause_arrivals = []
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


# Marks samples captured shortly after a sharp turn, to be discarded.
def turn_discard_mask(label_t_ms: np.ndarray, turn_t_ms: np.ndarray, discard_window_ms: float) -> np.ndarray:
    if len(turn_t_ms) == 0:
        return np.zeros(len(label_t_ms), dtype=bool)
    discard = np.zeros(len(label_t_ms), dtype=bool)
    for t_turn in turn_t_ms:
        discard |= (label_t_ms >= t_turn) & (label_t_ms < t_turn + discard_window_ms)
    return discard


# Simulates how an eye would track the path with reaction lag and a max pursuit speed.
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
    dt_ms = 1000.0 / 60.0
    t = 0.0
    gaze = np.array(keyframes.target_at(0.0), dtype=np.float64)
    sim_t, sim_gaze = [0.0], [gaze.copy()]
    end_t = max(float(query_t_ms.max()), keyframes.duration_ms)
    cm_per_px = screen_w_cm / screen_w_px
    while t < end_t:
        t = min(t + dt_ms, end_t)
        target = keyframes.target_at(t - min(reaction_ms, t))
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
