"""Per-user calibration ON TOP of (--quick) or INSTEAD of (--full) the
trained model.

--quick (default, unchanged): ~15 seconds, 9 dots. The trained model
(scripts/train_model.py) already knows the general eye+head -> screen
mapping from many recorded frames. This only shows 9 dots to measure how
THIS person/camera/seat differs, and fits a tiny scale+shift correction
(see ml/model.py: AffineCorrection). Compare with calibrate.py, which has to
learn the entire mapping from scratch with ~80 dots.

--full: ~20-30 seconds, continuous anchor+raster path (see gaze/trajectory.py),
collecting every usable frame instead of a handful of per-dot samples, and
fitting a RidgeCV model directly on them (see gaze/full_calibration.py) --
the validated fix for cases where --quick's small affine correction isn't
enough, per the written-up findings: more dots alone didn't help, a
piecewise correction hurt, and an uncontrolled moving-target calibration
did badly because its screen coverage was poor, but a *programmed* path
with deliberately even coverage, run long enough, matches or beats static
calibration. An early version of this path chased even coverage by racing
through anchor-to-anchor transitions -- up to ~68 deg/s of visual angle,
well past where human smooth pursuit holds up -- which was both unpleasant
to follow and (measured via gaze/trajectory.simulate_pursuit_gaze) a real
source of label noise, since the fixed-lag label model assumes continuous
pursuit. Anchor transitions are now paced to a comfortable target angular
velocity instead (gaze/trajectory.ANCHOR_TRANSITION_DEG_S), which cut
simulated label error by roughly 7x; a dedicated post-arrival settle window
was also tried and measured to add no further benefit once pacing alone
fixed the underlying problem (see trajectory.py's ANCHOR_SETTLE_MS comment).

Unlike calibrate.py you may move your head naturally -- the model handles it.
Run: python scripts/quick_calibrate.py            (ESC aborts; --quick is the default)
     python scripts/quick_calibrate.py --full
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.capture.webcam import WebcamSource
from doomscroll.dpi import ensure_dpi_aware, get_screen_size
from doomscroll.gaze.full_calibration import DEFAULT_FULL_CALIBRATION_PATH, FullCalibrator
from doomscroll.gaze.trajectory import ANCHOR_SETTLE_MS, build_keyframes, measure_velocity, turn_discard_mask
from doomscroll.ml.features import FrameFeaturizer
from doomscroll.ml.labels import format_coverage, screen_coverage
from doomscroll.ml.model import DEFAULT_CORRECTION_PATH, DEFAULT_MODEL_PATH, TrainedGazeModel, fit_user_correction

WINDOW_NAME = "doomscroll quick calibration"
# Tried a denser, top-weighted grid (3x5=15 dots) and a piecewise top/bottom
# correction here -- both tested WORSE than this plain 3x3 grid on real
# held-out data (see lab_analysis/ validation notes), because a quick-
# calibration-sized sample (a handful of points per region) makes a richer
# correction noisier, not better. Reverted. If top-of-screen accuracy still
# isn't good enough, the validated fix is a bigger-budget calibration (RidgeCV
# from ~1000 frames, not this affine correction) -- see --full below, not a
# different dot grid.
STEPS = (0.1, 0.5, 0.9)
POINTS = [(fx, fy) for fy in STEPS for fx in STEPS]  # 3x3 grid, fractions of the screen
SETTLE_S = 0.7  # let the eyes arrive
CAPTURE_S = 0.8
MIN_FRAMES_PER_POINT = 5
RADIUS = 18

# --full defaults. Duration is the 20-30s target from the findings (coverage
# matters more than sample count, and 30-60s of continuous collection is
# where it was shown to pay off; 20-30s is the budget end of that range).
FULL_DURATION_S = 25.0
# Typical saccadic-latency ballpark: the eye reaches a moving target this
# long after it moved, so the frame captured at time t is best labeled with
# where the target was at (t - lag_ms), not where it is *now*. Configurable
# because the right value depends on the trajectory's own speed and on the
# person (see gaze/trajectory.py docstring).
FULL_LAG_MS = 150.0
# How long after a sharp raster direction change (a row reversal) to
# discard frames, to skip the pursuit-catch-up saccade rather than label it
# as a (momentarily wrong) fixation. Anchor arrivals get their own, longer
# settle window -- see --anchor-settle-ms / gaze/trajectory.ANCHOR_SETTLE_MS.
FULL_TURN_DISCARD_MS = 250.0
FULL_TIME_BLOCK_MS = 1500.0  # CV/grouping granularity for the continuous recording
MIN_FULL_SAMPLES = 200


def show_dot(w: int, h: int, px: tuple[int, int]) -> bool:
    """Draw one dot; returns True if ESC was pressed."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(canvas, px, RADIUS, (0, 220, 255), -1)
    cv2.circle(canvas, px, RADIUS, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, canvas)
    return (cv2.waitKey(1) & 0xFF) == 27


def run_quick_calibration(w: int, h: int, model_path: Path) -> None:
    model = TrainedGazeModel.load(model_path)  # deliberately WITHOUT any old correction
    featurizer = FrameFeaturizer()

    predicted, truth = [], []
    try:
        with WebcamSource(0) as cam:
            for fx, fy in POINTS:
                px = (int(fx * w), int(fy * h))
                fracs = []
                start = time.monotonic()
                while time.monotonic() - start < SETTLE_S + CAPTURE_S:
                    if show_dot(w, h, px):
                        print("Aborted; nothing saved.")
                        return
                    ok, frame = cam.read()
                    if not ok or time.monotonic() - start < SETTLE_S:
                        continue
                    feats, _reason = featurizer.process(frame, cam.frame_timestamp_ms())
                    if feats is not None:
                        fracs.append(model.predict_frac(feats))
                if len(fracs) >= MIN_FRAMES_PER_POINT:
                    predicted.append(np.median(fracs, axis=0))  # median: robust to a stray frame
                    truth.append([fx, fy])
                else:
                    print(f"  no usable frames for dot at ({fx}, {fy}); skipped")
    finally:
        featurizer.close()

    if len(predicted) < 6:
        sys.exit("Too few usable dots; nothing saved. Check lighting / that your face stays in view.")

    predicted_arr, truth_arr = np.array(predicted), np.array(truth)
    diag = (w**2 + h**2) ** 0.5

    # Per-dot diagnostic BEFORE any correction -- this is the thing last run's
    # summary number couldn't show. Sorted worst-first so a region-specific
    # problem (e.g. "everything near the top") jumps out immediately.
    raw_err_px = np.linalg.norm((predicted_arr - truth_arr) * [w, h], axis=1)
    print("\nPer-dot raw error (worst first):")
    for idx in np.argsort(-raw_err_px):
        fx, fy = truth_arr[idx]
        region = "TOP" if fy <= 0.25 else ("BOTTOM" if fy >= 0.75 else "MID")
        print(f"  dot ({fx:.2f},{fy:.2f}) [{region:6s}]  raw_pred=({predicted_arr[idx][0]:.2f},{predicted_arr[idx][1]:.2f})  "
              f"error={raw_err_px[idx]:5.0f}px ({raw_err_px[idx]/diag*100:4.1f}%)")
    top_mask = truth_arr[:, 1] <= 0.25
    if top_mask.any():
        print(f"\nTOP-region (y<=0.25) mean raw error: {raw_err_px[top_mask].mean():.0f}px "
              f"({raw_err_px[top_mask].mean()/diag*100:.1f}%)  vs. rest: "
              f"{raw_err_px[~top_mask].mean():.0f}px ({raw_err_px[~top_mask].mean()/diag*100:.1f}%)")

    correction, before, after = fit_user_correction(predicted_arr, truth_arr, (w, h))
    # same training-pipeline reporting convention (% of diagonal as the primary,
    # resolution-independent figure; px kept secondary) -- see ml/training.py.
    print(f"\nmean error: {before / diag * 100:.1f}% diagonal ({before:.0f}px) without correction, "
          f"{after / diag * 100:.1f}% diagonal ({after:.0f}px) with (leave-one-out)")
    if after < before:
        correction.save(DEFAULT_CORRECTION_PATH)
        print(f"Saved -> {DEFAULT_CORRECTION_PATH}")
    else:
        print("Correction doesn't help for you; not saving (the base model is used as-is). "
              "Check the per-dot breakdown above -- if error is concentrated in one region, "
              "a single global scale+shift can't fix that; see the written-up findings for why.")


def _collect_full_frames(cam: WebcamSource, featurizer: FrameFeaturizer, w: int, h: int, keyframes) -> tuple[np.ndarray, np.ndarray]:
    """Runs the continuous anchor+raster path for keyframes.duration_ms,
    collecting EVERY usable frame (not a median-per-dot aggregate -- see
    module docstring requirement). Returns (capture_t_ms, features), both
    (N,) / (N, FEATURE_DIM), in capture order. Raises KeyboardInterrupt on ESC."""
    capture_t_ms: list[float] = []
    features: list[np.ndarray] = []
    start = time.monotonic()
    duration_ms = keyframes.duration_ms
    while True:
        now_ms = (time.monotonic() - start) * 1000.0
        if now_ms >= duration_ms:
            break
        x_frac, y_frac = keyframes.target_at(now_ms)
        px = (int(np.clip(x_frac, 0.0, 1.0) * w), int(np.clip(y_frac, 0.0, 1.0) * h))
        if show_dot(w, h, px):
            raise KeyboardInterrupt
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        feats, _reason = featurizer.process(frame, cam.frame_timestamp_ms())
        if feats is not None:
            capture_t_ms.append(now_ms)
            features.append(feats)
    return np.array(capture_t_ms, dtype=np.float64), np.array(features, dtype=np.float64)


def run_full_calibration(
    w: int,
    h: int,
    duration_s: float,
    lag_ms: float,
    turn_discard_ms: float,
    anchor_settle_ms: float,
    save_path: Path,
) -> None:
    keyframes = build_keyframes(total_duration_ms=duration_s * 1000.0)

    # Trackability diagnostic: how fast the dot itself moves, in degrees/sec
    # of visual angle (see gaze/trajectory.py). A real run that measures
    # "too fast to follow" should show this exceeding ~30 deg/s somewhere --
    # if it doesn't, the complaint is about something other than target speed.
    vel = measure_velocity(keyframes)
    over = vel.segments_over(30.0).sum()
    print(f"[full] target speed: mean={vel.mean_deg_s:.1f} deg/s, p90={vel.p90_deg_s:.1f} deg/s, "
          f"max={vel.max_deg_s:.1f} deg/s ({over} segment(s) over the ~30 deg/s smooth-pursuit comfort ceiling)")

    featurizer = FrameFeaturizer()
    try:
        with WebcamSource(0) as cam:
            print(f"Full calibration starting (~{duration_s:.0f}s). ESC to abort.")
            try:
                capture_t_ms, features = _collect_full_frames(cam, featurizer, w, h, keyframes)
            except KeyboardInterrupt:
                print("Aborted; nothing saved.")
                return
    finally:
        featurizer.close()

    print(f"\n[full] collected {len(features)} raw frames "
          f"({len(features) / (duration_s if duration_s > 0 else 1):.1f} usable frames/sec)")
    if len(features) < MIN_FULL_SAMPLES:
        sys.exit(f"Too few usable frames ({len(features)} < {MIN_FULL_SAMPLES}); nothing saved. "
                  "Check lighting / that your face stays in view.")

    # 1) Label generation: the programmed target position at each frame's
    # capture time, shifted by the configurable lag correction (see
    # FULL_LAG_MS). Clipped to the path's own time range since a frame
    # captured right at the very end can't be shifted past it.
    label_t_ms = np.clip(capture_t_ms - lag_ms, 0.0, keyframes.duration_ms)
    targets_frac = keyframes.target_at(label_t_ms)

    # 2) Direction-change handling: discard frames whose label falls inside
    # the configurable post-turn window (raster row reversals), or inside
    # the longer post-arrival settle window (anchor jumps -- see
    # gaze/trajectory.ANCHOR_SETTLE_MS), instead of training on a target the
    # eyes haven't caught up with yet.
    turn_discard = turn_discard_mask(label_t_ms, keyframes.turn_t_ms, turn_discard_ms)
    settle_discard = turn_discard_mask(label_t_ms, keyframes.anchor_arrival_t_ms, anchor_settle_ms)
    kept = ~(turn_discard | settle_discard)
    print(f"[full] discarding {int(turn_discard.sum())} frames within {turn_discard_ms:.0f}ms of a raster "
          f"direction change ({len(keyframes.turn_t_ms)} turns), {int(settle_discard.sum())} within "
          f"{anchor_settle_ms:.0f}ms of an anchor arrival ({len(keyframes.anchor_arrival_t_ms)} anchors)")
    features, targets_frac, capture_t_ms = features[kept], targets_frac[kept], capture_t_ms[kept]

    if len(features) < MIN_FULL_SAMPLES:
        sys.exit(f"Too few frames left after direction-change discard ({len(features)} < {MIN_FULL_SAMPLES}); "
                  "nothing saved. Try a larger --turn-discard-ms budget or a longer --duration.")

    # Diagnostics.
    print(f"[full] sample count after filtering: {len(features)}")
    print("[full] region coverage (%, row=y top->bottom, col=x left->right):")
    print(format_coverage(screen_coverage(targets_frac)))

    groups = (capture_t_ms // FULL_TIME_BLOCK_MS).astype(np.int64)
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        from sklearn.model_selection import GroupKFold

        oof_pred = np.zeros_like(targets_frac)
        for train_idx, test_idx in GroupKFold(n_splits=min(5, n_groups)).split(features, targets_frac[:, 0], groups=groups):
            fold_model = FullCalibrator()
            fold_model.fit(features[train_idx], targets_frac[train_idx])
            oof_pred[test_idx] = np.array([fold_model.predict(f) for f in features[test_idx]])
        oof_err_frac = np.linalg.norm(oof_pred - targets_frac, axis=1) / np.sqrt(2.0)
        print(f"[full] calibration fit score (out-of-fold mean error, {n_groups} time-block folds): "
              f"{oof_err_frac.mean() * 100:.1f}% of screen diagonal")
        top_mask = targets_frac[:, 1] <= 0.25
        if top_mask.any():
            print(f"[full] estimated TOP-region (y<=0.25) error: {oof_err_frac[top_mask].mean() * 100:.1f}% "
                  f"of screen diagonal (n={int(top_mask.sum())})")
    else:
        print("[full] not enough time-block groups for an out-of-fold fit score; fitting on everything.")

    calibrator = FullCalibrator()
    calibrator.fit(features, targets_frac, groups=groups)
    print(f"[full] RidgeCV selected alpha (x, y): {calibrator.selected_alpha}")
    calibrator.save(save_path)
    print(f"Saved -> {save_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                    help="which trained model to calibrate against in --quick mode "
                    "(default: models/gaze_model.joblib)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                       help="~15s, 9-dot affine correction on top of the trained model (default).")
    mode.add_argument("--full", action="store_true",
                       help="~20-30s continuous anchor+raster calibration; fits a RidgeCV model "
                       "directly on the collected frames instead of the trained model + a small correction.")
    ap.add_argument("--duration", type=float, default=FULL_DURATION_S,
                     help=f"--full only: target duration in seconds (default {FULL_DURATION_S:.0f}).")
    ap.add_argument("--lag-ms", type=float, default=FULL_LAG_MS,
                     help=f"--full only: label lag correction in ms (default {FULL_LAG_MS:.0f}) -- "
                     "a frame at time t is labeled with the target position at (t - lag_ms).")
    ap.add_argument("--turn-discard-ms", type=float, default=FULL_TURN_DISCARD_MS,
                     help=f"--full only: discard window after a raster direction change (row reversal), "
                     f"in ms (default {FULL_TURN_DISCARD_MS:.0f}).")
    ap.add_argument("--anchor-settle-ms", type=float, default=ANCHOR_SETTLE_MS,
                     help=f"--full only: discard window after each anchor-jump arrival, in ms "
                     f"(default {ANCHOR_SETTLE_MS:.0f}) -- longer than --turn-discard-ms since these "
                     "are bigger jumps than a raster row reversal.")
    ap.add_argument("--full-save", type=Path, default=DEFAULT_FULL_CALIBRATION_PATH,
                     help="--full only: where to save the fitted RidgeCV calibration profile.")
    args = ap.parse_args()

    ensure_dpi_aware()
    w, h = get_screen_size()

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, 0, 0)  # same macOS fullscreen workaround as calibrate.py
    cv2.resizeWindow(WINDOW_NAME, w, h)
    for _ in range(5):
        cv2.imshow(WINDOW_NAME, np.zeros((h, w, 3), dtype=np.uint8))
        cv2.waitKey(30)

    try:
        if args.full:
            run_full_calibration(
                w, h, args.duration, args.lag_ms, args.turn_discard_ms, args.anchor_settle_ms, args.full_save
            )
        else:
            run_quick_calibration(w, h, args.model)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
