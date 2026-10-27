"""Evaluate the production inference pipeline end to end: raw HGB vs.
HGB + quick calibration vs. HGB + calibration + OOD gate.

Leave-person-out (the correct metric per the investigation -- leave-session-out
overestimates performance when one participant dominates session count).
"Quick calibration" is simulated the same way scripts/quick_calibrate.py
collects it live: a 3x3 grid of dots, a handful of frames per dot, nearest
to each dot's true position in the held-out person's own data.

    python scripts/evaluate_pipeline.py --out lab_analysis/pipeline_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from doomscroll.ml.dataset import load_all
from doomscroll.ml.labels import LabelSettings
from doomscroll.ml.model import fit_user_correction
from doomscroll.ml.ood import GazeConfidenceGate, fit_detector

ROOT = Path(__file__).resolve().parents[1] / "data"
DOT_STEPS = (0.1, 0.5, 0.9)
DOTS = [(fx, fy) for fy in DOT_STEPS for fx in DOT_STEPS]  # same 3x3 grid as quick_calibrate.py
FRAMES_PER_DOT = 8


def pool(labeled, ids):
    X = np.vstack([labeled[i].X for i in ids])
    y = np.vstack([labeled[i].y_frac for i in ids])
    wh = np.vstack([np.tile(labeled[i].screen_wh, (len(labeled[i].X), 1)) for i in ids]).astype(float)
    return X, y, wh


def errstat(pred_frac, true_frac, wh):
    e = np.linalg.norm((pred_frac - true_frac) * wh, axis=1)
    return dict(mean_px=float(e.mean()), median_px=float(np.median(e)), p90_px=float(np.percentile(e, 90)))


def pick_dot_calibration_set(y):
    used = set()
    dot_groups = []
    for fx, fy in DOTS:
        d = np.linalg.norm(y - np.array([fx, fy]), axis=1)
        chosen = []
        for idx in np.argsort(d):
            if idx in used:
                continue
            chosen.append(idx)
            used.add(idx)
            if len(chosen) >= FRAMES_PER_DOT:
                break
        dot_groups.append(((fx, fy), chosen))
    return dot_groups, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    ap.add_argument("--participants", nargs="+", default=["jingyi", "qiyun", "ysabelle"])
    ap.add_argument("--lag-ms", type=float, default=150.0)
    ap.add_argument("--ood-method", default="isolation_forest")
    ap.add_argument("--threshold-percentile", type=float, default=90.0,
                    help="gate threshold = this percentile of the detector's TRAINING scores "
                         "(matches train_model.py's default derivation; see lab_analysis/ood_sweep.py "
                         "for why the ACTUAL rejection rate on a novel person overshoots this nominal target)")
    ap.add_argument("--out", type=Path, default=Path("lab_analysis/pipeline_eval.csv"))
    args = ap.parse_args()

    settings = LabelSettings(lag_ms=args.lag_ms, mode="either")
    all_sessions = load_all(args.datasets)
    subset = [s for s in all_sessions if s.participant in args.participants]
    labeled = {s.session_id: s.labeled(settings) for s in subset}  # LabeledFrames already carries .screen_wh
    participant_of = {s.session_id: s.participant for s in subset}

    rows = []
    for held_out in args.participants:
        train_ids = [sid for sid, p in participant_of.items() if p != held_out]
        test_ids = [sid for sid, p in participant_of.items() if p == held_out]
        if not train_ids or not test_ids:
            continue

        Xtr, ytr, _ = pool(labeled, train_ids)
        Xte, yte, whte = pool(labeled, test_ids)

        hgb = MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=250, random_state=0))
        hgb.fit(Xtr, ytr)
        raw_pred = hgb.predict(Xte)

        detector = fit_detector(args.ood_method, Xtr)
        threshold = float(np.percentile(detector.score(Xtr), args.threshold_percentile))
        ood_scores = detector.score(Xte)

        dot_groups, calib_idx_set = pick_dot_calibration_set(yte)
        calib_idx = np.array(sorted(calib_idx_set))
        eval_mask = np.ones(len(Xte), dtype=bool)
        eval_mask[calib_idx] = False

        # --- 1. raw HGB ---
        r1 = errstat(raw_pred[eval_mask], yte[eval_mask], whte[eval_mask])
        r1["rejected_frame_rate"] = 0.0
        rows.append(dict(held_out=held_out, pipeline="1_raw_hgb", **r1))

        # --- 2. HGB + quick calibration ---
        predicted_dots = [np.median(raw_pred[idxs], axis=0) for _, idxs in dot_groups]
        truth_dots = [list(xy) for xy, _ in dot_groups]
        correction, _, _ = fit_user_correction(np.array(predicted_dots), np.array(truth_dots), tuple(whte[0]))
        corrected_pred = np.array([correction.apply(p) for p in raw_pred[eval_mask]])
        r2 = errstat(corrected_pred, yte[eval_mask], whte[eval_mask])
        r2["rejected_frame_rate"] = 0.0
        rows.append(dict(held_out=held_out, pipeline="2_hgb_plus_calibration", **r2))

        # --- 3. HGB + calibration + OOD gate ---
        # t_ms_fake: a monotonically increasing synthetic clock, since these held-out
        # frames aren't literally 33ms apart once filtered -- the gate's hold-duration
        # logic only needs monotonicity for this offline replay; live, it uses real
        # frame timestamps (see phase2_gaze_scroll.py).
        gate = GazeConfidenceGate(threshold=threshold, max_hold_ms=1500.0)
        gated_preds, gated_truth, gated_wh = [], [], []
        n_trust = n_hold = n_suppress = 0
        t_ms_fake = 0.0
        eval_order = np.where(eval_mask)[0]
        corrected_eval = np.array([correction.apply(raw_pred[i]) for i in eval_order])
        for k, i in enumerate(eval_order):
            t_ms_fake += 33.0
            point_frac = corrected_eval[k]
            point_px = point_frac * whte[i]
            decision, gated_point_px = gate.decide(ood_scores[i], point_px, t_ms_fake)
            if decision == "trust":
                n_trust += 1
            elif decision == "hold":
                n_hold += 1
            else:
                n_suppress += 1
                continue  # no output this frame -> not counted in accuracy, counted in reject rate
            gated_preds.append(gated_point_px / whte[i])
            gated_truth.append(yte[i])
            gated_wh.append(whte[i])

        if gated_preds:
            r3 = errstat(np.array(gated_preds), np.array(gated_truth), np.array(gated_wh))
        else:
            r3 = dict(mean_px=float("nan"), median_px=float("nan"), p90_px=float("nan"))
        total = len(eval_order)
        r3["rejected_frame_rate"] = n_suppress / total
        r3["held_frame_rate"] = n_hold / total
        rows.append(dict(held_out=held_out, pipeline="3_hgb_plus_calibration_plus_gate", **r3))

        print(f"[{held_out}] trust={n_trust} hold={n_hold} suppress={n_suppress} (of {total} eval frames)", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    print(f"\n{'held_out':10s} {'pipeline':32s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'reject%':>8s}")
    for r in rows:
        print(f"{r['held_out']:10s} {r['pipeline']:32s} {r['mean_px']:7.0f} {r['median_px']:7.0f} "
              f"{r['p90_px']:7.0f} {r['rejected_frame_rate']*100:7.1f}%")


if __name__ == "__main__":
    main()
