"""Honest leave-one-participant-out comparison: ridge vs hgb, each run through
the full production pipeline (raw -> +calibration -> +calibration+OOD gate),
on the 4-person dataset (jingyi, qiyun, ysabelle, anni[non-lab]).

This is what should decide the production model -- NOT leave-session-out
error, which overestimates performance whenever one participant dominates
session count (see earlier findings).

    python scripts/compare_ridge_hgb_pipeline.py --out lab_analysis/ridge_vs_hgb.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from doomscroll.ml.dataset import load_all
from doomscroll.ml.labels import LabelSettings
from doomscroll.ml.model import fit_user_correction
from doomscroll.ml.ood import GazeConfidenceGate, fit_detector
from doomscroll.ml.training import CANDIDATES

ROOT = Path(__file__).resolve().parents[1] / "data"
DOT_STEPS = (0.1, 0.5, 0.9)
DOTS = [(fx, fy) for fy in DOT_STEPS for fx in DOT_STEPS]
FRAMES_PER_DOT = 8
CATASTROPHIC_DIAG_PCT = 50.0  # error beyond this % of screen diagonal counts as a catastrophic failure


def pool(labeled, ids):
    X = np.vstack([labeled[i].X for i in ids])
    y = np.vstack([labeled[i].y_frac for i in ids])
    wh = np.vstack([np.tile(labeled[i].screen_wh, (len(labeled[i].X), 1)) for i in ids]).astype(float)
    return X, y, wh


def errstat(pred_frac, true_frac, wh):
    e_px = np.linalg.norm((pred_frac - true_frac) * wh, axis=1)
    diag = np.linalg.norm(wh, axis=1)
    e_pct = e_px / diag * 100
    return dict(
        mean_px=float(e_px.mean()), median_px=float(np.median(e_px)), p90_px=float(np.percentile(e_px, 90)),
        mean_pct=float(e_pct.mean()), median_pct=float(np.median(e_pct)), p90_pct=float(np.percentile(e_pct, 90)),
        catastrophic_rate=float((e_pct > CATASTROPHIC_DIAG_PCT).mean()),
    )


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
    ap.add_argument("--participants", nargs="+", default=["jingyi", "qiyun", "ysabelle", "anni"])
    ap.add_argument("--lag-ms", type=float, default=150.0)
    ap.add_argument("--ood-method", default="isolation_forest")
    ap.add_argument("--gate-threshold-pct", type=float, default=99.0)
    ap.add_argument("--out", type=Path, default=Path("lab_analysis/ridge_vs_hgb.csv"))
    args = ap.parse_args()

    settings = LabelSettings(lag_ms=args.lag_ms, mode="either")
    all_sessions = load_all(args.datasets)
    subset = [s for s in all_sessions if s.participant in args.participants]
    labeled = {s.session_id: s.labeled(settings) for s in subset}
    participant_of = {s.session_id: s.participant for s in subset}
    present = sorted(set(participant_of.values()))
    print(f"participants present: {present}", file=sys.stderr)

    rows = []
    for model_name in ["ridge", "hgb"]:
        for held_out in present:
            train_ids = [sid for sid, p in participant_of.items() if p != held_out]
            test_ids = [sid for sid, p in participant_of.items() if p == held_out]
            Xtr, ytr, _ = pool(labeled, train_ids)
            Xte, yte, whte = pool(labeled, test_ids)

            model = CANDIDATES[model_name]()
            model.fit(Xtr, ytr)
            raw_pred = model.predict(Xte)

            detector = fit_detector(args.ood_method, Xtr)
            threshold = float(np.percentile(detector.score(Xtr), args.gate_threshold_pct))
            ood_scores = detector.score(Xte)

            dot_groups, calib_idx_set = pick_dot_calibration_set(yte)
            calib_idx = np.array(sorted(calib_idx_set))
            eval_mask = np.ones(len(Xte), dtype=bool)
            eval_mask[calib_idx] = False

            # 1. raw
            r1 = errstat(raw_pred[eval_mask], yte[eval_mask], whte[eval_mask])
            rows.append(dict(model=model_name, held_out=held_out, stage="1_raw", n_calib=0, rejected_frame_rate=0.0, **r1))

            # 2. + calibration
            predicted_dots = [np.median(raw_pred[idxs], axis=0) for _, idxs in dot_groups]
            truth_dots = [list(xy) for xy, _ in dot_groups]
            correction, before_px, after_loo_px = fit_user_correction(np.array(predicted_dots), np.array(truth_dots), tuple(whte[0]))
            corrected_eval = np.array([correction.apply(p) for p in raw_pred[eval_mask]])
            r2 = errstat(corrected_eval, yte[eval_mask], whte[eval_mask])
            rows.append(dict(model=model_name, held_out=held_out, stage="2_plus_calibration",
                              n_calib=len(calib_idx), rejected_frame_rate=0.0, **r2))

            # 3. + calibration + OOD gate
            gate = GazeConfidenceGate(threshold=threshold, max_hold_ms=1500.0)
            eval_order = np.where(eval_mask)[0]
            corrected_all_eval = np.array([correction.apply(raw_pred[i]) for i in eval_order])
            gated_preds, gated_truth, gated_wh = [], [], []
            n_trust = n_hold = n_suppress = 0
            t_fake = 0.0
            for k, i in enumerate(eval_order):
                t_fake += 33.0
                point_px = corrected_all_eval[k] * whte[i]
                decision, gated_px = gate.decide(ood_scores[i], point_px, t_fake)
                if decision == "trust":
                    n_trust += 1
                elif decision == "hold":
                    n_hold += 1
                else:
                    n_suppress += 1
                    continue
                gated_preds.append(gated_px / whte[i])
                gated_truth.append(yte[i])
                gated_wh.append(whte[i])
            if gated_preds:
                r3 = errstat(np.array(gated_preds), np.array(gated_truth), np.array(gated_wh))
            else:
                r3 = dict(mean_px=float("nan"), median_px=float("nan"), p90_px=float("nan"),
                          mean_pct=float("nan"), median_pct=float("nan"), p90_pct=float("nan"), catastrophic_rate=float("nan"))
            r3["rejected_frame_rate"] = n_suppress / len(eval_order)
            r3["n_calib"] = len(calib_idx)
            rows.append(dict(model=model_name, held_out=held_out, stage="3_plus_gate", **r3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}\n")

    print(f"{'model':6s} {'held_out':10s} {'stage':20s} {'mean_px':>8s} {'med_px':>8s} {'p90_px':>8s} {'catastroph%':>12s} {'reject%':>8s}")
    for r in rows:
        print(f"{r['model']:6s} {r['held_out']:10s} {r['stage']:20s} {r['mean_px']:8.0f} {r['median_px']:8.0f} "
              f"{r['p90_px']:8.0f} {r['catastrophic_rate']*100:11.1f}% {r['rejected_frame_rate']*100:7.1f}%")


if __name__ == "__main__":
    main()
