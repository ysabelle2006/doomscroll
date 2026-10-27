"""Train the gaze model from built datasets and report honest held-out error.

    python scripts/train_model.py                    # compare candidates, save the best
    python scripts/train_model.py --sweep-lag        # also find the best cursor-lag
    python scripts/train_model.py --model mlp        # force a model

Prints, for every candidate, mean/median/p90 pixel error on HELD-OUT data
(whole sessions or time blocks, never random frames) and the error per
head-yaw range -- the number that shows whether head angle still hurts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.ml.dataset import load_all
from doomscroll.ml.features import FEATURE_NAMES
from doomscroll.ml.labels import LABEL_MODES, LabelSettings, format_coverage, screen_coverage
from doomscroll.ml.model import DEFAULT_MODEL_PATH, TrainedGazeModel
from doomscroll.ml.ood import fit_detector
from doomscroll.ml.training import CANDIDATES, evaluate_candidates, fit_final, pool

ROOT = Path(__file__).resolve().parents[1] / "data"
LAG_SWEEP_MS = (-100, 0, 100, 200, 300)


def print_report(report: dict) -> None:
    # % of screen diagonal is the PRIMARY figure -- resolution-independent, so it's
    # comparable across participants/screens (raw px is not: e.g. a 3200x2000 vs a
    # 1512x982 recording give very different px errors for the same real accuracy).
    # Pixels are kept alongside for backwards comparison, clearly secondary.
    print("held-out error (held out by whole session/participant):\n")
    for name, r in report.items():
        print(f"{name}")
        print(f"  Mean error:   {r['mean_pct']:5.1f}% diagonal   ({r['mean_px']:.0f} px)")
        print(f"  Median error: {r['median_pct']:5.1f}% diagonal   ({r['median_px']:.0f} px)")
        print(f"  P90 error:    {r['p90_pct']:5.1f}% diagonal   ({r['p90_px']:.0f} px)")
        print()
    for name, r in report.items():
        print(f"{name}: error by head yaw  (% diagonal, primary -- px, secondary)")
        for row in r["by_yaw"]:
            print(f"  {row['label']:18s} n={row['n']:6d}  mean={row['mean_pct']:5.1f}%  ({row['mean_px']:.0f}px)")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--model", choices=list(CANDIDATES), help="skip comparison, train this one")
    ap.add_argument("--lag-ms", type=float, default=0.0, help="label = cursor at (t + lag); see ml/labels.py")
    ap.add_argument("--sweep-lag", action="store_true", help="try several lags with the ridge model and pick the best")
    ap.add_argument("--window-ms", type=float, default=600.0)
    ap.add_argument("--min-travel-px", type=float, default=40.0)
    ap.add_argument("--max-speed", type=float, default=3000.0,
                    help="drop frames where the cursor moves faster than this px/s (flicks; eyes can't follow)")
    ap.add_argument("--label-mode", choices=LABEL_MODES, default="moving",
                    help="moving = cursor recently moving; clicks = only near mouse-button presses "
                         "(strongest 'eyes are here' signal for games); either = both")
    ap.add_argument("--ood-method", choices=["isolation_forest", "mahalanobis", "lof", "none"],
                    default="isolation_forest",
                    help="confidence/OOD detector fit on the final training features and saved with "
                         "the model, for GazeConfidenceGate at live inference (see ml/ood.py). "
                         "isolation_forest is the evaluated default -- see lab_analysis/ood_sweep.*; "
                         "'none' skips it (old-style model file, no gate available).")
    args = ap.parse_args()

    sessions = load_all(args.datasets)
    print(f"{len(sessions)} session(s): {', '.join(s.session_id for s in sessions)}")

    def settings(lag):
        return LabelSettings(lag, args.window_ms, args.min_travel_px, args.max_speed, args.label_mode)

    def labeled(lag):
        return pool([s.labeled(settings(lag)) for s in sessions])

    lag = args.lag_ms
    if args.sweep_lag:
        scores = {}
        scores_pct = {}
        for cand_lag in LAG_SWEEP_MS:
            r = evaluate_candidates(labeled(cand_lag), ["ridge"])["ridge"]
            scores[cand_lag] = r["mean_px"]  # selection stays on px -- unchanged behaviour
            scores_pct[cand_lag] = r["mean_pct"]
            print(f"  lag {cand_lag:+5d} ms -> ridge mean error {scores_pct[cand_lag]:5.1f}% diagonal  ({scores[cand_lag]:.0f}px)")
        lag = min(scores, key=scores.get)
        print(f"best lag: {lag} ms")

    data = labeled(lag)
    print(f"{len(data.X)} labeled frames; cross-validation: {data.cv_kind}")
    print("Where the labels are on screen (rows = top..bottom, cols = left..right; "
          "empty cells = regions the model never saw):")
    print(format_coverage(screen_coverage(data.y)) + "\n")
    report = evaluate_candidates(data, [args.model] if args.model else None)
    print_report(report)

    best = args.model or min(report, key=lambda n: report[n]["mean_px"])
    print(f"\nTraining final '{best}' on all data -> {args.out}")

    ood_detector = None
    ood_threshold_default = None
    if args.ood_method != "none":
        print(f"Fitting OOD detector ({args.ood_method}) on final training features...")
        ood_detector = fit_detector(args.ood_method, data.X)
        # Default gate threshold: 99th percentile of training scores, NOT 90th.
        # Measured directly (scripts/evaluate_pipeline.py): a 90th-percentile
        # threshold looks like "reject ~10%" on paper but actually suppressed
        # scrolling 64% of the time for a genuinely novel person (qiyun/
        # ysabelle leave-person-out) -- unusable. 99th percentile brings that to
        # a sane ~14-15% while keeping ysabelle's real p90 improvement intact;
        # qiyun's error isn't well explained by this detector at any threshold
        # tested (see lid_ratio_y / visual-inspection findings), so don't expect
        # the gate alone to fix every person -- re-run evaluate_pipeline.py
        # after any retrain to confirm this still holds.
        ood_threshold_default = float(np.percentile(ood_detector.score(data.X), 99))
        print(f"  default gate threshold (99th pct of training scores): {ood_threshold_default:.3f}")

    model = TrainedGazeModel(
        fit_final(best, data),
        metadata={"kind": best, "lag_ms": lag, "n_frames": int(len(data.X)), "features": list(FEATURE_NAMES),
                  "sessions": [s.session_id for s in sessions], "label_settings": vars(settings(lag)), "cv": data.cv_kind,
                  "heldout_report": report[best], "ood_method": args.ood_method,
                  "ood_threshold_default": ood_threshold_default},
        ood_detector=ood_detector,
    )
    model.save(args.out)


if __name__ == "__main__":
    main()
