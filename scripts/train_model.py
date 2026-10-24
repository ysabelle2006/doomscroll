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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.ml.dataset import load_all
from doomscroll.ml.features import FEATURE_NAMES
from doomscroll.ml.labels import LABEL_MODES, LabelSettings, format_coverage, screen_coverage
from doomscroll.ml.model import DEFAULT_MODEL_PATH, TrainedGazeModel
from doomscroll.ml.training import CANDIDATES, evaluate_candidates, fit_final, pool

ROOT = Path(__file__).resolve().parents[1] / "data"
LAG_SWEEP_MS = (-100, 0, 100, 200, 300)


def print_report(report: dict) -> None:
    print(f"{'model':8s} {'mean':>8s} {'median':>8s} {'p90':>8s}   (pixels, held-out)")
    for name, r in report.items():
        print(f"{name:8s} {r['mean_px']:8.0f} {r['median_px']:8.0f} {r['p90_px']:8.0f}")
    for name, r in report.items():
        print(f"\n{name}: error by head yaw")
        for label, n, err in r["by_yaw"]:
            print(f"  {label:18s} n={n:6d}  mean={err:6.0f}px")


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
        for cand_lag in LAG_SWEEP_MS:
            scores[cand_lag] = evaluate_candidates(labeled(cand_lag), ["ridge"])["ridge"]["mean_px"]
            print(f"  lag {cand_lag:+5d} ms -> ridge mean error {scores[cand_lag]:.0f}px")
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
    model = TrainedGazeModel(
        fit_final(best, data),
        metadata={"kind": best, "lag_ms": lag, "n_frames": int(len(data.X)), "features": list(FEATURE_NAMES),
                  "sessions": [s.session_id for s in sessions], "label_settings": vars(settings(lag)), "cv": data.cv_kind, "heldout_report": report[best]},
    )
    model.save(args.out)


if __name__ == "__main__":
    main()
