"""Train the appearance-based gaze CNN (eye crops + face + landmark features).

    python scripts/train_cnn.py                     # honest held-out test, then train final model
    python scripts/train_cnn.py --skip-cv           # just train the final model (fast iteration)
    python scripts/train_cnn.py --no-aux            # pure-pixel CNN (ablation: what do the images add?)
    python scripts/train_cnn.py --epochs 40 --train-stride 2
    python scripts/train_cnn.py --cv person         # hold out whole PEOPLE: works on someone new?

Needs crop datasets from scripts/build_crops.py, and torch (pip install torch).
Prints the CNN's held-out pixel error NEXT TO ridge and hgb trained on the SAME
frames with the SAME folds, so the comparison is fair. Output: models/gaze_cnn.pt
(use it with: phase2_gaze_scroll.py --model models/gaze_cnn.pt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doomscroll.ml.cnn import (
    DEFAULT_CNN_PATH, CnnGazeModel, TrainConfig, baseline_oof, cross_validate, pick_device, pool_crops,
    px_errors, tail_split, train_net,
)
from doomscroll.ml.crop_dataset import load_all_crops
from doomscroll.ml.features import FEATURE_NAMES, YAW_INDEX
from doomscroll.ml.model import DEFAULT_MODEL_PATH
from doomscroll.ml.training import error_by_group, error_by_yaw

ROOT = Path(__file__).resolve().parents[1] / "data"


def summarize(err: np.ndarray, yaw: np.ndarray, who: np.ndarray) -> dict:
    return {"mean_px": float(err.mean()), "median_px": float(np.median(err)),
            "p90_px": float(np.percentile(err, 90)), "by_yaw": error_by_yaw(err, yaw),
            "by_person": error_by_group(err, who)}


def print_report(report: dict) -> None:
    print(f"\n{'model':8s} {'mean':>8s} {'median':>8s} {'p90':>8s}   (pixels, held-out, same frames & folds)")
    for name, r in report.items():
        print(f"{name:8s} {r['mean_px']:8.0f} {r['median_px']:8.0f} {r['p90_px']:8.0f}")
    names = list(report)
    print(f"\nerror by head yaw (mean px):  {'  '.join(f'{n:>7s}' for n in names)}")
    for i, (label, n, _) in enumerate(report[names[0]]["by_yaw"]):
        print(f"  {label:18s} n={n:6d}  " + "  ".join(f"{report[m]['by_yaw'][i][2]:7.0f}" for m in names))
    print(f"\nerror by participant (mean px): {'  '.join(f'{n:>7s}' for n in names)}")
    for i, (label, n, _) in enumerate(report[names[0]]["by_person"]):
        print(f"  {label:18s} n={n:6d}  " + "  ".join(f"{report[m]['by_person'][i][2]:7.0f}" for m in names))


def default_lag() -> float:
    """Reuse the lag train_model.py --sweep-lag picked, if that model exists."""
    try:
        import joblib

        return float(joblib.load(DEFAULT_MODEL_PATH)["metadata"].get("lag_ms", 0.0))
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=ROOT / "crops")
    ap.add_argument("--out", type=Path, default=DEFAULT_CNN_PATH)
    ap.add_argument("--lag-ms", type=float, default=None, help="default: the lag stored in models/gaze_model.joblib, else 0")
    ap.add_argument("--window-ms", type=float, default=600.0)
    ap.add_argument("--min-travel-px", type=float, default=40.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-stride", type=int, default=1, help="use every n-th training frame")
    ap.add_argument("--no-face", action="store_true", help="drop the face branch")
    ap.add_argument("--no-aux", action="store_true", help="drop the landmark-feature branch (pixels only)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--skip-cv", action="store_true", help="skip the held-out evaluation")
    ap.add_argument("--max-folds", type=int, default=5)
    ap.add_argument("--cv", choices=["session", "person"], default="session",
                    help="hold out whole sessions (default) or whole participants")
    args = ap.parse_args()

    lag = default_lag() if args.lag_ms is None else args.lag_ms
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, train_stride=args.train_stride,
                      use_face=not args.no_face, use_aux=not args.no_aux, augment=not args.no_augment,
                      device=args.device)

    sessions = load_all_crops(args.crops)
    if not sessions:
        sys.exit(f"No crop datasets in {args.crops}. Run scripts/build_crops.py first.")
    print(f"{len(sessions)} session(s): {', '.join(s.session_id for s in sessions)}")
    data = pool_crops(sessions, lag, args.window_ms, args.min_travel_px, group_by=args.cv)
    print(f"{len(data)} labeled frames, lag {lag:+.0f} ms, device {pick_device(cfg.device)}; cv: {data.cv_kind}")

    report = None
    if not args.skip_cv:
        print("\nHeld-out evaluation (one network per fold):")
        pred = cross_validate(data, cfg, args.max_folds)
        yaw = data.aux[:, YAW_INDEX]
        report = {"cnn": summarize(px_errors(pred, data), yaw, data.participant)}
        for name, p in baseline_oof(data, max_folds=args.max_folds).items():
            report[name] = summarize(px_errors(p, data), yaw, data.participant)
        print_report(report)

    print(f"\nTraining final CNN on all {len(data)} frames (last 10% of each session for early stopping) ...")
    itr, iva = tail_split(data)
    net, norm, info = train_net(data.take(itr), data.take(iva), cfg)
    model = CnnGazeModel(net, norm, metadata={
        "kind": "cnn", "lag_ms": lag, "n_frames": int(len(data)), "features": list(FEATURE_NAMES),
        "sessions": [s.session_id for s in sessions], "cv": data.cv_kind, "config": cfg.to_dict(),
        "best_epoch": info["best_epoch"], "final_val_px": info["best_val_px"],
        "heldout_report": report["cnn"] if report else None,
    })
    model.save(args.out)
    print(f"saved -> {args.out}  (best epoch {info['best_epoch']}, inner-val {info['best_val_px']:.0f}px)")
    if report:
        args.out.with_suffix(".report.json").write_text(json.dumps({"cv": data.cv_kind, **report}, indent=2))


if __name__ == "__main__":
    main()
