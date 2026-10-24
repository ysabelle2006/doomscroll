"""Calibration accuracy benchmark: answers "did my change to the gaze
calibration pipeline make it more or less accurate?"

By default this re-fits GazeCalibrator against your own most recently
recorded real calibration session (scripts/calibrate.py auto-saves one to
CALIBRATION_DIR/last_dataset.npz every time it runs) and reports mean
validation error in pixels. Re-fitting against already-captured data is
fast and fully reproducible -- no webcam, no repeating a live session --
so you can edit calibration.py / gaze_vector.py / head_pose.py and
immediately see whether accuracy on that same real session got better or
worse.

Typical workflow:
    1. Run scripts/calibrate.py once (as you normally would) to get a real
       recorded dataset.
    2. python scripts/bench_calibration.py --save-baseline
    3. Make your change to the calibration code.
    4. python scripts/bench_calibration.py
       -> prints IMPROVED/REGRESSED/UNCHANGED vs. the saved baseline.

If you don't have a real recorded session yet (e.g. on a machine with no
webcam, or in CI), pass --synthetic to benchmark against a fixed synthetic
dataset with a known ground truth instead (see tests/synthetic_calibration.py)
-- less representative of your actual face/webcam, but still catches a
regression in the regression/fitting logic itself.

--synthetic defaults to the current still-point grid protocol; pass
--protocol pursuit to benchmark the same calibration.py model-fitting code
against a synthetic continuous cursor-pursuit dataset instead (see
generate_pursuit_session() in tests/synthetic_calibration.py). Note this
only answers "did my change to the MODEL help or hurt, given pursuit-shaped
data" -- it does NOT compare the two protocols against each other; for that,
use scripts/compare_calibration_protocols.py instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from doomscroll.gaze.calibration import CALIBRATION_DIR, GazeCalibrator  # noqa: E402

DEFAULT_DATASET_PATH = CALIBRATION_DIR / "last_dataset.npz"
_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _default_baseline_path(protocol: str) -> Path:
    # Separate baseline file per synthetic protocol -- comparing a "still"
    # baseline against a "pursuit" run (or vice versa) would report a huge,
    # meaningless "regression" that's really just a different dataset.
    suffix = "" if protocol == "still" else f"_{protocol}"
    return _FIXTURES_DIR / f"calibration_baseline{suffix}.json"


def _load_real_dataset(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"train_features", "train_targets", "train_groups", "val_features", "val_targets", "screen_w", "screen_h"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"{path} is missing expected arrays: {sorted(missing)}")
    return {k: data[k] for k in required}


def _load_synthetic_dataset(seed: int, protocol: str = "still") -> dict:
    from tests.synthetic_calibration import generate_pursuit_session, generate_synthetic_session

    sess = generate_pursuit_session(seed=seed) if protocol == "pursuit" else generate_synthetic_session(seed=seed)
    return {
        "train_features": sess.train_features,
        "train_targets": sess.train_targets,
        "train_groups": sess.train_groups,
        "val_features": sess.val_features,
        "val_targets": sess.val_targets,
        "screen_w": sess.screen_w,
        "screen_h": sess.screen_h,
    }


def run_benchmark(dataset: dict) -> dict:
    calibrator = GazeCalibrator()
    calibrator.fit(dataset["train_features"], dataset["train_targets"], groups=dataset["train_groups"])

    mean_error_px = calibrator.validate(dataset["val_features"], dataset["val_targets"])
    screen_w, screen_h = int(dataset["screen_w"]), int(dataset["screen_h"])
    diag = (screen_w**2 + screen_h**2) ** 0.5

    # Same gain/compression diagnostic calibrate.py itself prints -- see its
    # comment for why this matters beyond raw validation error: a heavily
    # regularized fit can minimize validation error while still shrinking
    # the model's dynamic range (predictions clustering closer to the
    # center than real targets do).
    train_preds = np.array([calibrator.predict(f) for f in dataset["train_features"]])
    target_range = dataset["train_targets"].max(axis=0) - dataset["train_targets"].min(axis=0)
    pred_range = train_preds.max(axis=0) - train_preds.min(axis=0)
    gain = (pred_range / np.where(target_range == 0, 1, target_range)).tolist()

    return {
        "mean_error_px": float(mean_error_px),
        "pct_of_diagonal": float(100 * mean_error_px / diag),
        "gain_x": float(gain[0]),
        "gain_y": float(gain[1]),
        "selected_alpha_x": float(calibrator.selected_alpha[0]),
        "selected_alpha_y": float(calibrator.selected_alpha[1]),
        "n_train_samples": int(dataset["train_features"].shape[0]),
        "n_val_samples": int(dataset["val_features"].shape[0]),
    }


def _print_metrics(metrics: dict, label: str) -> None:
    print(f"\n{label}")
    print(f"  mean validation error : {metrics['mean_error_px']:.1f}px  ({metrics['pct_of_diagonal']:.2f}% of screen diagonal)")
    print(f"  gain (x, y)           : {metrics['gain_x']:.3f}, {metrics['gain_y']:.3f}  (1.0 is ideal; well below means compressed)")
    print(f"  selected alpha (x, y) : {metrics['selected_alpha_x']:.4g}, {metrics['selected_alpha_y']:.4g}")
    print(f"  samples               : {metrics['n_train_samples']} train / {metrics['n_val_samples']} validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Recorded dataset .npz (default: your last real calibrate.py run)")
    parser.add_argument("--synthetic", action="store_true", help="Benchmark against the synthetic dataset instead of a real recorded session")
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument(
        "--protocol", choices=["still", "pursuit"], default="still",
        help="Synthetic capture protocol to generate (--synthetic only). 'pursuit' still only tests the MODEL -- see this script's docstring.",
    )
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline metrics JSON to compare against / save to (default depends on --protocol)")
    parser.add_argument("--save-baseline", action="store_true", help="Save this run's metrics as the new baseline instead of comparing")
    args = parser.parse_args()
    if args.baseline is None:
        args.baseline = _default_baseline_path(args.protocol)

    if args.synthetic:
        dataset = _load_synthetic_dataset(args.synthetic_seed, args.protocol)
        source_label = f"synthetic dataset (protocol={args.protocol}, seed={args.synthetic_seed})"
    else:
        if not args.dataset.exists():
            print(
                f"No recorded dataset found at {args.dataset}.\n"
                "Run `python scripts/calibrate.py` once first (it auto-saves one), "
                "or pass --synthetic to benchmark against a synthetic dataset instead."
            )
            raise SystemExit(1)
        dataset = _load_real_dataset(args.dataset)
        source_label = f"recorded session: {args.dataset}"

    metrics = run_benchmark(dataset)
    _print_metrics(metrics, f"Benchmark ({source_label}):")

    if args.save_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        payload = {**metrics, "source": source_label, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        args.baseline.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved as new baseline -> {args.baseline}")
        return

    if not args.baseline.exists():
        print(f"\nNo baseline saved yet at {args.baseline}. Re-run with --save-baseline to set one.")
        return

    baseline = json.loads(args.baseline.read_text())
    before = baseline["mean_error_px"]
    after = metrics["mean_error_px"]
    delta_px = after - before
    delta_pct = 100 * delta_px / before if before else float("inf")

    print(f"\nvs. baseline ({baseline.get('saved_at', 'unknown date')}, {baseline.get('source', 'unknown source')}):")
    print(f"  {before:.1f}px -> {after:.1f}px  ({delta_px:+.1f}px, {delta_pct:+.1f}%)")
    if abs(delta_pct) < 2.0:
        print("  -> UNCHANGED (within noise)")
    elif delta_px < 0:
        print("  -> IMPROVED")
    else:
        print("  -> REGRESSED")


if __name__ == "__main__":
    main()
