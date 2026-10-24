"""Head-to-head synthetic comparison: still-point fixation calibration vs.
continuous cursor-pursuit calibration -- answers "if I change calibrate.py
from showing a sequence of still points to having the eyes follow a moving
cursor, does that help or hurt final calibration accuracy?"

This is a DIFFERENT question from scripts/bench_calibration.py, which asks
"did my change to the calibration MODEL (calibration.py) help or hurt."
bench_calibration.py is sensitive to algorithm changes (regression type,
polynomial degree, feature scaling, ...) but blind to changes in the data
*capture* protocol itself -- its --synthetic mode always assumes the
current still-point grid design (or, with --protocol pursuit, a fixed
pursuit design) rather than comparing the two. This script instead holds
the model fixed and varies the capture protocol, using
tests/synthetic_calibration.py's generate_protocol_comparison_session(),
which fits GazeCalibrator once on still-point training data and once on
cursor-pursuit training data, then validates BOTH against the exact same
held-out validation set -- a fair, paired, fully reproducible comparison.

What's modeled, and the key assumption to scrutinize: continuous pursuit
means gaze can never track a moving target perfectly. There's a known
response latency (roughly 100-150ms) before the eye begins following, and
the eye typically moves a little slower than the target (pursuit "gain"
below 1). If a pursuit-based calibrate.py logs "wherever the cursor was
drawn" as the training label for each captured frame, that lag/undershoot
becomes label noise/bias no amount of model fitting can remove -- unlike a
still-point protocol with a brief settle/dwell time, where gaze has
already caught up to the target before a sample is taken. That tradeoff is
modeled explicitly via PURSUIT_LATENCY_S / PURSUIT_GAIN in
synthetic_calibration.py; use --latency / --gain below to see how sensitive
the conclusion is to those assumptions (set them to 0 / 1.0 to simulate
"perfect" pursuit tracking with no lag at all, as an upper bound).

A second, smaller factor beyond lag/gain: even with --latency 0 --gain 1.0
(perfect tracking), pursuit still comes out somewhat worse than the
still-point grid here, because sampling a moving target at a fixed frame
rate naturally spends more time near the path's turning points (where it
moves slowly) than its middle (where it moves fastest) -- unlike the
still-point grid, which was deliberately laid out as evenly-spaced points.
That is a real property of continuous-sweep sampling, not a modeling bug;
a denser/slower sweep or a deliberately non-sinusoidal path would reduce it.

What's NOT modeled: any advantage/disadvantage of one protocol being more
tedious, slower, or more error-prone for the user to actually sit through,
and any difference in landmark-detector/MediaPipe noise between the two
(both reuse the same per-frame measurement noise model). This is also not
a substitute for trying both protocols with a real live session -- it's a
cheap, repeatable way to sanity-check the idea and see how sensitive the
answer is before spending time rewriting calibrate.py's capture loop.

Usage:
    python scripts/compare_calibration_protocols.py
    python scripts/compare_calibration_protocols.py --seeds 20
    python scripts/compare_calibration_protocols.py --latency 0.08 --gain 0.95
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from doomscroll.gaze.calibration import GazeCalibrator  # noqa: E402
from tests import synthetic_calibration as sc  # noqa: E402


def _fit_and_validate(features: np.ndarray, targets: np.ndarray, groups: np.ndarray, val_features: np.ndarray, val_targets: np.ndarray) -> float:
    calibrator = GazeCalibrator()
    calibrator.fit(features, targets, groups=groups)
    return float(calibrator.validate(val_features, val_targets))


def run_comparison(seed: int) -> dict:
    session = sc.generate_protocol_comparison_session(seed=seed)
    still_error = _fit_and_validate(
        session.still_train_features, session.still_train_targets, session.still_train_groups,
        session.val_features, session.val_targets,
    )
    pursuit_error = _fit_and_validate(
        session.pursuit_train_features, session.pursuit_train_targets, session.pursuit_train_groups,
        session.val_features, session.val_targets,
    )
    return {"still_error_px": still_error, "pursuit_error_px": pursuit_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10, help="Number of independent synthetic sessions to average over (default: 10)")
    parser.add_argument("--latency", type=float, default=None, help=f"Override pursuit latency in seconds (default: {sc.PURSUIT_LATENCY_S})")
    parser.add_argument("--gain", type=float, default=None, help=f"Override pursuit gain, <=1.0 (default: {sc.PURSUIT_GAIN})")
    args = parser.parse_args()

    if args.latency is not None:
        sc.PURSUIT_LATENCY_S = args.latency
    if args.gain is not None:
        sc.PURSUIT_GAIN = args.gain

    still_errors, pursuit_errors = [], []
    for seed in range(args.seeds):
        result = run_comparison(seed)
        still_errors.append(result["still_error_px"])
        pursuit_errors.append(result["pursuit_error_px"])

    still_mean, pursuit_mean = float(np.mean(still_errors)), float(np.mean(pursuit_errors))
    still_std, pursuit_std = float(np.std(still_errors)), float(np.std(pursuit_errors))

    print(f"Still-point protocol    : {still_mean:.1f}px mean error  (+/- {still_std:.1f}px across {args.seeds} seeds)")
    print(f"Cursor-pursuit protocol  : {pursuit_mean:.1f}px mean error  (+/- {pursuit_std:.1f}px across {args.seeds} seeds)")
    print(f"  (assumed pursuit latency={sc.PURSUIT_LATENCY_S * 1000:.0f}ms, gain={sc.PURSUIT_GAIN})")

    delta = pursuit_mean - still_mean
    pct = 100 * delta / still_mean if still_mean else float("inf")
    print(f"\npursuit - still = {delta:+.1f}px ({pct:+.1f}%)")
    if abs(pct) < 5.0:
        print("-> roughly a WASH under these assumptions (within normal seed-to-seed spread)")
    elif delta < 0:
        print("-> cursor-pursuit calibration looks BETTER under these assumptions")
    else:
        print("-> still-point calibration looks BETTER under these assumptions")

    print(
        "\nThis is a synthetic model of pursuit lag/gain, not a measurement of your\n"
        "actual tracking -- try --latency/--gain to see how sensitive the result is\n"
        "(e.g. --latency 0 --gain 1.0 for a best-case pursuit assumption), and confirm\n"
        "with a real recorded session (two live calibrate.py runs, one per protocol,\n"
        "compared via scripts/bench_calibration.py) before committing to a rewrite."
    )


if __name__ == "__main__":
    main()
