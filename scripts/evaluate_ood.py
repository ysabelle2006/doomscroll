"""Evaluate OOD-detector methods for gating catastrophic gaze predictions.

Fits each candidate detector (mahalanobis / isolation_forest / lof) on one
participant's held-IN training features, scores a held-OUT participant's
frames, and sweeps rejection thresholds (calibrated as percentiles of the
TRAINING score distribution, since that's what's actually available at
deployment time -- no test labels). For each threshold: rejection rate,
and accepted-frame mean/median/p90 pixel error.

    python scripts/evaluate_ood.py --out lab_analysis/ood_sweep.csv

Writes a CSV (one row per method x held-out-person x threshold-percentile)
and a PNG (rejection-rate vs p90, one panel per held-out person).
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
from doomscroll.ml.ood import METHODS, fit_detector

ROOT = Path(__file__).resolve().parents[1] / "data"
REJECT_PERCENTILES = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]


def pool(sessions_by_id, ids):
    X = np.vstack([sessions_by_id[i].X for i in ids])
    y = np.vstack([sessions_by_id[i].y_frac for i in ids])
    wh = np.vstack([np.tile(sessions_by_id[i].screen_wh, (len(sessions_by_id[i].X), 1)) for i in ids]).astype(float)
    return X, y, wh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    ap.add_argument("--participants", nargs="+", default=["jingyi", "qiyun", "ysabelle"])
    ap.add_argument("--lag-ms", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=Path("lab_analysis/ood_sweep.csv"))
    ap.add_argument("--plot", type=Path, default=Path("lab_analysis/ood_sweep.png"))
    args = ap.parse_args()

    settings = LabelSettings(lag_ms=args.lag_ms, mode="either")
    all_sessions = load_all(args.datasets)
    subset = [s for s in all_sessions if s.participant in args.participants]
    labeled = {s.session_id: s.labeled(settings) for s in subset}
    participant_of = {s.session_id: s.participant for s in subset}

    rows = []
    for held_out in args.participants:
        train_ids = [sid for sid, p in participant_of.items() if p != held_out]
        test_ids = [sid for sid, p in participant_of.items() if p == held_out]
        if not train_ids or not test_ids:
            continue
        Xtr, ytr, _ = pool(labeled, train_ids)
        Xte, yte, whte = pool(labeled, test_ids)

        print(f"[{held_out}] fitting regressor on {len(Xtr)} train frames, scoring {len(Xte)} held-out frames", file=sys.stderr)
        reg = MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=250, random_state=0))
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xte)
        err_px = np.linalg.norm((pred - yte) * whte, axis=1)

        for method in METHODS:
            print(f"[{held_out}] fitting OOD detector: {method}", file=sys.stderr)
            det = fit_detector(method, Xtr)
            train_scores = det.score(Xtr)
            test_scores = det.score(Xte)

            for pct in REJECT_PERCENTILES:
                threshold = np.percentile(train_scores, 100 - pct) if pct > 0 else np.inf
                accept = test_scores <= threshold
                rej_rate = 1 - accept.mean()
                if accept.sum() > 0:
                    acc_err = err_px[accept]
                    mean_e, med_e, p90_e = acc_err.mean(), np.median(acc_err), np.percentile(acc_err, 90)
                else:
                    mean_e = med_e = p90_e = float("nan")
                rows.append(dict(held_out=held_out, method=method, target_reject_pct=pct,
                                  actual_reject_rate=rej_rate, accepted_mean_px=mean_e,
                                  accepted_median_px=med_e, accepted_p90_px=p90_e,
                                  n_accepted=int(accept.sum()), n_total=len(accept)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        people = args.participants
        fig, axes = plt.subplots(1, len(people), figsize=(6 * len(people), 5), sharey=True)
        if len(people) == 1:
            axes = [axes]
        colors = {"mahalanobis": "#1e5fa8", "isolation_forest": "#b8860b", "lof": "#2e7d32"}
        for ax, person in zip(axes, people):
            for method in METHODS:
                sub = [r for r in rows if r["held_out"] == person and r["method"] == method]
                sub.sort(key=lambda r: r["actual_reject_rate"])
                xs = [r["actual_reject_rate"] * 100 for r in sub]
                ys = [r["accepted_p90_px"] for r in sub]
                ax.plot(xs, ys, marker="o", label=method, color=colors[method])
            ax.set_title(f"held out: {person}")
            ax.set_xlabel("rejection rate (%)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("accepted-frame p90 error (px)")
        axes[0].legend()
        fig.suptitle("OOD-gate tradeoff: rejection rate vs. accepted-frame p90 error (leave-person-out)")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150)
        print(f"wrote plot -> {args.plot}")
    except Exception as e:
        print(f"plotting skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
