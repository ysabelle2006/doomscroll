# Estimates the best cursor-lag value by sweeping candidates and
# scoring each with ridge regression.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from doomscroll.ml.dataset import load_all
from doomscroll.ml.model import DEFAULT_MODEL_PATH
from doomscroll.ml.training import evaluate_candidates, pool
from doomscroll.recording.session import FEATURES_DIR

# Labeling window/travel threshold and lag values to sweep.
WINDOW_MS = 600.0
MIN_TRAVEL_PX = 40.0
LAG_SWEEP_MS = (-100, 0, 100, 200, 300)


# Loads sessions, picks the best lag, and saves it to the model file.
def main() -> None:
    sessions = load_all(FEATURES_DIR)
    if not sessions:
        sys.exit(f"No sessions found in {FEATURES_DIR} -- run build_dataset.py first.")
    print(f"{len(sessions)} session(s): {', '.join(s.session_id for s in sessions)}")

    scores: dict[int, float] = {}
    for lag in LAG_SWEEP_MS:
        data = pool([s.labeled(lag, WINDOW_MS, MIN_TRAVEL_PX) for s in sessions], group_by="session")
        scores[lag] = evaluate_candidates(data, ["ridge"])["ridge"]["mean_px"]
        print(f"  lag {lag:+5d} ms -> ridge mean error {scores[lag]:.0f}px")

    best_lag = min(scores, key=scores.get)
    best_data = pool([s.labeled(best_lag, WINDOW_MS, MIN_TRAVEL_PX) for s in sessions], group_by="session")
    print(f"best lag: {best_lag} ms ({scores[best_lag]:.0f}px)")

    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": {
                "lag_ms": float(best_lag),
                "n_frames": int(len(best_data.X)),
                "sessions": [s.session_id for s in sessions],
            },
        },
        DEFAULT_MODEL_PATH,
    )
    print(f"saved lag_ms={best_lag} -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
