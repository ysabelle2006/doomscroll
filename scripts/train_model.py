"""Re-estimate the CNN's cursor-lag from the built feature datasets.

    python scripts/train_model.py

Writes the result to models/gaze_model.joblib's metadata, where
scripts/train_cnn.py's default_lag() picks it up automatically on its next
retrain -- app.py's "Record, train & calibrate" flow runs build_dataset.py
then this script, right before retraining the CNN, so every new recording
feeds back into the CNN's lag estimate without anyone having to run this
by hand.

This used to also train and compare a standalone predictor for the math
model (ridge vs. mlp vs. histogram-gradient-boosting) for research into
its own recipe. That's not needed in production: scripts/calibrate.py's
GazeCalibrator is fit fresh from each session's own calibration sweep and
never loads anything from this file, and nothing else loads a saved
predictor out of it either -- the only thing anything in this project
reads back out of models/gaze_model.joblib is train_cnn.py's
default_lag(), which just wants metadata["lag_ms"]. So that's the only
thing computed and saved here now. The old multi-model comparison/
evaluation tooling is still in git history if that research is ever
revisited.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from doomscroll.ml.dataset import load_all
from doomscroll.ml.model import DEFAULT_MODEL_PATH
from doomscroll.ml.training import evaluate_candidates, pool
from doomscroll.recording.session import FEATURES_DIR

WINDOW_MS = 600.0
MIN_TRAVEL_PX = 40.0
LAG_SWEEP_MS = (-100, 0, 100, 200, 300)


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
