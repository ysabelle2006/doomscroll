"""Per-user calibration for the CNN gaze-model pipeline (--model cnn in
scripts/calibrate.py and scripts/phase2_gaze_scroll.py).

The CNN itself is trained once, offline, on data_raw recordings (see
scripts/build_crops.py and scripts/train_cnn.py, following
docs/ML_PIPELINE.md) -- never during calibration. Calibrating only fits a
lightweight per-user AffineCorrection (see doomscroll.ml.model) on top of
that pretrained model's own raw predictions, using the exact same
trajectory dot-sweep capture the math-model pipeline uses (see
scripts/calibrate.py's run_trajectory_capture(), via its sample_fn hook):
during the sweep, every usable frame's CNN prediction is paired with the
dot's own (lag-corrected) position, and at the end those pairs are fit
into one scale+shift correction instead of a GazeCalibrator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from doomscroll.gaze.head_pose import HeadPose
from doomscroll.ml.model import AffineCorrection, correction_path_for, fit_user_correction, load_gaze_model

# Deliberately NOT imported from doomscroll.ml.cnn: that module imports torch
# at the top (needed for the CNN itself), so importing it eagerly here would
# make torch a hard requirement just to import this module, even for code
# that never touches the CNN pipeline. Same literal path ml.cnn.DEFAULT_CNN_PATH
# resolves to -- CnnGazeModel.save()'s own default.
DEFAULT_CNN_PATH = Path(__file__).resolve().parents[3] / "models" / "gaze_cnn.pt"


def load_pretrained_cnn(path: Path | str = DEFAULT_CNN_PATH):
    """The CNN trained offline by scripts/train_cnn.py, with no per-user
    correction applied yet -- calibration is what produces that
    correction, so loading one already baked in here would double it up."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No trained CNN model at {path}. Run scripts/build_crops.py then "
            "scripts/train_cnn.py first (see docs/ML_PIPELINE.md)."
        )
    return load_gaze_model(path, with_correction=False)


def make_cnn_sample_fn(
    cnn_model,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, HeadPose], np.ndarray | None]:
    """Builds the per-frame sample_fn run_trajectory_capture() takes: the
    CNN's own raw (uncalibrated) screen-fraction prediction for this
    frame, in place of the math pipeline's handcrafted feature vector.
    run_trajectory_capture() still does everything else identically for
    both pipelines -- lag-correcting the label, discarding frames near a
    turn/anchor, grouping."""

    def sample_fn(
        frame_bgr: np.ndarray, frame_gray: np.ndarray, points_px: np.ndarray, head_pose: HeadPose
    ) -> np.ndarray | None:
        return cnn_model.predict_frac_frame(points_px, head_pose, frame_gray)

    return sample_fn


def fit_and_save_cnn_correction(
    raw_preds_frac: np.ndarray,
    targets_px: np.ndarray,
    screen_wh: tuple[int, int],
    model_path: Path | str = DEFAULT_CNN_PATH,
) -> tuple[AffineCorrection, float, float, bool, Path]:
    """raw_preds_frac: (N, 2) the CNN's own predict_frac_frame() output
    captured during the trajectory sweep (what make_cnn_sample_fn()'s
    sample_fn returned, same array run_trajectory_capture() would
    otherwise have filled with handcrafted feature vectors). targets_px:
    (N, 2) the matching dot positions in pixels -- run_trajectory_capture()
    returns these in pixels regardless of pipeline, so they're converted
    to fractions here since AffineCorrection works in fraction space, same
    as the model's own output.

    Saves to the CNN-specific correction path (correction_path_for(),
    e.g. affine_correction_cnn.json) so it never collides with the math
    pipeline's own GazeCalibrator profile. If the fitted correction's
    leave-one-out error isn't actually better than the CNN's raw
    predictions, an identity correction is saved instead (the "don't use
    it if it doesn't help" rule doomscroll.ml.model.fit_user_correction()
    describes) rather than a correction that would make things worse.

    Returns (correction actually saved, error before, leave-one-out error
    after, whether the fitted correction was the one used, path saved to).
    """
    targets_frac = targets_px / np.array(screen_wh, dtype=np.float64)
    fitted, before_px, after_px = fit_user_correction(raw_preds_frac, targets_frac, screen_wh)
    used_fit = after_px < before_px
    correction = fitted if used_fit else AffineCorrection.identity()
    path = correction_path_for(model_path)
    correction.save(path)
    return correction, before_px, after_px, used_fit, path
