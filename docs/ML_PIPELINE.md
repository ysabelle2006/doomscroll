# Training our own gaze model

## The pipeline

```
record_session.py      build_dataset.py        train_model.py                train_cnn.py
webcam + cursor  --->  MediaPipe features ---> re-estimate CNN lag_ms   --->  retrain CNN (reads lag_ms)
(per group member)     (one .npz / session)    models/gaze_model.joblib
data/recordings/       datasets/features/
```

(`datasets/` is a separate top-level folder from `data/` -- see `datasets/README.md`: `data/recordings/` is just raw per-session staging, `datasets/` is the permanent, anonymized archive of what's actually trained on. The math model -- `scripts/calibrate.py` / `phase2_gaze_scroll.py --model math` -- is a completely separate pipeline that never touches any of this; see "Not done / next steps" below.)

| Step | Code | Why it's separate |
|---|---|---|
| Record | `recording/recorder.py` | No ML deps; light loop = steady fps. Anyone can collect. |
| Featurize | `ml/features.py`, `ml/dataset.py` | Slow (MediaPipe on every frame). Run once per session. |
| Label | `ml/labels.py` | Cheap; done at training time so lag/filters can be swept. |
| Train | `ml/training.py` | Fast; iterate on models and evaluation freely. |
| Use | `ml/model.py` | `predict_px(features, screen_size)`; same features as training. |

## How the sync works

One process, one loop, one clock. For every frame the camera delivers we stamp
`t = perf_counter() - t0`, read the cursor, then write the frame to
`webcam.mp4` and `(frame_idx, t_ms, cursor_x, cursor_y)` to `frames.csv`.
Frame *i* of the video is row *i* of the CSV, so there is nothing to align
afterwards. Consequences:

- Use `t_ms` from the CSV, never `frame_idx / fps`. Webcams drop and stretch
  frames; the mp4 fps header is only nominal. Real fps is in `meta.json`.
- The optional screen recording (`--screen`) uses the same clock; to find the
  screen frame for a webcam frame use `recording.screen.nearest_index`.
- It is not needed for training. The cursor log is the label.
- Residual error = camera latency + the eye arriving before the hand. Handled
  as ONE number, `lag_ms` (label = cursor at `t + lag`), found automatically by
  `train_model.py` (it always sweeps a handful of candidate lags now, no flag
  needed) and fed to the CNN's `train_cnn.py`.

> If a teammate already recorded screen/face with separate tools (OBS, QuickTime, ...)
> those can't be synced as reliably. Re-record with `record_session.py`, or write a
> converter that produces the same `frames.csv` format (see `recording/session.py`).

## Collecting good data (most important part)

The label is "where the cursor is", which equals "where the eyes are" ONLY if
the person is looking at the cursor. So:

1. Keep looking at the cursor and keep moving it over the whole screen
   (corners and edges especially). Idle-cursor frames are discarded automatically.
2. Cover head poses, with a note in `--notes`: forward, turned left/right,
   tilted up/down, closer/further. This is what fixes the head-angle problem:
   the model sees many head poses *with true labels*.
3. Vary lighting, glasses, time of day, and use more than one person.
   Each recording = its own session. More sessions = more trustworthy test numbers.
4. Aim for >= 25 fps and >= 60% usable frames (build_dataset warns otherwise).

```bash
python scripts/record_session.py --participant nicole --duration 300 --notes "head forward, bright room"
python scripts/record_session.py --participant nicole --duration 300 --notes "head turned left ~25deg"
```
`data/` is git-ignored on purpose: it's face video. Share it privately, with consent.

## Re-estimating the CNN's cursor-lag

```bash
python scripts/build_dataset.py
python scripts/train_model.py
```
`train_model.py` doesn't train or compare a standalone predictor anymore -- it
sweeps a handful of candidate lags (ridge only, as a quick, cheap proxy)
against whatever's currently in `datasets/features/`, and stores whichever
lag scored lowest as `lag_ms` in `models/gaze_model.joblib`. `train_cnn.py`'s
`default_lag()` reads that automatically on its next retrain. `app.py`'s
"Record, train & calibrate" already runs both of these steps itself (right
after `build_crops.py`, right before retraining the CNN), so this isn't
something you'd normally run by hand.

The old multi-candidate comparison (`ridge`/`mlp`/`hgb`, with held-out error
by head-yaw and by participant) used to live here, for researching the math
model's own recipe -- it's still in git history if that's ever worth
revisiting. But `scripts/calibrate.py`'s `GazeCalibrator` (the actual live
math model -- see "Shorter calibration" below) never loaded anything this
script produced, so keeping that comparison as a production step was cost
with no runtime benefit.

## Shorter calibration

The two prediction pipelines calibrate independently, each with the
mechanic suited to how it actually works:

- **Math model**: `scripts/calibrate.py` fits a `GazeCalibrator` from
  scratch every session -- there's no separate pretrained math model to
  build on, so it needs the longer dot-grid or continuous-trajectory
  capture (see its own docstring's `--protocol`).
- **CNN**: `scripts/quick_calibrate.py` instead fits only a tiny
  per-user scale+shift (`ml/model.py`'s `AffineCorrection`) on top of
  the CNN already trained offline (see "CNN on eye crops" below) -- a
  quick 9-dot round (~15s), since the CNN already knows the general
  mapping and only needs nudging for this person/camera/seat. Only
  saves the fit if its leave-one-out error beats "no correction".

Run the app with `phase2_gaze_scroll.py --model math` (or `--model cnn`).

## Changing things safely

- New/changed feature: edit `FEATURE_NAMES` and `extract_features` together
  (`ml/features.py`), then `build_dataset.py --force` and retrain. Old datasets
  and models refuse to load against changed features instead of failing silently.
- New model type: add one line to `CANDIDATES`.
- Tests: `pytest tests` (no camera/MediaPipe needed; uses fakes).

## CNN on eye crops (appearance-based model)

Instead of hand-made landmark numbers, a small CNN looks at the eye pixels.

```bash
pip install torch                                  # or: uv pip install -e ".[macos,cnn]"
python scripts/build_crops.py --preview            # MediaPipe once -> datasets/crops/*.npz (+ crop contact sheets)
python scripts/train_cnn.py                        # held-out test vs ridge/hgb on the same frames, then save
python scripts/quick_calibrate.py                   # per-user correction, quick 9-dot round (~15s)
python scripts/phase2_gaze_scroll.py --model cnn
```

| Piece | Code | Notes |
|---|---|---|
| What pixels the CNN sees | `ml/eye_crops.py` | Each eye cut along its corner line (roll removed), sized by eye width, 36x60, hist-equalized; 64x64 face. Used offline AND live. |
| Crop dataset | `ml/crop_dataset.py` | Like `dataset.py`; crops stored for usable frames only; labels made at train time. |
| Model + training | `ml/cnn.py` | Shared eye tower x2 + face tower + the 11 landmark features -> FC -> (x, y) screen fraction. Huber loss in pixels. |
| Live | `CnnGazeModel` via `ml.model.load_gaze_model` | `.pt` -> CNN, `.joblib` -> sklearn; same `predict_frac_frame(points, pose, gray)`. CPU, ~ms per frame. |

Honest evaluation is the same as before: whole sessions held out (>= 3 sessions),
early stopping on the last 10% of each *training* session, and ridge/hgb are
re-run on exactly the same frames and folds so the table is apples to apples.
`--no-aux` trains a pixels-only network: the gap between it and the default
shows how much the images add on top of the landmark features.

Data reality check: a CNN is data-hungry. With one person it will learn *that
person*; expect it to beat ridge/hgb on held-out sessions of the same person
first, and only generalise to new people after several people are recorded.
More sessions (different lighting, head poses, days) help more than more epochs.
Look at the `--preview` contact sheet before training: if eyes aren't centred
in the crops, nothing downstream will work.

## Not done / next steps

- Temporal model (a few frames stacked) and per-user fine-tuning of the CNN's last layer.
- `--model math` (scripts/calibrate.py and scripts/phase2_gaze_scroll.py's default) is a
  completely separate pipeline from everything in this doc -- GazeCalibrator on the
  handcrafted feature vector, nothing here affects it.
