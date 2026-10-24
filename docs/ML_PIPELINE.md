# Training our own gaze model

## The pipeline

```
record_session.py      build_dataset.py        train_model.py          phase2_gaze_scroll.py --model
webcam + cursor  --->  MediaPipe features ---> compare models, save --> live scrolling
(per group member)     (one .npz / session)    models/gaze_model.joblib  (+ quick_calibrate.py, ~15 s)
data/recordings/       data/datasets/
```

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
  as ONE number, `lag_ms` (label = cursor at `t + lag`), found with
  `train_model.py --sweep-lag`.

> If a teammate already recorded screen/face with separate tools (OBS, QuickTime, ...)
> those can't be synced as reliably. Re-record with `record_session.py`, or write a
> converter that produces the same `frames.csv` format (see `recording/session.py`).

## Collecting data by playing games

Games give varied, unscripted eye and cursor movement, but only if the cursor
is a real pointer. Choose accordingly:

| Good | Bad |
|---|---|
| Mouse-driven games: strategy/RTS, city builders, card games, osu!/aim trainers, puzzle, browser games, map-based games | Mouse-look games (FPS / third-person): the game locks the cursor to the centre, so cursor position says nothing about gaze |
| Windowed or borderless window, on the monitor the camera is on | Exclusive fullscreen (cursor/coordinates can differ), gamepad games (no cursor) |

```bash
python scripts/record_session.py --participant nicole --activity game:osu --duration 900 --notes "head forward"
```
- **F8** stops, **F9** pauses/resumes. These are global, so they work while the game has focus.
  Pause for menus, loading screens and chat. Paused frames are kept in the files (so video and CSV
  stay in step) but flagged `active=0` and never trained on.
- The recorder logs mouse-button state. People look at what they click, so
  `train_model.py --label-mode clicks` (only frames just before a click) is the cleanest,
  smallest label set; `either` combines it with "cursor recently moving". Compare them by
  held-out error rather than assuming.
- `--max-speed` (default 3000 px/s) drops flick moves the eyes can't follow.
- After each recording a warning appears if the cursor sat on one point for >50% of the
  session (the game captured the mouse). Discard such sessions.
- `train_model.py` prints a screen-coverage grid. Empty cells mean the model never saw
  those regions; play something that uses the whole screen, or add a session that does.
- Games bias gaze toward the centre and toward targets, so keep a few minutes of
  free cursor sweeping to the edges/corners in the mix, since the scroll trigger works at the bottom edge.

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

## Training and honest evaluation

```bash
python scripts/build_dataset.py
python scripts/train_model.py --sweep-lag
```
Candidates are in `CANDIDATES` (`ml/training.py`): `ridge` (stand-in for the old
linear calibration, i.e. the baseline), `mlp`, `hgb`. Error is measured on
held-out whole sessions (>= 3 sessions) or 20 s time blocks (fewer). A random
frame split would leak near-duplicate neighbouring frames and look great but
mean nothing. The report also prints error per head-yaw range, which is the
direct evidence for "does head angle still hurt?". For the TA report, show that
table for `ridge` on the old 7 features vs the new model.

## Shorter calibration

The shared model learns the general mapping; `quick_calibrate.py` shows 9 dots
(~15 s) to fit only a scale+shift (`AffineCorrection`) for this user, and only
saves it if its leave-one-out error beats "no correction".
Run the app with `phase2_gaze_scroll.py --model`.

## Changing things safely

- New/changed feature: edit `FEATURE_NAMES` and `extract_features` together
  (`ml/features.py`), then `build_dataset.py --force` and retrain. Old datasets
  and models refuse to load against changed features instead of failing silently.
- New model type: add one line to `CANDIDATES`.
- Tests: `pytest tests` (no camera/MediaPipe needed; uses fakes).

## Not done / next steps

- A CNN on eye crops (true appearance-based model) usually beats landmark features
  but needs far more data (tens of thousands of frames, many people). Add
  eye-crop saving in `build_dataset` and a torch model in `ml/` if the data allows.
- The old `calibrate.py` and the non-`--model` path in `phase2_gaze_scroll.py` are untouched.
