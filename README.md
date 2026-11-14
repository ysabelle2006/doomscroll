# doomscroll

Gaze/head-tracking content-aware auto-scroll. Tracks where you're looking via webcam, maps that to screen coordinates with a calibrated model, and uses it to drive content-aware scroll behavior.

Demo Video: https://youtu.be/ZLQ8kBFiwiI

## Setup

Requires Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`).

```
uv sync
```

(or `pip install -e .` if you're not using `uv`). On Windows, also install the `windows` extra:

```
uv sync --extra windows   # installs pywin32
```

On macOS, the scroll/windowing backends and the Vision-based OCR content scanner (`pyobjc-framework-Quartz`/`Cocoa`/`Vision`) are plain `sys_platform == 'darwin'`-gated dependencies, so a plain `uv sync` on macOS already includes them -- no separate extra needed.

The MediaPipe face landmark model is already checked into `models/face_landmarker.task` -- no separate download needed. A pretrained CNN gaze model (`models/gaze_cnn.pt`) and a baseline training dataset (`datasets/crops/`, `datasets/features/`) are checked in too, so the CNN path already works on a brand-new clone, before you've recorded anything yourself.

The CNN pipeline (used by "Create personal model" / "Quick personal calibration" below -- the math-model pipeline doesn't need it) additionally needs `torch`:

```
uv sync --extra cnn
```

## Running the app

```
uv run app.py
```

(or `python app.py` with the venv active). This opens `app.py`'s Tk launcher -- the actual way to run doomscroll day to day. It's a thin GUI over the scripts in `scripts/`: it calls them as subprocesses with the right flags and in the right order, and surfaces their progress/output as popups and a log, rather than reimplementing anything itself. Those scripts can also be run by hand (see **Pipelines and system design** below) if you want more control.

The main page has two setup paths:

- **Standard Setup -- "Start eye tracking"**: runs `scripts/calibrate.py` (the math model -- no training step) with your choice of calibration type (Fast/trajectory or Precision/dots), optionally pooled with past sessions (`--pool`), then launches `scripts/phase2_gaze_scroll.py` to start scrolling. The quickest way to get the app running.
- **Personalised Setup**:
  - **"Create personal model"**: records a webcam session for however long you set (`scripts/record_session.py`), builds a crop dataset and a features dataset from it (`scripts/build_crops.py` / `build_dataset.py`), re-estimates the CNN's cursor lag (`scripts/train_model.py`), retrains the CNN on every session now in `datasets/crops/` (`scripts/train_cnn.py`), fits a quick per-user correction on top of it (`scripts/quick_calibrate.py`), then launches `phase2_gaze_scroll.py --model cnn`. Slowest path, but the most accurate, since it actually trains a model on your own eyes.
  - **"Quick personal calibration"**: skips recording/training and only runs `quick_calibrate.py` then `phase2_gaze_scroll.py --model cnn` against whichever CNN is already saved at `models/gaze_cnn.pt`. Requires a CNN to already exist (either the checked-in one, or one "Create personal model" produced).

Once scrolling is running, a "Stop Scrolling" button appears on the main page. (Running a script directly from a terminal instead of through the launcher: Ctrl+C stops it.)

The **Settings** page has:

- a live log of every subprocess the launcher runs, for debugging;
- light/dark appearance, independent of the OS theme;
- a **Developer view** toggle, which runs every step with `phase2_gaze_scroll.py --dev-view`, drawing the live OCR bounding-box overlay over whatever window you're reading, so you can see what the content scanner is doing;
- **Clear all calibrations**, which deletes the saved math-model profile, the CNN's saved per-user correction, and every stored calibration session (the ones `--pool` pools against) -- both pipelines need a full recalibration afterward.

## Running tests

```
uv run pytest
```

(or `pytest` with the venv active). Runs the suite under `tests/` -- gaze vector math, head pose estimation, pupil refinement, smoothing, calibration fitting (including synthetic end-to-end accuracy benchmarks), calibration session storage, the pose-variant/front-readout calibration UI logic, the attention gate, the content-aware scroll trigger, the macOS Vision bounding-box mapping, and the full recording -> dataset -> training -> model pipeline (the CNN tests need `torch`; they're skipped automatically via `pytest.importorskip` if it isn't installed). Everything runs against synthetic, deterministic fixtures -- no camera, MediaPipe model, or display needed.

pytest creates a `.pytest_cache/` directory to track state like last-failed tests between runs -- it's local scratch state, not something to commit (pytest drops its own `.gitignore` inside it).

## File structure

```
app.py                    Tk launcher -- the app's actual entry point (see "Running the app")

scripts/                  Command-line entry points the launcher calls as subprocesses
  debug_view.py              Phase-1 sanity check: webcam + landmarks + head pose, live
  calibrate.py               Math-model calibration (dot-grid or continuous-trajectory protocol)
  quick_calibrate.py         CNN per-user correction (quick 9-dot pass)
  record_session.py          Records one webcam+cursor (+ optional screen) data-collection session
  build_crops.py             Session(s) -> eye/face crop dataset (datasets/crops/), for the CNN
  build_dataset.py           Session(s) -> handcrafted-feature dataset (datasets/features/), for the math model's lag estimate
  train_model.py             Re-estimates the CNN's cursor-lag from datasets/features/
  train_cnn.py               Trains the CNN on datasets/crops/, evaluated against ridge/hgb baselines
  phase2_gaze_scroll.py      The live runtime: calibrated gaze + content-aware scroll loop

src/doomscroll/           The installable package
  capture/                   Webcam input (webcam.py)
  landmarks/                 Face landmark backends (base.py, mediapipe_backend.py)
  gaze/                      Gaze vector math, head pose, Kalman smoothing, the math calibration model, calibration session storage, trajectory-capture helpers
  fixation/                  Window-relative gaze mapping, the attention gate, the scroll-trigger dwell logic
  content/                   On-screen text block scanning (macOS Vision OCR) + gaze-vs-content dwell tracking
  ml/                        Training/inference pipeline: features, labels, datasets (handcrafted + crop), ridge/hgb training, the CNN, and model load/save/correction
  recording/                 Session recording (webcam+cursor+screen capture) and on-disk session layout
  scroll/                    Platform scroll-injection backends (macOS/Windows)
  windowing/                 Platform focused-window-bounds backends (macOS/Windows)
  debug/                     Debug overlays (webcam HUD + the macOS on-screen content overlay)
  dpi.py                     Windows DPI-awareness + cross-platform screen size

tests/                     Unit + synthetic integration tests (see "Running tests")

models/                    Checked-in: face_landmarker.task (MediaPipe), gaze_cnn.pt (CNN), gaze_model.joblib (CNN's lag estimate)
datasets/                  Checked-in baseline training data (crops/, features/) -- see datasets/README.md
lab1/ .. lab5/             SC2006 course lab deliverables (requirements, design docs, diagrams) -- unrelated to running the app
INDEX.md                   Index of the lab1-lab5 deliverables
```

(`data/` -- raw per-session recordings from `record_session.py` -- is created at runtime and is git-ignored; "Create personal model" deletes a session's raw files from it automatically once they've been extracted into `datasets/`.)

## Pipelines and system design

doomscroll predicts where you're looking on screen and scrolls down automatically once it decides you've finished reading what's currently visible. There are two independent ways it turns a webcam frame into a screen point (the math model and the CNN), and one shared decision loop that turns that point into a scroll.

### 1. Gaze estimation: two independent pipelines

Both pipelines start from the same MediaPipe face landmarks (`landmarks/mediapipe_backend.py`) and head pose estimate (`gaze/head_pose.py`, solved with OpenCV's `solvePnP`), but diverge from there.

**Math model** (`gaze/calibration.py`'s `GazeCalibrator`, used by `scripts/calibrate.py` / `phase2_gaze_scroll.py --model math`, the default):

- Builds a 10-value feature vector per frame (`build_feature_vector`): the 4-value handcrafted gaze vector (mean + per-eye-difference iris offset, `gaze/gaze_vector.py`), plus head pose (yaw/pitch/roll) and head translation (x/y/z).
- Fits a polynomial `RidgeCV` straight from that feature vector to a screen point, plus a local-correction term for nearby training samples -- entirely from the calibration session you just did. There's no separate pretrained math model to build on, which is why its calibration is the longer dot-grid or continuous-trajectory capture.
- Calibration sessions are saved to disk (`gaze/session_store.py`) and can optionally be pooled with past sessions (`--pool`) that share the same screen size and capture protocol.

**CNN model** (`ml/cnn.py`'s `GazeNet`, used by `scripts/train_cnn.py` / `phase2_gaze_scroll.py --model cnn`):

- Looks at eye/face pixels directly instead of only handcrafted numbers: two eye crops through one shared-weight conv tower (`ml/eye_crops.py`, 36x60, roll-corrected, hist-equalized), plus a 64x64 face crop through its own tower, concatenated with the same 11-value landmark feature vector used by the ridge/hgb baselines (`ml/features.py`), through fully-connected layers to a screen fraction. Trained with a Huber-style (smooth L1) loss in pixel space.
- Has to be trained offline first (`scripts/train_cnn.py`, on `datasets/crops/`), since it's a real neural net rather than a per-session fit. Because that's slow and data-hungry, the repo ships a pretrained `models/gaze_cnn.pt` plus the baseline dataset it was trained on, so the CNN path works on a fresh clone.
- Calibrating it per-user only needs a tiny affine (scale+shift) correction on top (`scripts/quick_calibrate.py`'s 9-dot pass, `ml/model.py`'s `AffineCorrection`) -- hence "quick" calibration.

### 2. Training data pipeline (feeds the CNN, and the CNN's lag estimate)

```
record_session.py  -->  build_crops.py    --MediaPipe-->  datasets/crops/*.npz    --> train_cnn.py   --> models/gaze_cnn.pt
(one session,       \-> build_dataset.py --MediaPipe-->  datasets/features/*.npz --> train_model.py --> lag_ms (read by train_cnn.py)
 one shared clock)
```

- `record_session.py` runs one process, one loop, one clock: every frame gets a timestamp and the cursor position at that instant, written to `webcam.mp4` + `frames.csv` (frame *i* of the video is row *i* of the CSV -- nothing to align afterward). The cursor position IS the training label, under the assumption the person was looking at it.
- `build_crops.py` / `build_dataset.py` each run MediaPipe once per session (slow) to turn raw video into crops/features -- this only has to happen once per session, not once per training run.
- Labels (which frame's cursor position to use, and a lag correction for the delay between the eye arriving and the hand following) are applied at *training* time (`ml/labels.py`), not at build time, so lag and filtering can be swept without re-running MediaPipe. `train_model.py` sweeps a handful of lag candidates (scored with ridge, as a cheap proxy) and saves the best as `lag_ms`; `train_cnn.py` reads that automatically on its next retrain.
- `train_cnn.py` holds out whole sessions (or whole people, with `--cv person`) for an honest test, with ridge and gradient-boosting baselines re-run on the exact same frames and folds so the comparison is fair, then trains the final model on everything.
- `datasets/` (crops + features) is a separate, permanent, anonymized archive from `data/` (raw per-session staging, never committed) -- see `datasets/README.md` for the naming/anonymization scheme. The math model never touches any of this; it only ever uses whichever session you just calibrated with.

### 3. The live runtime (`phase2_gaze_scroll.py`)

Per frame: webcam -> MediaPipe landmarks -> head pose -> (math feature vector + `GazeCalibrator`) or (crops + `CnnGazeModel`) -> a predicted screen point -> `gaze/smoothing.py`'s Kalman filter + EMA pass to de-jitter it. Three mostly-independent systems then act on that point:

- **Attention gate** (`fixation/attention_gate.py`): pauses scrolling automatically if your head pose says you've looked away for longer than a grace period, and resumes automatically once you look back -- no shortcut key, no manual pause.
- **Content-aware scroll trigger** (`fixation/scroll_trigger.py` + `content/gaze_content.py`): a 3-tier dwell-based decision -- (1) gaze near the bottom 15% of the window, (2) gaze has dwelled on and passed the lowest visible line of text, (3) every text block between the last scroll position and the current gaze has already been dwelled on long enough to count as read. Tiers 2-3 need on-screen text blocks from `content/macos_backend.py`'s Vision OCR scan, so they're macOS-only -- Windows always runs tier-1-only (bottom-of-window).
- **Scroll injection** (`scroll/`) and **focused-window bounds** (`windowing/`): platform backends (macOS via Quartz, Windows via win32) that actually send the scroll and tell the content scanner which window to OCR.

`debug/` draws optional visual aids: a webcam HUD (iris points, a gaze minimap) and, on macOS, a transparent on-screen overlay outlining the content blocks the scanner found and whether each counts as read yet (`--dev-view`).
