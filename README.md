# doomscroll

Gaze/head-tracking, content-aware auto-scroll. Uses your webcam + MediaPipe
face landmarks to figure out where on screen you're looking, and scrolls the
focused window when you dwell near the bottom edge.

This is Phase 2 of the project: a trained gaze model (or per-user calibration)
+ an out-of-distribution confidence gate + a naive dwell trigger ("look at the
bottom 15% of the focused window for 1s -> scroll down"). No content-awareness
yet (that's Phase 3/4 — see [Project layout](#project-layout)).

## Requirements

- Python 3.11+ (developed against 3.13)
- A webcam
- Windows or macOS (Linux is not implemented for the scroll/window backends)

## Setup

```bash
# from the project root (the folder containing pyproject.toml)
python3 -m venv .venv

# macOS/Linux
source .venv/bin/activate
pip install -e ".[macos]"

# Windows (PowerShell)
.venv\Scripts\activate
pip install -e ".[windows]"
```

If `pyobjc` fails to build on macOS, install the Xcode command-line tools
first: `xcode-select --install`.

```bash
pip install "mediapipe==0.10.31"
```

(Or just `uv sync` — a `uv.lock` is checked in.)

### macOS permissions

Grant these to whatever process runs Python (Terminal, or your IDE's
integrated terminal — e.g. "Visual Studio Code" rather than "Terminal" if
you're running from there) in **System Settings > Privacy & Security**:

- **Camera** — for webcam capture
- **Screen Recording** — for reading the focused window's bounds
- **Accessibility** — for injecting synthetic scroll events into other apps

macOS only lists an app under a permission category after that app has
actually *tried* to use it once. So the first run of each script below may
need to happen twice: once to trigger the permission prompt/list entry, and
again after granting it.

## Testing, in order

Run everything from the project root, with the venv activated.

1. **`python scripts/debug_view.py`** — live webcam + face-mesh landmarks +
   head pose overlay. Confirms the camera and MediaPipe are working. Press
   `q` to quit.

2. **`python scripts/window_bounds_watch.py`** — prints the focused window's
   bounds on a timer. Switch/resize/full-screen a few windows and confirm the
   numbers update correctly. `Ctrl+C` to stop.

3. **`python scripts/scroll_test.py`** — **Windows only** (uses `msvcrt` for
   keypresses). Keypress-triggered synthetic scroll against whatever app is
   focused; validates event injection before anything else is built on it.
   Skip this on macOS.

4. **`python scripts/calibrate.py`** — multi-angle on-screen gaze
   calibration. Keep your head still at each block's orientation and move
   only your eyes to follow each dot. Ends with a validation pass reporting
   mean pixel error. Press `ESC` to abort.

5. **`python scripts/phase2_gaze_scroll.py`** — the real thing. Run
   calibration first (or use `--model`, see below). Click into a
   browser/PDF/other scrollable app to focus it, then look toward the bottom
   of that window for about a second. Press `q` in the debug window to quit.

The debug overlay in step 5 prints live status (head pose, gaze fraction,
dwell progress) on the frame, and prints `SCROLL triggered` to the console
each time a scroll fires — useful for telling apart "the trigger never
fires" (calibration/window-bounds issue) from "it fires but nothing scrolls"
(permissions/injection issue).

### Tests

```bash
pytest tests
```

No camera/MediaPipe needed — `tests/test_recording_pipeline.py` runs the
record -> build-dataset -> train pipeline end to end against fake camera,
cursor, and feature-extractor fixtures.

## Training our own gaze model

Record webcam + cursor, build features, train, run with `--model`. Full
explanation (how sync works, how to collect data, how to evaluate) is in
[`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md).

```bash
python scripts/record_session.py --participant <name> --duration 300
python scripts/build_dataset.py
python scripts/train_model.py --sweep-lag
python scripts/quick_calibrate.py            # optional, ~15 s
python scripts/phase2_gaze_scroll.py --model models/gaze_model_4person.joblib
```

`train_model.py` writes to `models/gaze_model.joblib` by default — that path
is git-ignored (see [What's not committed](#whats-not-committed)), so a fresh
clone must either retrain locally or point `--model` at the committed
`models/gaze_model_4person.joblib`.

Evaluation/analysis scripts (all write into `lab_analysis/`, which is also
git-ignored — regenerate as needed):

- `scripts/evaluate_pipeline.py` — leave-person-out error for the production
  pipeline (raw HGB vs. + quick calibration vs. + calibration + OOD gate).
- `scripts/evaluate_ood.py` — sweeps out-of-distribution detector methods
  (mahalanobis / isolation forest / LOF) and rejection thresholds.
- `scripts/compare_ridge_hgb_pipeline.py` — leave-one-participant-out
  comparison of ridge vs. HGB through the full production pipeline.

## Known issues & fixes already applied here

- **mediapipe 1.0.x crashes on macOS** with a `DrishtiMetalHelper` /
  `Service is unavailable` fatal error inside the face-detector sub-graph,
  even with `delegate=BaseOptions.Delegate.CPU` set. Pin to `mediapipe==0.10.21`
  (see Setup above) until this is resolved upstream.
- **`landmarks/mediapipe_backend.py`** now explicitly passes
  `delegate=BaseOptions.Delegate.CPU` to `BaseOptions`. Without it, some
  mediapipe versions default to GPU on macOS, which is unsupported/unstable
  there.
- **`capture/webcam.py`** had a bug where a device index (e.g. `0`) was
  stringified (`str(self._source)`) on any non-Windows platform before being
  passed to `cv2.VideoCapture`. This turns the *integer camera index* `0`
  into the *string filename* `"0"`, which OpenCV then fails to open as a
  video file (`Couldn't read video stream from file "0"`). Fixed to pass
  integer sources through unchanged on macOS/Linux.
- **Synthetic scroll events silently doing nothing on macOS**: `CGEventPost`
  is silently dropped by macOS if the calling process lacks Accessibility
  permission — no exception is raised. If `phase2_gaze_scroll.py` prints
  `SCROLL triggered` but nothing actually scrolls, check Accessibility
  permissions first.

## Project layout

```
src/doomscroll/
  capture/      webcam frame source
  landmarks/    MediaPipe face-landmark backend
  gaze/         gaze vector, head pose, calibration (quick + full), pupil
                refinement, smoothing, trajectory paths, session storage
  fixation/     attention gate + window-relative mapping
  recording/    data collection: webcam + cursor recorder, session format,
                optional screen recording
  ml/           features, labels, dataset building, training, trained-model
                wrapper, OOD (out-of-distribution) gating
  scroll/       platform scroll injectors (Windows/macOS)
  windowing/    platform focused-window-bounds providers (Windows/macOS)
  debug/        live debug overlay (head pose, gaze, dwell progress)
  config/       (reserved — empty placeholder)
  content/      (reserved — Phase 3/4 content-awareness, not implemented yet)
  dpi.py        DPI/screen-scaling helpers
scripts/        the test/validation/training/evaluation scripts described above
models/         face_landmarker.task (MediaPipe model) + the committed
                production gaze model (see below)
docs/           ML_PIPELINE.md — data collection, training, evaluation deep-dive
tests/          pytest suite (fakes everything camera/MediaPipe-related)
```

## What's not committed

`.gitignore` excludes, on top of the usual Python build/cache artifacts:

| Path | Why |
|---|---|
| `data/` | Raw webcam recordings and `.npz` training datasets — personal/biometric data (face video). Never commit; share privately, with consent. |
| `lab_analysis/` | Generated evaluation CSVs/PNGs/frame crops from the scripts above — regenerate locally rather than committing stale output. |
| `lab3/` | Lab-course deliverables (diagrams, reports) unrelated to the running application. |
| `models/gaze_model.joblib`, `models/gaze_model_qiyun_ysabelle.joblib`, `models/gaze_model_jingyi_qiyun_ysabelle.joblib` | Superseded/intermediate trained models. |

Still committed: `models/face_landmarker.task` (MediaPipe's model, required to
run anything) and `models/gaze_model_4person.joblib` (the current production
model — pass it explicitly via `--model`, since it isn't the default output
path of `train_model.py`).
