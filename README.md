# doomscroll

Gaze/head-tracking content-aware auto-scroll. Tracks where you're looking via webcam, maps that to screen coordinates with a calibrated model, and (eventually) uses it to drive content-aware scroll behavior.

## Setup

Requires Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`).

```
uv sync
```

(or `pip install -e .` if you're not using `uv`). On Windows, also install the `windows` extra (`pywin32`); on macOS, the `macos` extra (`pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa`) — these back the OS-specific scroll/windowing backends.

The MediaPipe face landmark model is already checked into `models/face_landmarker.task` — no separate download needed.

## Running it

1. **`scripts/debug_view.py`** — Phase 1 sanity check: webcam capture + face-mesh landmarks + head pose, rendered live. Run this first to confirm your camera and landmark detection are working before anything else.
2. **`scripts/calibrate.py`** — run this before the main app (see Calibration below).
3. **`scripts/phase2_gaze_scroll.py`** — the end-to-end loop: calibrated gaze mapping + attention gate + a trivial dwell-to-scroll rule. Press `q` to quit.

## Calibration

```
python scripts/calibrate.py
```

Two passes:

- **Main pass** — a 16-point on-screen grid. Keep your head still and forward, track each dot with your eyes only.
- **Head-pose variation pass** — immediately after, you'll be asked to hold a few different natural head positions (lean left/right, lean closer/sit back, tilt chin up/down) while repeating a short dot sequence at each. This teaches the model to correct for incidental head movement instead of only ever having seen you sit still. Each pose shows a live readout and waits for you to press SPACE once you're actually holding the position (green = roughly there; it's a visual guide only — SPACE always advances regardless of color, since everyone's comfortable "slight lean" differs). Press ESC at any time to abort.

A validation pass follows, reporting mean pixel error.

**Flags:**

- `--pool` — also fits against this session pooled with your most recent stored sessions, and keeps whichever fit (session-only or pooled) actually validates better. Pooling is opt-in, not default, because your physical setup (seating, lighting, camera position) drifts between sittings, and pooling across a setup change can make the fit worse, not better — see `src/doomscroll/gaze/session_store.py`'s docstring for the full reasoning. `--pool` prints both validation numbers so you can judge empirically.
- `--list-history` — list stored calibration sessions and exit.
- `--clear-history` — delete all stored calibration sessions and exit. Do this after any real setup change (new desk, new lighting, different chair height).

Every run is saved as its own timestamped session under your platform's app-data directory (`CALIBRATION_DIR/sessions/`, via `platformdirs`), independent of pooling.

## Testing

```
pytest
```

Runs the unit/synthetic test suite (`tests/`) — gaze vector math, head pose, smoothing, calibration fitting, session storage, and the pose-variant readout logic — all against synthetic, deterministic fixtures (no camera or MediaPipe model needed).

(pytest creates a `.pytest_cache/` directory to track state like last-failed tests between runs. It's local scratch state, not something to commit — pytest already drops its own `.gitignore` inside that directory, so it's self-ignoring regardless of this project's `.gitignore`.)

To check whether a change to the calibration pipeline itself helped or hurt, using your own *real* recorded data:

```
python scripts/bench_calibration.py --save-baseline   # once, to establish a baseline
python scripts/bench_calibration.py                    # after each change, to compare against it
```

This re-fits against whatever `scripts/calibrate.py` most recently captured (`CALIBRATION_DIR/last_dataset.npz`) and reports IMPROVED/REGRESSED/UNCHANGED. A real baseline is also checked in at `tests/fixtures/calibration_baseline.json` for reference.

To compare the still-point vs. cursor-pursuit calibration *protocols* (a different question — holds the model fixed, varies how data is captured) on fully synthetic, paired data:

```
python scripts/compare_calibration_protocols.py
```

## Project layout

```
scripts/            Entry-point scripts (calibrate, debug view, main app, benchmarking tools)
src/doomscroll/
  capture/           Webcam input
  landmarks/         Face landmark backends (MediaPipe)
  gaze/              Gaze vector math, head pose, calibration model, session storage, smoothing
  fixation/          Attention gate + window-relative mapping
  scroll/            Platform scroll backends (macOS/Windows)
  windowing/          Platform window-focus backends (macOS/Windows)
  debug/             Debug overlay rendering
tests/               Unit + synthetic integration tests
models/              Checked-in MediaPipe face landmark model
```
