# doomscroll

Gaze/head-tracking content-aware auto-scroll. Tracks where you're looking via webcam, maps that to screen coordinates with a calibrated model, and uses it to drive content-aware scroll behavior.

## Setup

Requires Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`).

```
uv sync
```

(or `pip install -e .` if you're not using `uv`). On Windows, also install the `windows` extra (`pywin32`). On macOS, the scroll/windowing backends and the Vision-based OCR content scanner need `pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa`, and `pyobjc-framework-Vision` — these are plain `sys_platform == 'darwin'`-gated dependencies, so a plain `uv sync` / `pip install -e .` on macOS already includes them.

The MediaPipe face landmark model is already checked into `models/face_landmarker.task` — no separate download needed.

## Running it

1. **`scripts/debug_view.py`** — Phase 1 sanity check: webcam capture + face-mesh landmarks + head pose, rendered live. Run this first to confirm your camera and landmark detection are working before anything else.
2. **`scripts/calibrate.py`** — run this before the main app (see Calibration below).
3. **`scripts/phase2_gaze_scroll.py`** — the end-to-end loop: calibrated gaze mapping + attention gate + a 3-tier content-aware scroll decision (see `src/doomscroll/fixation/scroll_trigger.py`). Press `q` to quit.

**Live-tuning flags** (all optional; each has a default used when omitted):

- `--smoothing-process-var` / `--smoothing-measurement-var` / `--smoothing-ema-alpha` — tune `GazePointSmoother`'s Kalman filter and extra EMA pass (defaults: `8.0` / `1.5` / `0.7`). Lower process-var / higher measurement-var / higher ema-alpha = smoother but more lag; the reverse = more responsive but more jittery.
- `--blink-min-ear` — eye-aspect-ratio floor below which a frame is treated as a blink and skipped rather than fed to the smoother (default: `gaze_vector.DEFAULT_MIN_EYE_ASPECT_RATIO`, `0.08` — deliberately looser than calibration's own `0.12`, see `scripts/calibrate.py`'s `CALIBRATION_MIN_EAR` comment, so a genuine fast downward glance isn't mistaken for a blink). If the gaze point visibly dips and curves in a short semicircle/diagonal right around a real blink rather than freezing cleanly, that's a few narrowed-but-not-yet-closed frames leaking past this looser gate — try raising it toward `0.12`.

## Calibration

```
python scripts/calibrate.py
```

The default protocol (`--protocol trajectory`) sweeps a single dot across the whole screen on its own in a programmed path (a handful of anchor points, then a full-screen raster sweep), for about 40 seconds. You just follow it with your eyes — nothing to click or trace yourself. Every usable webcam frame during the sweep becomes a training sample (not just one per dot), labeled with the dot's own position a fixed lag earlier (to correct for smooth-pursuit lag), with frames near a sharp turn or an anchor arrival discarded.

Immediately after, you'll be asked to hold a few different natural head positions (lean left/right, lean closer/sit back, tilt chin up/down) while repeating a short capture at each. This teaches the model to correct for incidental head movement instead of only ever having seen you sit still. Each pose shows a live readout and waits for you to press SPACE once you're actually holding the position (green = roughly there; it's a visual guide only — SPACE always advances regardless of color, since everyone's comfortable "slight lean" differs). Press ESC at any time to abort.

A validation pass follows, reporting mean pixel error.

**`--protocol dots`** — an alternative to the continuous sweep above: a 16-point on-screen grid instead of a single moving dot. Keep your head still and forward, track each dot with your eyes only, for a fixed capture window per point. It fits the exact same `GazeCalibrator` used by the default trajectory protocol — just from a handful of per-dot samples instead of continuously-collected ones — and saves to the same profile, so switching between `--protocol dots` and `--protocol trajectory` costs nothing but a recalibration:

```
python scripts/calibrate.py --protocol dots
```

**Flags:**

- `--pool` — also fits against this session pooled with your most recent stored sessions, and keeps whichever fit (session-only or pooled) actually validates better. Pooling is opt-in, not default, because your physical setup (seating, lighting, camera position) drifts between sittings, and pooling across a setup change can make the fit worse, not better — see `src/doomscroll/gaze/session_store.py`'s docstring for the full reasoning. `--pool` prints both validation numbers so you can judge empirically. Pooling only ever combines sessions that share screen size *and* capture protocol — a dot-grid session and a trajectory session are different enough distributions (static fixations vs. continuous lag-corrected pursuit) that mixing them risks silently training on nonsense.
- `--protocol {dots,trajectory}` — capture mechanic (default: `trajectory`).
- `--lag-ms` — `--protocol trajectory` only: how far back (ms) to shift each captured frame's target label to correct for smooth-pursuit lag (default: 220). If live predictions sit systematically off in one direction, try nudging this up or down (e.g. +-50ms) and recalibrating.
- `--poly-degree` — degree of the polynomial feature expansion `GazeCalibrator` fits. Defaults to `2` (quadratic) for `--protocol trajectory` and `1` (linear) for `--protocol dots` — pass it explicitly to override either. `--protocol trajectory`'s continuous capture produces far more independent calibration groups than the dot-grid's ~16, which is specifically the condition under which a `2` (quadratic) fit stops extrapolating wildly and can instead correct real curvature in the eye-rotation-to-screen mapping; the dot-grid protocol stays at `1` since it doesn't have that many independent points. See `src/doomscroll/gaze/calibration.py`'s `GazeCalibrator` docstring for the full reasoning.
- `--list-history` — list stored calibration sessions and exit.
- `--clear-history` — delete all stored calibration sessions and exit. Do this after any real setup change (new desk, new lighting, different chair height).

Every run is saved as its own timestamped session under your platform's app-data directory (`CALIBRATION_DIR/sessions/`, via `platformdirs`), independent of pooling.

## Testing

```
pytest
```

Runs the unit/synthetic test suite (`tests/`) — gaze vector math, head pose, smoothing, calibration fitting, session storage, and the pose-variant readout logic — all against synthetic, deterministic fixtures (no camera or MediaPipe model needed).

(pytest creates a `.pytest_cache/` directory to track state like last-failed tests between runs. It's local scratch state, not something to commit — pytest already drops its own `.gitignore` inside that directory, so it's self-ignoring regardless of this project's `.gitignore`.)

## Project layout

```
scripts/            Entry-point scripts (calibrate, debug view, main app)
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
