# doomscroll

Gaze/head-tracking content-aware auto-scroll. Tracks where you're looking via webcam, maps that to screen coordinates with a calibrated model, and (eventually) uses it to drive content-aware scroll behavior.

## Setup

Requires Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.13"` in `pyproject.toml`).

```
uv sync
```

(or `pip install -e .` if you're not using `uv`). On Windows, also install the `windows` extra (`pywin32`); on macOS, the `macos` extra (`pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa`, `pyobjc-framework-Vision`) — these back the OS-specific scroll/windowing backends and the Vision-based OCR content scanner.

The MediaPipe face landmark model is already checked into `models/face_landmarker.task` — no separate download needed.

To use the L2CS-Net gaze feature backend (`--features l2cs`, see Gaze feature backends below), also install the `l2cs` extra:

```
uv sync --extra l2cs
```

(or `pip install .[l2cs]`). This pulls in `torch`, `torchvision`, and `safetensors`. The default `--features handcrafted` path needs none of this — the `l2cs` extra and its heavy imports are entirely optional and only touched when you explicitly ask for that backend.

Two more pieces `--features l2cs` needs are deliberately *not* checked into git (both gitignored — a ~96MB weights file and a full vendored third-party repo don't belong in this repo's history):

- **Model weights** — a Gaze360-pretrained L2CS-Net (ResNet50) checkpoint, as a safetensors file at `models/l2cs_gaze360_resnet50.safetensors`.
- **The vendored L2CS-Net repo** — clone it into place with:
  ```
  git clone https://github.com/Ahmednull/L2CS-Net ml/third_party/L2CS-Net
  ```
  (`src/doomscroll/gaze/l2cs_backend.py` imports from it directly via `sys.path`, not as an installed package.)

## Running it

1. **`scripts/debug_view.py`** — Phase 1 sanity check: webcam capture + face-mesh landmarks + head pose, rendered live. Run this first to confirm your camera and landmark detection are working before anything else.
2. **`scripts/calibrate.py`** — run this before the main app (see Calibration below).
3. **`scripts/phase2_gaze_scroll.py`** — the end-to-end loop: calibrated gaze mapping + attention gate + a trivial dwell-to-scroll rule. Press `q` to quit.

## Calibration

```
python scripts/calibrate.py
```

The capture protocol depends on `--features` (see Gaze feature backends below) — this section describes the default, `--features handcrafted`. Two passes:

- **Main pass** — a 16-point on-screen grid. Keep your head still and forward, track each dot with your eyes only.
- **Head-pose variation pass** — immediately after, you'll be asked to hold a few different natural head positions (lean left/right, lean closer/sit back, tilt chin up/down) while repeating a short dot sequence at each. This teaches the model to correct for incidental head movement instead of only ever having seen you sit still. Each pose shows a live readout and waits for you to press SPACE once you're actually holding the position (green = roughly there; it's a visual guide only — SPACE always advances regardless of color, since everyone's comfortable "slight lean" differs). Press ESC at any time to abort.

A validation pass follows, reporting mean pixel error.

`--features l2cs` uses a different capture mechanic entirely — a continuous cursor-trace instead of discrete dots — covered in Gaze feature backends below, not here.

**Flags:**

- `--pool` — also fits against this session pooled with your most recent stored sessions, and keeps whichever fit (session-only or pooled) actually validates better. Pooling is opt-in, not default, because your physical setup (seating, lighting, camera position) drifts between sittings, and pooling across a setup change can make the fit worse, not better — see `src/doomscroll/gaze/session_store.py`'s docstring for the full reasoning. `--pool` prints both validation numbers so you can judge empirically.
- `--list-history` — list stored calibration sessions and exit.
- `--clear-history` — delete all stored calibration sessions and exit. Do this after any real setup change (new desk, new lighting, different chair height).

Every run is saved as its own timestamped session under your platform's app-data directory (`CALIBRATION_DIR/sessions/`, via `platformdirs`), independent of pooling.

## Gaze feature backends

Both `scripts/calibrate.py` and `scripts/phase2_gaze_scroll.py` accept a `--features {handcrafted,l2cs}` flag (default `handcrafted`) selecting which per-frame gaze signal feeds the calibrated Ridge model:

- **`handcrafted`** (default) — the original 4-dim eye-in-head vector derived from MediaPipe's eye/iris landmarks, concatenated with 6-dim head pose/translation (10-dim total). Needs nothing beyond the base install.
- **`l2cs`** — a pretrained L2CS-Net (ResNet50, Gaze360-trained) appearance-based estimate (2-dim yaw/pitch) in place of the hand-crafted eye vector, concatenated with the same 6-dim head pose/translation (8-dim total). Needs the `l2cs` extra (see Setup above) and the vendored model weights, and prefers Apple Silicon's `mps` device over CPU when available. In offline comparisons across recorded sessions, this gave a sizeable accuracy improvement on higher-resolution screens and was roughly a wash on lower-resolution ones — judge it empirically on your own setup rather than assuming it's strictly better.

The two backends also use different calibration *capture protocols* in `scripts/calibrate.py`, not just a different feature vector: `--features handcrafted` uses the discrete 16-point dot-grid described in Calibration above (`run_sequence`/`collect_samples_for_point`); `--features l2cs` uses a continuous cursor-trace instead — a single zigzag line you trace with your own mouse cursor while looking at it, sampled continuously rather than at fixed points (`run_continuous_trace`). Both share the same readiness screens and head-pose variation pass, and both save/validate/report through the same tail end of `calibrate.py` — see that file's module docstring for the full mechanics and the pursuit-lag correction the trace protocol applies.

A calibration profile and any stored calibration session are tagged with the backend that produced them (`feature_backend`, defaulting to `handcrafted` for anything saved before this flag existed). `--pool` and history pooling only ever combine sessions that share both screen size and feature backend — mixing backends would either crash on a shape mismatch or, worse, silently train on nonsense.

**The two backends are fully interchangeable, with separate saved profiles.** `calibrate.py` saves to a per-backend path (`CALIBRATION_DIR/default_handcrafted.joblib` / `default_l2cs.joblib`, via `calibration.py`'s `profile_path_for_backend()`), not a single shared file — calibrating with one `--features` value never overwrites the other's saved profile. `phase2_gaze_scroll.py` loads through the matching `load_profile_for_backend()`, which automatically picks up whichever backend you pass it via `--features` — so you can run `--features handcrafted` today and `--features l2cs` tomorrow without recalibrating the one you're switching back to, as long as you've calibrated each backend at least once. (If you calibrated before this per-backend split existed, your one existing profile is picked up automatically as a fallback for whichever backend it's tagged with — but the *other* backend will need a first calibration run, since there was never more than one saved profile to begin with under the old shared-path scheme.)

All of this backend selection lives in one place in each script (`_init_gaze_backend()` in `phase2_gaze_scroll.py`, the "Protocol dispatch" section in `calibrate.py`) — the rest of each file's logic (the capture loop, `build_feature_vector()`, `calibrator.predict()`) is backend-agnostic and doesn't know or care which one is active.

**Live per-frame inference performance for `--features l2cs` has not been verified against a real camera** — it needs checking on an actual machine with the extra installed; only the pure/non-torch pieces (face cropping, device selection logic) are covered by the automated tests.

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
