# CNN gaze model: what was added and changed

The CNN is an extra model. **Your existing models are not touched.** `models/gaze_model.joblib`,
`data/datasets/`, `build_dataset.py`, `train_model.py`, `ml/training.py`, `ml/dataset.py`, `ml/labels.py`
and your current per-user correction file all work exactly as before.

## How to run

```bash
pip install torch                                   # one-time; or: uv pip install -e ".[macos,cnn]"
python scripts/build_crops.py --preview             # slow, once: MediaPipe -> data/crops/*.npz
#   open data/crops/*_preview.png and check the eyes sit in the middle of each crop
python scripts/train_cnn.py                         # held-out test (CNN vs ridge vs hgb), then saves models/gaze_cnn.pt
python scripts/quick_calibrate.py --model models/gaze_cnn.pt      # optional ~15 s per-user nudge
python scripts/phase2_gaze_scroll.py --model models/gaze_cnn.pt   # run with the CNN

python scripts/phase2_gaze_scroll.py --model        # still runs your current model, unchanged
```

Useful options:

| Command | What it does |
|---|---|
| `build_crops.py --stride 2` | Process every 2nd frame: half the build time and file size (neighbouring frames are near-duplicates). |
| `build_crops.py --force` | Rebuild after changing `ml/eye_crops.py` or `ml/features.py`. |
| `train_cnn.py --skip-cv` | Skip the held-out test and only train the final model (much faster, use it while experimenting). |
| `train_cnn.py --epochs 15 --train-stride 2` | Shorter training. |
| `train_cnn.py --no-aux` | Pixels-only CNN (no landmark features). Compare it with the default to see what the images add. |
| `train_cnn.py --no-face` | Drop the face branch. |
| `train_cnn.py --lag-ms 200` | Set the label lag. Default: the lag saved in `models/gaze_model.joblib` (from `train_model.py --sweep-lag`), else 0. |
| `train_cnn.py --device cpu` | Force CPU (default picks `mps` on a Mac with Apple silicon). |

Time: the default run trains one network per held-out session (3 for your data) plus the final one.
Try `--skip-cv --epochs 3` first to check everything runs and see how long an epoch takes.
Memory: about 0.7 GB of crops for 80k frames, with a few copies made while training. If it's tight, use `--stride 2`.

## New files

| File | What it is |
|---|---|
| `src/doomscroll/ml/eye_crops.py` | What pixels the CNN sees. Each eye is cut along its two corner landmarks (head tilt removed), sized by eye width, 36x60 grayscale, histogram-equalized; plus a 64x64 face crop. Used for both training and live use. |
| `src/doomscroll/ml/crop_dataset.py` | Builds and loads one `.npz` per session in `data/crops/`: eye/face crops, the 11 landmark features, cursor log, timestamps. Crops are stored only for usable frames. Labels are made at training time with the existing `ml/labels.py`. |
| `src/doomscroll/ml/cnn.py` | The PyTorch model (shared eye conv tower x2 + face tower + 11 landmark features, then FC to an (x, y) screen fraction), augmentation, training loop, held-out cross-validation (same rules as `ml/training.py`), ridge/hgb baselines on the same frames, and `CnnGazeModel` for live use and save/load. |
| `scripts/build_crops.py` | Runs MediaPipe over `data/recordings/` and writes `data/crops/<session>.npz` (the CNN version of `build_dataset.py`). `--preview` saves a contact sheet of crops. |
| `scripts/train_cnn.py` | Trains the CNN, prints held-out error for cnn/ridge/hgb (plus error by head yaw), and saves `models/gaze_cnn.pt` and `models/gaze_cnn.report.json`. |
| `tests/test_cnn.py` | Tests with synthetic video (no webcam or MediaPipe): crops, crop dataset, CNN learning from pixels, save/load. Skipped automatically if torch isn't installed. |
| `docs/CNN_CHANGES.md` | This file. |

## Edited files (existing behaviour unchanged)

| File | Change |
|---|---|
| `src/doomscroll/ml/features.py` | Added a `FrameResult` dataclass and a `FrameFeaturizer.process_full()` method that also returns the landmarks, head pose and gray frame. `process()` now calls it and returns the same `(features, reason)` as before. `FEATURE_NAMES` and `extract_features` are unchanged, so existing datasets and models still load. |
| `src/doomscroll/ml/model.py` | Added `TrainedGazeModel.predict_frac_frame()` (same result as before, takes landmarks/pose/gray), `correction_path_for()` (`.pt` models use `affine_correction_cnn.json`, the current model keeps `affine_correction.json`), and `load_gaze_model()` (`.joblib` loads the current model, `.pt` loads the CNN). Nothing existing was changed. |
| `scripts/phase2_gaze_scroll.py` | `--model` now accepts a `.pt` file too. Uses `load_gaze_model()`; for a `.joblib` model the prediction is the same as before. Default path is still `models/gaze_model.joblib`. |
| `scripts/quick_calibrate.py` | New `--model PATH` option (default: `models/gaze_model.joblib`, as before). Saves the correction next to the right model (`affine_correction.json` or `affine_correction_cnn.json`). |
| `pyproject.toml` | New optional extra `cnn = ["torch>=2.2"]`. Normal installs don't need torch. |
| `docs/ML_PIPELINE.md` | Replaced the "CNN on eye crops" bullet under "Not done / next steps" with a section on the CNN pipeline. |

## Undo

Delete the new files above and `data/crops/`, `models/gaze_cnn.pt`. The edits to existing files are
additions only, so leaving them in place doesn't change how the current model behaves.

---

## Update (26 Oct, evening): `--cv person` (hold out whole people)

New option for **both** training scripts. The default is still `--cv session`, so earlier commands give the same numbers as before.

| `--cv` | Held out in each test fold | Answers |
|---|---|---|
| `session` (default) | one recording session | "Does it work on a new recording of someone it has already seen?" |
| `person` | every session of one participant | "Does it work on someone it has never seen?" (expect a higher error) |

```bash
uv run python scripts/train_cnn.py   --crops data/crops_v2 --cv person --epochs 10 --no-face --train-stride 2 --out models/gaze_cnn_v2_person.pt
uv run python scripts/train_model.py --cv person --model hgb --out models/gaze_model_hgb_v2_person.joblib
```

People are told apart by the `participant` field in each session's `meta.json` (stored in the `.npz` files when they are built).
Each person must use the same name in all their sessions; if you change a name, rebuild with `--force`.
Both scripts now also print an **error by participant** table next to the head-yaw table.
`--cv person` only changes the test; the final saved model is still trained on everyone.

Files changed:

| File | Change |
|---|---|
| `src/doomscroll/ml/dataset.py` | `LabeledFrames` has a `participant` field (default `""`), filled in by `SessionDataset.labeled()`. |
| `src/doomscroll/ml/training.py` | `pool(..., group_by="session" \| "person")`; `Pooled.participant`; new `error_by_group()`; `evaluate_candidates` adds `by_person` to the report. Errors clearly if fewer than 2 people. |
| `src/doomscroll/ml/cnn.py` | `Arrays.participant`; `pool_crops(..., group_by=...)` with the same rules; fold log shows `test = <person>`. |
| `scripts/train_model.py` | `--cv {session,person}`; prints error by participant. |
| `scripts/train_cnn.py` | `--cv {session,person}`; prints error by participant; the report JSON now also records which test was used (`"cv"` key). |
| `tests/test_recording_pipeline.py`, `tests/test_cnn.py` | Tests that no person is ever in both train and test, and that fewer than 2 people gives a clear error. |

---

## Update (27 Oct): content-aware scrolling (OCR) from feat-002

Brings in Ysabelle's OCR work (branch `feat-002`) and connects it to **your** `phase2_gaze_scroll.py`,
keeping `--model` (CNN `.pt` or `.joblib`). No retraining needed; the gaze model is unchanged.

```bash
uv sync --extra macos --extra cnn          # installs pyobjc-framework-Vision (Apple OCR)
uv run python scripts/phase2_gaze_scroll.py --model models/gaze_cnn_v2.pt
uv run python scripts/phase2_gaze_scroll.py --model models/gaze_cnn_v2.pt --content-only   # OCR rules only
```

Click into the window you want to read. A transparent overlay is drawn on top of it:
green box = line already read, red = not read yet, blue = last line on screen,
yellow line = "end of content" threshold, magenta dot = where you're looking.

Scroll rules (`fixation/scroll_trigger.py`):
1. gaze in the bottom part of the window (`BOTTOM_DWELL_FRACTION`, 0.80) for `DWELL_MS` (700) -> scroll
2. OCR says you've read the last visible line and moved past it, for `DWELL_MS` -> scroll
3. no unread line between your gaze and the window bottom, for `CONTENT_DWELL_MS` (1000) -> scroll

Files:

| File | Change |
|---|---|
| `src/doomscroll/content/` | From feat-002 (identical): Apple Vision OCR of the focused window on a background thread, and the "which lines have been read" tracker. |
| `src/doomscroll/fixation/scroll_trigger.py` | New, from feat-002 (unchanged): the 3-rule scroll decision. |
| `src/doomscroll/debug/screen_overlay.py` | New, from feat-002 (unchanged): transparent click-through overlay, hidden from OCR. |
| `src/doomscroll/windowing/macos_backend.py` | Replaced with feat-002's version: ignores the script's own windows, which fixes the preview-window focus bug. |
| `tests/test_gaze_content.py`, `test_scroll_trigger.py`, `test_content_macos_backend.py` | From feat-002. Note: `test_default_dwell_has_a_floor_even_for_a_near_zero_width_block` fails on feat-002 too (test expects a ~500 ms minimum, code has `SEEN_DWELL_MIN_MS = 150`). |
| `scripts/phase2_gaze_scroll.py` | Starts the OCR scanner, tracker, overlay and trigger; the old single dwell rule is replaced by `ScrollTrigger`; new `--content-only` flag; new `CONTENT_DWELL_MS` setting; your 0.80 / 700 ms settings are kept. **Difference from feat-002:** rules 2 and 3 only run when OCR has found at least one line (with zero lines, rule 3 is always true and would scroll every ~1.8 s on startup, blank pages or videos). |
| `pyproject.toml` | `pyobjc-framework-Vision` added to the `macos` extra. |
