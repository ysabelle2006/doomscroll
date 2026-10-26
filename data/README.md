# Gaze-tracking ML data

Recorded sessions for the calibration-free gaze-tracking experiments (see
`ml/` at the project root). Each session is one person doing one ~5-minute
recording while a cursor moves around the screen and they follow it with
their eyes -- the cursor's logged position is the ground-truth gaze target.

## Layout

```
data/raw/<participant>/<session_id>/
    meta.json              # session metadata (see below)
    frames.csv             # raw per-frame cursor log, synced to the webcam recording
    <session_id>.mp4       # webcam recording (not yet present for most sessions below)
    <session_id>.npz       # optional: pre-extracted feature vectors (see below)
```

Not every session folder currently has every file -- the two samples
checked in so far (`ysabelle_20261026_135832` and `ysabelle_20261026_132332`)
are from two *different* recordings, included just to document the format:
the first has `meta.json` + `frames.csv` (no video or extracted features
yet), the second has only the pre-extracted `.npz` (no meta.json, csv, or
video for that one). A complete session folder should eventually have all
four files once the full dataset is downloaded from Google Drive -- same
`data/raw/<participant>/<session_id>/` convention, so whatever extraction
tooling follows doesn't need to change.

Across the full dataset (4 participants so far, ~1 hour combined), expect
multiple sessions per participant, each recorded with a different head
position/distance relative to the screen (see `meta.json`'s `notes` field,
e.g. `"headright"`) -- that positional diversity is what the models in
`ml/` are meant to learn to be robust to, so don't discard or merge it away
when splitting data for training/evaluation. A fair test of "does this
generalize across position shifts" means holding out whole sessions (by
`notes`) or whole participants from training, not a random frame-level
split.

## `meta.json`

```json
{
  "session_id": "ysabelle_20261026_135832",
  "participant": "ysabelle",
  "screen_w": 3200, "screen_h": 2000,       // participant's screen resolution, px
  "camera_index": 0,
  "frame_w": 1920, "frame_h": 1080,          // webcam capture resolution
  "measured_fps": 29.63,
  "n_frames": 8890,
  "duration_s": 299.99,
  "started_at": "2026-10-26T13:58:39",
  "has_screen_video": false,
  "notes": "headright",                      // free-text head-position/distance label for this session
  "activity": "free",                         // cursor-following protocol used
  "schema_version": 2,
  "extra": {}
}
```

## `frames.csv`

One row per webcam frame, synced by `frame_idx`/`t_ms` to the (eventual)
`.mp4`:

```
frame_idx,t_ms,cursor_x,cursor_y,mouse_down,active
0,2.125,3084.0,352.0,0,1
```

`cursor_x`/`cursor_y` are the ground-truth gaze target, in the same
screen-pixel space as `meta.json`'s `screen_w`/`screen_h`.

## `<session_id>.npz`

Pre-extracted features, produced by running this project's own
`gaze/gaze_vector.py` + `gaze/head_pose.py` extraction over the webcam
recording (frame by frame) and pairing each frame's result with that
frame's cursor position from `frames.csv`. Keys:

- `X` `(N, 11)` float64 -- columns given by `feature_names`, in order:
  `gaze_mean_x, gaze_mean_y, gaze_diff_x, gaze_diff_y` (the 4-dim eye-in-head
  vector from `GazeFeature.vector`), `lid_ratio_y` (eye-aspect-ratio-like
  blink signal -- NOT part of the deployed `build_feature_vector()` output,
  used only for blink filtering), `yaw, pitch, roll` (head rotation,
  degrees), `head_tx, head_ty, head_tz` (head translation, solvePnP units).
  Rows where `reasons == "blink"` have `NaN` here -- drop them, and drop
  the `lid_ratio_y` column too, to match the production 10-dim feature set
  built by `src/doomscroll/gaze/calibration.py::build_feature_vector()`
  exactly.
- `cursor_xy` `(N, 2)` float64 -- ground truth, screen pixels.
- `t_ms` `(N,)` float64 -- chronological; use for leakage-safe time-based
  splits (consecutive frames are highly correlated, so a random split
  overstates how well a model actually generalizes).
- `mouse_down`, `active` `(N,)` int -- as in `frames.csv`.
- `reasons` `(N,)` str -- `"ok"` or `"blink"` per frame.
- `feature_names` `(11,)` str -- the `X` column order above.
- `info` -- JSON string: `{"session_id", "participant", "screen_wh"}`.

An earlier one-off script loaded sessions in this format to fit the
production `GazeCalibrator` against a couple of alternative regressors and
compare held-out pixel error -- that comparison tooling isn't currently
checked in, but the `.npz` format itself is unchanged and still the
expected input for anything that replaces it.
