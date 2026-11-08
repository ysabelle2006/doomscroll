# Gaze-tracking training data

The project's permanent, anonymized archive of data actually used to train
the two gaze models -- separate from `data/`, which is just the raw
per-session staging area a NEW recording passes through before being
extracted into here (see `data/README.md`). Nothing in this folder is
regenerable without the original raw recording, which may no longer exist
once it's been extracted and the raw copy cleaned up.

Unlike `data/` (raw face video, never committed), `crops/` and `features/`
here ARE committed to git on purpose: this is the baseline dataset anyone
cloning the repo gets for free, so "Record, train & calibrate" has
something real to retrain the CNN (and re-estimate its lag) against even
on a brand new clone, before they've recorded a single session of their
own. The only thing in this folder that stays local and out of git is
`.participant_map.local.json` -- see below.

## Layout

```
datasets/
    crops/<id>.npz      # CNN model's training data (scripts/build_crops.py's output)
    features/<id>.npz   # math model's training data (scripts/build_dataset.py's output)
```

`scripts/train_cnn.py` loads everything in `crops/`; `scripts/train_model.py`
loads everything in `features/` -- both via a plain directory listing, so a
new recording's two files just need to land in these same two folders
(which `scripts/build_crops.py` / `build_dataset.py` do by default) to be
included in the next training run alongside every session already here.
No merging step, no manual bookkeeping.

## Naming: `<id>.npz` = `<hashid>_<date>_<recordID>.npz`

- `hashid` -- a short, deterministic, one-way hash of the participant's
  name (`doomscroll.recording.session.participant_hashid()`), used
  instead of their name anywhere a session is referenced on disk. The
  same person's name always hashes to the same `hashid`, so their
  sessions stay grouped together, without their actual name appearing in
  any file or folder name, this README included.
- `date` -- `YYYYMMDD` the session was recorded.
- `recordID` -- a running count PER PARTICIPANT (per `hashid`), starting
  at `001`, assigned in recording order and never reused or reset (not
  reset daily, even if several sessions share a `date`). A brand new
  recording picks up this count where a participant's existing sessions
  (here, or still sitting unbuilt in `data/recordings/`) left off --
  see `doomscroll.recording.session.next_record_id()`.

This replaces the original on-disk naming, which was `<participant
name>_<YYYYMMDD>_<HHMMSS>` -- every session already here has been
renamed into the scheme above, and every new recording is assigned an id
in this scheme from the start (`scripts/record_session.py`).

## Re-identifying a session (your own data only)

If this project's own baseline sessions were just migrated into this
naming scheme, `datasets/.participant_map.local.json` records the
original name <-> `hashid` mapping and the original <-> new id for each
session. That file is for your own reference only: it is **not** tracked
by git (it's the one line still listed in `.gitignore` for this folder --
`crops/` and `features/` themselves are committed, see above) and should
never be shared or published alongside this folder -- doing so defeats the
entire point of renaming everything in the first place. There's no equivalent record for
sessions recorded after the migration, since `scripts/record_session.py`
never writes the real name to disk at all, anywhere, for those.

Note `hashid` is a deterministic hash, not strong anonymization: with
only a handful of real participants, anyone who suspects a specific name
can hash it themselves (same salt, in `doomscroll.recording.session`) and
check for a match. It keeps plain names out of filenames and paths
(this README, git history, a casual directory listing, anything copied
around) -- it does not make a targeted guess unrecoverable.
