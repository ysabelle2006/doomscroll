# Lab 5 Deliverables

Updated design documentation for Lab 5 (Project Demo and Submission), per the Lab 5 manual
section 3.2.2: an updated SRS (including Use Case Model), class diagrams of key classes, and a
UI state machine / dialog map -- all rewritten to match the current codebase rather than
re-publish Lab 1/Lab 2's content unchanged.

| File | What it is |
|---|---|
| [`srs.pdf`](srs.pdf) | Software Requirements Specification v2.0. Supersedes `lab1/requirements.docx`. Starts with a "Changes Since Lab 2" section listing every requirement that was corrected, and why -- several Lab 1/2 requirements described behaviour (configurable scroll direction/sensitivity/speed, a confidence-based auto-recalibration prompt, manual-scroll suppression) that the implementation never had. |
| [`use_case_model.pdf`](use_case_model.pdf) | Use Case Model v2.0. Supersedes `lab1/use_case_model.docx` and `lab2/use_case_descriptions.pdf`. 13 use cases grouped by Camera Access, Calibration, Gaze Tracking, the Content-Aware Scroll Trigger, Scrolling/Session Control, and Application Control. |
| [`class_diagram.jpg`](class_diagram.jpg) | Key classes and their relationships across the launcher, the gaze-estimation pipeline (Math and CNN models), the content-aware scroll trigger, and the recording/training pipeline. Supersedes `lab2/class_diagram.jpg`, which modelled a boundary/control/entity architecture (`AutoscrollController`, `ImportantContentRegion`, etc.) the codebase does not actually have. |
| [`dialog_map.jpg`](dialog_map.jpg) | The real UI flow: the Tk launcher's Main/Settings pages, the OS-level windows each `scripts/*.py` subprocess opens (Calibration, Recording, Quick Calibrate, the live Running/Autoscroll window), and every dialog between them. Supersedes `lab2/dialog_map.jpg`, which modelled screens (`ReaderOverlay`, a `RecalibrationPromptDialog`) that do not exist. |

## Why these differ from Lab 1 / Lab 2

Labs 1-2 described the system around a binary "Important Content" / "Unimportant Content"
classification, user-configurable scroll direction/sensitivity/speed, a keyboard shortcut to
pause/resume, and an automatic recalibration prompt triggered by a sustained low confidence
score. None of that was ever implemented. The real system instead:

- Decides whether to scroll with a 3-tier trigger (bottom-of-window / reached-end-of-text /
  nothing-unseen-ahead), content-aware only on macOS via Apple Vision OCR -- Windows always runs
  tier-1-only.
- Only ever scrolls downward, by a fixed step; there is no direction, sensitivity, or speed
  setting anywhere.
- Pauses automatically (not via a shortcut) through an Attention Gate when the user looks away,
  and resumes automatically when they look back. Ending a session is done by closing its window.
- Has no live confidence score and no automatic recalibration prompt; recalibration is always a
  deliberate action from the Main Page, with an immediate "run / redo / cancel" checkpoint after
  every calibration run.
- Ships two independent, interchangeable gaze-estimation pipelines (a per-session Math Model and
  an offline-trained CNN Model with its own recording/training flow), which Labs 1-2 never
  distinguished.

`srs.pdf`'s "Changes Since Lab 2" section and each affected use case's "Notes and Issues" field
give the full, requirement-by-requirement account.
