"""Tkinter launcher for doomscroll.

Gives two ways to get the gaze-scroll app running:

Quick Start calibrates and runs the math model (GazeCalibrator) -- the
fast path, no training involved. You choose which calibration protocol to
use (trajectory, the default, or dots -- see scripts/calibrate.py and
README.md's Calibration section for what each one actually does).

Full Start uses the CNN pipeline (see docs/ML_PIPELINE.md). "Record,
train & calibrate" records a long video of you looking at the moving
cursor, folds it (and any existing data/raw/<person> recordings) into the
training set, retrains the CNN, then runs scripts/quick_calibrate.py's
own 9-dot calibration (~15s) and launches. "Calibrate only" skips
straight to that same quick calibration against whatever CNN is already
trained -- faster, but since it only fits a small per-user correction on
top of that existing model instead of retraining it on your own data,
accuracy is usually worse. This calibration is unrelated to the math
model's dots/trajectory protocol choice above -- the CNN is always
calibrated via its own fixed 9-dot grid.

After every calibration step, in every flow, you're asked whether to
launch now or redo the calibration -- so a poor calibration (tracking
felt off during the sweep, or the printed validation error looked high)
can be retried as many times as you like before committing to a run.

Every step runs the real scripts/*.py as a subprocess (they open their own
fullscreen camera/OpenCV windows, so they can't run inside this window's
own event loop); the Settings page (reached via the Settings button) shows
their combined output in a log, and holds "Clear all calibrations".
"""
from __future__ import annotations

import importlib.util
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from platformdirs import user_data_dir

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
RECORDINGS_DIR = ROOT / "data" / "recordings"

# scripts/record_session.py prints "Screen WxH. Session: <id>" once it's
# assigned the new session its own anonymized id -- this is the only way
# app.py learns that id, since make_session_id() runs inside that
# subprocess, not here.
_SESSION_LINE_RE = re.compile(r"Session:\s*(\S+)")


def _extract_session_id(lines: list[str]) -> str | None:
    for line in lines:
        m = _SESSION_LINE_RE.search(line)
        if m:
            return m.group(1)
    return None
PYTHON = sys.executable

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CNN_MODEL_PATH = ROOT / "models" / "gaze_cnn.pt"

# Mirrors doomscroll.gaze.calibration.DEFAULT_PROFILE_PATH and
# doomscroll.ml.model.correction_path_for(CNN_MODEL_PATH), duplicated here
# (rather than imported) so this launcher never has to import the
# doomscroll package itself -- it only ever shells out to scripts/*.py.
# Each pipeline's saved calibration lives under a different app-data name
# ("gazescroll" vs "doomscroll") -- that's not a typo here, it's inherited
# as-is from those two modules.
MATH_PROFILE_PATH = Path(user_data_dir("gazescroll")) / "calibration" / "default.joblib"
CNN_CORRECTION_PATH = Path(user_data_dir("doomscroll")) / "calibration" / "affine_correction_cnn.json"
# Mirrors doomscroll.gaze.session_store.SESSIONS_DIR -- the stored raw
# per-session samples --pool fits against.
SESSIONS_DIR = MATH_PROFILE_PATH.parent / "sessions"

CLEAR_CALIBRATIONS_HELP = (
    "Deletes the saved math calibration profile, the saved CNN per-user correction, and all "
    "stored calibration sessions (used by --pool on the Quick Start box). Both pipelines will "
    "need a full recalibration before they'll run again, and --pool will have nothing to pool "
    "against until new sessions are captured."
)

# This app's own explicit light/dark palette -- applied via _apply_theme()
# rather than inherited from the OS/ttk theme engine, so the colors below
# are exactly what each mode looks like regardless of platform. "border"
# is the content boxes' outline color (see _boxed_frame()); "muted" and
# "warn" back the two named label styles used for dimmed explanatory text
# and warnings (Muted.TLabel / Warn.TLabel).
THEMES: dict[str, dict[str, str]] = {
    "light": {"bg": "#ffffff", "fg": "#000000", "border": "#000000", "muted": "#555555", "warn": "#aa3333"},
    "dark": {"bg": "#2b2b2b", "fg": "#ffffff", "border": "#ffffff", "muted": "#aaaaaa", "warn": "#ff6b6b"},
}


def _detect_system_dark_mode() -> bool:
    """Best-effort read of the OS-level light/dark preference, used only
    to pick this app's own default theme the first time it runs (the
    Settings page's Light/Dark choice is what actually governs the
    styling afterwards -- see _apply_theme() -- this is never consulted
    again once a theme is selected there)."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            return result.returncode == 0 and "dark" in result.stdout.strip().lower()
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0
    except Exception:
        pass
    return False  # Linux/unknown: no reliable cross-desktop signal, default to light


def legacy_person_dirs() -> list[Path]:
    """Existing data/raw/<person>/<session> recordings predating
    scripts/record_session.py's data/recordings/ layout -- folded into
    training by running scripts/build_crops.py once per person folder."""
    raw = ROOT / "data" / "raw"
    if not raw.is_dir():
        return []
    return sorted(p for p in raw.iterdir() if p.is_dir())


def _bring_window_to_front(root: tk.Tk) -> None:
    """Forces `root` to the front and focused. Needed right before any
    dialog shown right after a subprocess's own fullscreen OpenCV window
    closes (every calibration/run step) -- macOS doesn't reliably hand
    focus back to this app once that window is gone, so a dialog shown
    at that moment can open behind or unfocused without the window
    manager ever visibly announcing it. That's indistinguishable from
    the whole app being stuck: the dialog IS open and has grabbed all
    input app-wide (so clicking a button genuinely does nothing), it's
    just not something you can see to dismiss.

    Deliberately does NOT set -topmost on root: that was tried first,
    and it overcorrected -- it made root itself outrank the very dialog
    this is meant to help surface, since the dialog (a plain transient
    child of root, not itself marked topmost) doesn't inherit root's
    forced window level. Net effect: the dialog ended up stacked BEHIND
    this app's own forced-topmost main window -- still ahead of every
    OTHER app, just not ahead of root. lift()/focus_force() alone are
    enough to bring the whole app forward of other apps; update() forces
    Tk to actually act on that right now, synchronously, rather than
    leaving it queued to happen "eventually" -- without it, a dialog
    opened on the very next line can still end up shown before the
    window manager has actually finished the raise."""
    root.deiconify()
    root.lift()
    root.focus_force()
    root.update()


class CommandRunner:
    """Runs a worker function on a background thread (since the scripts
    it launches block for minutes and open their own OpenCV windows, which
    can't happen on the GUI thread) and streams subprocess output into a
    queue for the GUI thread to display.

    A worker is a callable(runner) -> bool that calls runner.run_cmd(...)
    for each subprocess step and, optionally, runner.ask_calibration_checkpoint()
    between a calibration step and whatever launches after it -- letting a
    flow redo calibration before committing to a run without re-running
    everything before it. Tkinter calls aren't thread-safe, so neither of
    those -- nor on_done once a step finishes -- ever touches the GUI
    directly from this (background) thread. Each instead puts a
    (callable, args) pair on action_queue, a plain queue.Queue (which IS
    genuinely thread-safe, unlike calling a Tk method directly), for
    App._poll_log() to pick up and actually run on the main/GUI thread,
    the same way it already drains log_queue every 100ms. This was
    originally done via root.after(0, ...) called straight from this
    thread instead -- which looks like it should work (and the call
    itself doesn't raise), but isn't actually guaranteed safe, and in
    practice turned out to silently never fire on at least one real
    setup: on_done (and so re-enabling the launcher's buttons) simply
    never happened after a failed step, with nothing in the log or a
    traceback to explain why. ask_calibration_checkpoint() still blocks
    this thread on a threading.Event until the GUI thread responds."""

    def __init__(self, root: tk.Tk, log_queue: "queue.Queue[str]") -> None:
        self.root = root
        self.log_queue = log_queue
        self.action_queue: "queue.Queue[tuple[Callable[..., None], tuple]]" = queue.Queue()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, worker: Callable[["CommandRunner"], bool], on_done: Callable[[bool], None]) -> None:
        if self.running:
            return

        def _run() -> None:
            try:
                success = worker(self)
            except Exception as exc:  # surfaced in the log instead of crashing silently
                self.log_queue.put(f"\nUnexpected error: {exc}\n")
                success = False
            self.action_queue.put((on_done, (success,)))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def run_cmd(self, label: str, cmd: list[str]) -> bool:
        """Runs one subprocess step to completion, streaming its combined
        stdout/stderr into the log. Returns True on exit code 0."""
        ok, _lines = self.run_cmd_capture(label, cmd)
        return ok

    def run_cmd_capture(self, label: str, cmd: list[str]) -> tuple[bool, list[str]]:
        """Same as run_cmd, but also returns every line the subprocess
        printed, so a caller can parse something out of it (e.g. the
        session id record_session.py assigns and prints, which app.py has
        no other way to learn). Returns (True on exit code 0, printed lines)."""
        self.log_queue.put(f"\n=== {label} ===\n$ {' '.join(cmd)}\n")
        lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except OSError as exc:
            self.log_queue.put(f"Failed to start: {exc}\n")
            return False, lines
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(line)
            lines.append(line)
        code = proc.wait()
        if code != 0:
            self.log_queue.put(f"\n'{label}' exited with code {code}.\n")
            return False, lines
        return True, lines

    def ask_calibration_checkpoint(self) -> str:
        """Blocks the worker thread right after a calibration step
        finished, and asks on the GUI thread whether to launch now or redo
        the calibration -- for when tracking felt off during the sweep, or
        the validation error just printed to the log looked too high.
        Returns 'run', 'retry', or 'cancel'."""
        result: dict[str, str] = {}
        done = threading.Event()

        def ask() -> None:
            _bring_window_to_front(self.root)
            choice = messagebox.askyesnocancel(
                "doomscroll launcher",
                "Calibration finished -- check the validation error printed above in the log.\n\n"
                "Yes = launch now with this calibration.\n"
                "No = redo the calibration.\n"
                "Cancel = stop here without launching.",
            )
            if choice is True:
                result["choice"] = "run"
            elif choice is False:
                result["choice"] = "retry"
            else:
                result["choice"] = "cancel"
            done.set()

        self.action_queue.put((ask, ()))
        done.wait()
        return result["choice"]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("doomscroll launcher")
        root.geometry("800x700")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.runner = CommandRunner(root, self.log_queue)
        self.buttons: list[ttk.Button] = []
        # Raw tk widgets (not ttk) that can't be recolored via ttk.Style --
        # _apply_theme() walks these directly whenever the theme changes.
        self.boxed_frames: list[tk.Frame] = []
        self.help_badges: list[tk.Canvas] = []
        self.checkboxes: list[Callable[[], None]] = []

        self._setup_styles()
        self._build_layout()
        self._apply_theme()  # log_text didn't exist yet during _setup_styles's call
        self._poll_log()
        self._show_main_page()

    def _setup_styles(self) -> None:
        """Shared look-and-feel bits computed once: header_font (box
        headers use this, larger+bolder than the plain body text inside
        the box), and theme_var -- this app's own Light/Dark choice
        (Settings page, see _build_settings_page()), defaulting to
        whichever one the OS itself is set to. Applying it (colors,
        including bg_color -- the current theme's background, used so
        the circular help badge's canvas doesn't show up as a mismatched
        square -- see _make_help_badge()) happens in _apply_theme(),
        which this also calls once up front."""
        default_font = tkfont.nametofont("TkDefaultFont")
        self.header_font = (default_font.actual("family"), 14, "bold")
        self.theme_var = tk.StringVar(value="dark" if _detect_system_dark_mode() else "light")
        self._theme_engine_set = False
        # Selecting a radio button normally also gives it keyboard focus,
        # which every ttk theme then outlines with a dotted/dashed focus
        # ring around the label -- the "box" that appears on selection.
        # Radiobutton.takefocus=0 via the option database (set once,
        # before any radiobuttons exist, rather than repeating it at
        # every ttk.Radiobutton(...) call site) stops them from ever
        # taking focus in the first place, so that ring never appears.
        self.root.option_add("*TRadiobutton.takeFocus", 0)
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Applies the current Light/Dark choice (self.theme_var) to
        every widget. ttk widgets go through named styles (so toggling
        later just reconfigures those styles and every widget using them
        repaints); the few plain tk widgets that can't be styled that way
        -- the black/white-bordered boxes (_boxed_frame()), the circular
        help badges' canvas background (_make_help_badge()), and the Log
        text box -- are walked directly via the lists those build up.

        Switches the ttk theme engine to "clam" the first time this runs:
        platform-native themes (e.g. macOS's "aqua") render most widgets
        natively and silently ignore configured colors, which would make
        an explicit Light/Dark choice unreliable; "clam" is a plain
        Tcl-drawn theme that honors them on every platform. Only done
        once -- repeatedly switching theme engines on every toggle isn't
        needed and risks losing the style customizations below."""
        theme = THEMES[self.theme_var.get()]
        self.bg_color = theme["bg"]
        self.border_color = theme["border"]
        self.theme_fg = theme["fg"]

        style = ttk.Style(self.root)
        if not self._theme_engine_set:
            style.theme_use("clam")
            self._theme_engine_set = True

        style.configure(".", background=theme["bg"], foreground=theme["fg"])
        for name in ("TFrame", "TLabel", "TButton", "TRadiobutton"):
            style.configure(name, background=theme["bg"], foreground=theme["fg"])
        style.configure("TEntry", fieldbackground=theme["bg"], foreground=theme["fg"])

        # clam's default TButton has a state-colored "active"/"pressed"
        # background -- the hover-highlight effect -- plus a heavier
        # bevel border than the rest of this app's thin, flat styling.
        # style.map(..., background=[]) clears clam's built-in hover/
        # press color changes entirely, so a button looks identical
        # whether or not the mouse is over it (foreground keeps only the
        # "disabled" override, so disabled buttons -- e.g. while a step
        # is running -- are still visually distinguishable).
        #
        # relief MUST be "solid" here, not "flat" -- Tk's border drawing
        # treats "flat" as "no border at all" and skips the border
        # element entirely regardless of bordercolor/width, which is why
        # it looked like buttons had no border whatsoever. "solid" is
        # the one relief that draws a plain, unshaded, single-color
        # outline (no 3D bevel). Tk borderwidth is always a whole number
        # of pixels -- 1 is already the thinnest a border can actually
        # render, so a lighter, mid-gray line (theme["muted"], the same
        # dimmed tone used for Muted.TLabel -- already tuned to read as
        # a soft gray against each theme's own background) is used here
        # instead of the full-strength black/white border color, to make
        # it *look* lighter without risking it rounding away to nothing.
        style.configure(
            "TButton",
            borderwidth=1,
            relief="solid",
            bordercolor=theme["muted"],
            lightcolor=theme["muted"],
            darkcolor=theme["muted"],
        )
        style.map("TButton", background=[], foreground=[("disabled", theme["muted"])])
        # Same deal as TButton above: clam recolors a Radiobutton's
        # background on hover by default. Clearing the state map here
        # (background/foreground are the two options every ttk style is
        # guaranteed to support, unlike e.g. indicator-specific options
        # that vary by theme and risk a TclError if the name is wrong)
        # means hovering changes nothing visually.
        style.map("TRadiobutton", background=[], foreground=[])
        style.configure("Header.TLabel", background=theme["bg"], foreground=theme["fg"], font=self.header_font)
        style.configure("Muted.TLabel", background=theme["bg"], foreground=theme["muted"])
        style.configure("Warn.TLabel", background=theme["bg"], foreground=theme["warn"])

        self.root.configure(bg=theme["bg"])
        for frame in self.boxed_frames:
            frame.configure(
                bg=theme["bg"], highlightbackground=theme["border"], highlightcolor=theme["border"]
            )
        for badge in self.help_badges:
            badge.configure(bg=theme["bg"])
        for redraw in self.checkboxes:
            redraw()
        if hasattr(self, "log_text"):
            self.log_text.configure(bg=theme["bg"], fg=theme["fg"], insertbackground=theme["fg"])

    def _section_header(self, parent: tk.Widget, text: str) -> ttk.Label:
        """Plain bold, larger-than-body text sitting directly on the
        page, NOT a LabelFrame's own title/legend -- that always renders
        as its own little box (its own background, its own alignment
        rules) that doesn't reliably line up with the content box below
        it. A plain Label here is a normal sibling widget, so it aligns
        with that box trivially (same padx) and inherits the page's own
        background, not a themed border's."""
        label = ttk.Label(parent, text=text, font=self.header_font, style="Header.TLabel")
        label.pack(anchor="w", padx=10, pady=(10, 2))
        return label

    def _boxed_frame(self, parent: tk.Widget) -> tk.Frame:
        """A plain rectangular box for a section's content, with a solid
        border in the current theme's border color (black in Light mode,
        white in Dark -- see THEMES/_apply_theme()), no title of its own
        -- the heading above it (_section_header()) is a separate sibling
        widget. A ttk.Frame's relief/borderwidth only draws a themed
        bevel in the theme's own colors, never an arbitrary color, so
        this uses a plain tk.Frame's highlight border instead:
        highlightthickness draws a flat, solid-color rectangle around the
        widget with no bevel shading. bg matches the current theme's
        background so the frame itself stays invisible against the page
        and only the edge shows. Registered in self.boxed_frames so
        _apply_theme() can recolor it if the user switches modes later."""
        frame = tk.Frame(
            parent,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.border_color,
            bg=self.bg_color,
        )
        self.boxed_frames.append(frame)
        return frame

    def _make_help_badge(self, parent: tk.Widget, command: Callable[[], None]) -> tk.Canvas:
        """A small circular '?' badge -- plain tk.Canvas, since a themed
        ttk.Button always renders as a rectangle and can't be made round.
        bg matches the surrounding frame so only the circle itself (not
        the canvas's own square bounding box) is visible."""
        size = 22
        canvas = tk.Canvas(parent, width=size, height=size, highlightthickness=0, bd=0, bg=self.bg_color)
        canvas.create_oval(1, 1, size - 1, size - 1, fill="#555", outline="#333")
        canvas.create_text(
            size // 2, size // 2, text="?", font=(self.header_font[0], 10, "bold"), fill="white"
        )
        canvas.bind("<Button-1>", lambda _event: command())
        self.help_badges.append(canvas)
        return canvas

    def _make_checkbox(self, parent: tk.Widget, text: str, variable: tk.BooleanVar) -> ttk.Frame:
        """A checkbox drawn entirely by hand -- a small tk.Canvas square
        with a hand-drawn checkmark tick when set -- rather than a
        themed ttk.Checkbutton. clam's own checkbox indicator draws an
        X instead of a tick, and splicing in another theme's indicator
        element for just that one shape turned out to be too fragile to
        rely on blind (it errored as a duplicate element, and once
        worked around that, rendered as no indicator at all -- both only
        discoverable by actually running the app, which isn't possible
        from this bridged session). Drawing the box and tick directly
        means exactly what appears is fully under this app's control,
        with no theme internals to fight. Registered in self.checkboxes
        so _apply_theme() can redraw it in the new colors on a
        Light/Dark switch."""
        size = 16
        row = ttk.Frame(parent)
        canvas = tk.Canvas(row, width=size, height=size, highlightthickness=0, bd=0, bg=self.bg_color)
        canvas.pack(side="left", padx=(0, 6))
        label = ttk.Label(row, text=text)
        label.pack(side="left")

        def redraw() -> None:
            canvas.configure(bg=self.bg_color)
            canvas.delete("all")
            canvas.create_rectangle(1, 1, size - 1, size - 1, outline=self.theme_fg, width=1)
            if variable.get():
                canvas.create_line(3, size // 2, size // 2 - 1, size - 4, width=2, fill=self.theme_fg)
                canvas.create_line(size // 2 - 1, size - 4, size - 3, 3, width=2, fill=self.theme_fg)

        def toggle(_event: "tk.Event | None" = None) -> None:
            variable.set(not variable.get())
            redraw()

        canvas.bind("<Button-1>", toggle)
        label.bind("<Button-1>", toggle)
        redraw()
        self.checkboxes.append(redraw)
        return row

    def _build_layout(self) -> None:
        # Two top-level pages sharing this window: the main page (Quick
        # Start / Full Start) and the settings page (Log + Clear all
        # calibrations) -- only one is ever packed into root at a time;
        # see _show_main_page()/_show_settings_page().
        self.main_page = ttk.Frame(self.root)
        self.settings_page = ttk.Frame(self.root)
        self._build_main_page(self.main_page)
        self._build_settings_page(self.settings_page)

    def _build_main_page(self, parent: ttk.Frame) -> None:
        self._section_header(parent, "Quick Start")
        quick = self._boxed_frame(parent)
        quick.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(
            quick,
            text="Calibrate and run with the math model (GazeCalibrator). Fast, no training needed.",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 2))

        protocol_frame = ttk.Frame(quick)
        protocol_frame.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(protocol_frame, text="Calibration type:").pack(side="left")
        self.quick_protocol_var = tk.StringVar(value="trajectory")
        ttk.Radiobutton(
            protocol_frame, text="Trajectory (default, faster)", variable=self.quick_protocol_var, value="trajectory"
        ).pack(side="left", padx=(8, 12))
        ttk.Radiobutton(
            protocol_frame, text="Dots (more accurate, slower, more retries)", variable=self.quick_protocol_var,
            value="dots",
        ).pack(side="left")
        ttk.Label(
            quick,
            text="Dots asks you to hold your head still and fixate 16 points one at a time, which "
            "tends to be more accurate than the trajectory sweep but takes longer overall and "
            "more often needs a point redone (if you blink or glance away mid-point). Trajectory "
            "follows one continuously-moving dot instead -- quicker, and bad frames are simply "
            "discarded rather than requiring a redo.",
            wraplength=740,
            justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.pool_var = tk.BooleanVar(value=False)
        self._make_checkbox(quick, "Pool with recent sessions (--pool)", self.pool_var).pack(
            anchor="w", padx=8, pady=(0, 2)
        )
        ttk.Label(
            quick,
            text="Also fits using your last few stored calibration sessions (same screen size "
            "and calibration type only) alongside this one, and keeps whichever fit actually "
            "validates better -- prints both numbers so you can judge rather than assuming it "
            "helped.",
            wraplength=740,
            justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        btn_quick = ttk.Button(quick, text="Quick start", command=self._on_quick_start)
        btn_quick.pack(anchor="w", padx=8, pady=(0, 8))
        self.buttons.append(btn_quick)

        self._section_header(parent, "Full Start")
        full = self._boxed_frame(parent)
        full.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(
            full,
            text="Uses the CNN gaze model (see docs/ML_PIPELINE.md) instead of the math model above.",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 2))

        if not TORCH_AVAILABLE:
            ttk.Label(
                full,
                text="torch isn't installed in this environment, so the CNN options below are "
                "disabled. Install it with: uv sync --extra cnn",
                wraplength=740,
                justify="left",
                style="Warn.TLabel",
            ).pack(anchor="w", padx=8, pady=(4, 8))

        record_frame = ttk.Frame(full)
        record_frame.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(
            record_frame,
            text="Record & train: look at the cursor while it moves around the screen for a long "
            "video, adding it to the training dataset, then train the CNN and calibrate it with "
            "a quick 9-dot round (~15s).",
            wraplength=740,
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        # No name field, deliberately: app.py never collects or asks for one.
        # record_session.py falls back to a random id generated once for this
        # machine and reused every time (get_default_participant()) when
        # --participant isn't passed, which is what every call from here does.
        #
        # Label + entry live in their own pack()-ed sub-frame, gridded as a
        # single column-0 cell, rather than splitting them across record_frame's
        # own column 0/1 -- the description label above is wraplength=740 and
        # spans column 0, which forces column 0 itself to ~740px; if the entry
        # sat in a separate column 1 of that same grid, its start position would
        # inherit that 740px offset, putting a large gap before it (this is
        # exactly what happened; shrinking the description's columnspan only
        # concentrated more of that 740px requirement onto fewer columns and
        # made the gap bigger, not smaller). Nesting sidesteps the column-width
        # math entirely: the sub-frame's own small pack layout decides the
        # label/entry spacing, independent of how wide column 0 ends up.
        duration_row = ttk.Frame(record_frame)
        duration_row.grid(row=1, column=0, sticky="w")
        ttk.Label(duration_row, text="Duration (s):").pack(side="left")
        self.duration_var = tk.StringVar(value="300")
        ttk.Entry(duration_row, textvariable=self.duration_var, width=8).pack(side="left", padx=(4, 0))

        btn_record = ttk.Button(record_frame, text="Record, train & calibrate", command=self._on_record_and_train)
        btn_record.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.buttons.append(btn_record)
        if not TORCH_AVAILABLE:
            btn_record.state(["disabled"])

        ttk.Separator(full).pack(fill="x", padx=8, pady=4)

        calib_frame = ttk.Frame(full)
        calib_frame.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(
            calib_frame,
            text="Calibrate only: fit the CNN pipeline to you with just a quick 9-dot "
            "calibration round (~15s), skipping the long recording.",
            wraplength=740,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            calib_frame,
            text="Warning: without also recording & training, accuracy will likely not be as "
            "good -- this only fits a small per-user correction on top of whatever CNN is "
            "already trained (models/gaze_cnn.pt), it does not retrain it on your own data.",
            wraplength=740,
            justify="left",
            style="Warn.TLabel",
        ).pack(anchor="w", pady=(4, 6))
        btn_calib_only = ttk.Button(calib_frame, text="Calibrate only (CNN)", command=self._on_calibrate_only)
        btn_calib_only.pack(anchor="w")
        self.buttons.append(btn_calib_only)
        if not TORCH_AVAILABLE:
            btn_calib_only.state(["disabled"])

        ttk.Button(parent, text="Settings", command=self._show_settings_page).pack(anchor="w", padx=10, pady=10)

    def _build_settings_page(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent)
        top_bar.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Button(top_bar, text="← Back", command=self._show_main_page).pack(side="left")

        self._section_header(parent, "Log")
        log_frame = self._boxed_frame(parent)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.log_text = tk.Text(log_frame, height=16, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, side="left")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(fill="y", side="right")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # No box here on purpose -- just a plain row sitting directly on
        # the page, same as the reset row below it.
        appearance_row = ttk.Frame(parent)
        appearance_row.pack(anchor="w", padx=10, pady=(16, 0))
        ttk.Label(appearance_row, text="Appearance:").pack(side="left")
        ttk.Radiobutton(
            appearance_row, text="Light", variable=self.theme_var, value="light", command=self._apply_theme
        ).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(
            appearance_row, text="Dark", variable=self.theme_var, value="dark", command=self._apply_theme
        ).pack(side="left")

        # No box here on purpose -- just the button and its help badge,
        # sitting directly on the page below it, with extra top margin so
        # it doesn't crowd the appearance row above it.
        reset_row = ttk.Frame(parent)
        reset_row.pack(anchor="w", padx=10, pady=(20, 10))
        btn_clear = ttk.Button(reset_row, text="Clear all calibrations", command=self._on_clear_calibrations)
        btn_clear.pack(side="left")
        self.buttons.append(btn_clear)
        self._make_help_badge(reset_row, self._show_clear_calibrations_help).pack(side="left", padx=(6, 0))

    def _show_main_page(self) -> None:
        self.settings_page.pack_forget()
        self.main_page.pack(fill="both", expand=True)

    def _show_settings_page(self) -> None:
        self.main_page.pack_forget()
        self.settings_page.pack(fill="both", expand=True)

    def _show_clear_calibrations_help(self) -> None:
        messagebox.showinfo("Clear all calibrations", CLEAR_CALIBRATIONS_HELP)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        # Anything CommandRunner needs run on the GUI thread (on_done
        # after a step, or the checkpoint dialog's ask()) arrives here
        # too, via self.runner.action_queue -- see CommandRunner's own
        # docstring for why that's queued rather than root.after()'d
        # directly from the worker thread.
        try:
            while True:
                fn, fn_args = self.runner.action_queue.get_nowait()
                fn(*fn_args)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for b in self.buttons:
            try:
                b.state(["!disabled"] if enabled else ["disabled"])
            except tk.TclError as exc:
                # One bad/stale widget reference must not stop the rest
                # of the list from being updated -- surfaced in the Log
                # (visible either way) rather than disappearing silently.
                self._append_log(f"\n_set_buttons_enabled: couldn't update a button: {exc}\n")

    def _start(self, worker: Callable[[CommandRunner], bool]) -> None:
        self._set_buttons_enabled(False)

        def on_done(success: bool) -> None:
            self._append_log(f"\n=== step finished (success={success}); re-enabling buttons ===\n")
            self._set_buttons_enabled(True)
            if not success:
                _bring_window_to_front(self.root)
                messagebox.showerror("doomscroll launcher", "A step failed or was cancelled -- see the log for details.")

        self.runner.start(worker, on_done)

    def _on_quick_start(self) -> None:
        protocol = self.quick_protocol_var.get()
        pool = self.pool_var.get()

        def worker(runner: CommandRunner) -> bool:
            while True:
                cmd = [PYTHON, str(SCRIPTS / "calibrate.py"), "--protocol", protocol]
                if pool:
                    cmd.append("--pool")
                ok = runner.run_cmd(
                    f"Calibrate (math model, --protocol {protocol}{', --pool' if pool else ''})",
                    cmd,
                )
                if not ok:
                    return False
                choice = runner.ask_calibration_checkpoint()
                if choice == "retry":
                    continue
                if choice == "cancel":
                    return False
                break
            return runner.run_cmd("Run (math model)", [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py")])

        self._start(worker)

    def _on_record_and_train(self) -> None:
        try:
            duration = float(self.duration_var.get().strip())
        except ValueError:
            messagebox.showwarning("doomscroll launcher", "Duration must be a number of seconds.")
            return

        post_steps: list[tuple[str, list[str]]] = []
        for person_dir in legacy_person_dirs():
            post_steps.append(
                (
                    f"Build crops ({person_dir.name}, existing data)",
                    [PYTHON, str(SCRIPTS / "build_crops.py"), "--recordings", str(person_dir)],
                )
            )
        post_steps.append(("Train CNN", [PYTHON, str(SCRIPTS / "train_cnn.py")]))

        def worker(runner: CommandRunner) -> bool:
            ok, lines = runner.run_cmd_capture(
                "Record session",
                [PYTHON, str(SCRIPTS / "record_session.py"), "--duration", str(duration)],
            )
            if not ok:
                return False
            session_id = _extract_session_id(lines)

            if not runner.run_cmd("Build crops (new recording)", [PYTHON, str(SCRIPTS / "build_crops.py")]):
                return False
            if not runner.run_cmd("Build features (new recording)", [PYTHON, str(SCRIPTS / "build_dataset.py")]):
                return False

            # The raw recording is only needed to extract it into
            # datasets/crops/<session_id>.npz and datasets/features/<session_id>.npz,
            # which the two steps above just did -- once that's succeeded, the raw
            # webcam.mp4/frames.csv/meta.json are no longer needed for model
            # training, so they're deleted here (datasets/ is the permanent,
            # anonymized archive; data/recordings/ is just the staging area, per
            # datasets/README.md). Only ever deletes the one folder whose id was
            # actually parsed from this run's own "Record session" output -- if
            # that parse fails for any reason, nothing is deleted rather than
            # risk guessing the wrong folder.
            if session_id:
                session_dir = RECORDINGS_DIR / session_id
                if session_dir.is_dir():
                    try:
                        shutil.rmtree(session_dir)
                        runner.log_queue.put(
                            f"\nDeleted raw recording {session_dir} "
                            f"(already extracted into datasets/crops/ and datasets/features/).\n"
                        )
                    except OSError as exc:
                        runner.log_queue.put(f"\nCouldn't delete raw recording {session_dir}: {exc}\n")
            else:
                runner.log_queue.put(
                    "\nCouldn't find this recording's session id in its own output -- "
                    "leaving its raw files in data/recordings/ rather than guessing.\n"
                )

            # Best-effort: re-estimate the CNN's cursor-lag from the just-updated
            # datasets/features/ archive and store it in models/gaze_model.joblib.
            # train_cnn.py's default_lag() reads this automatically on the retrain
            # just below. Not gated on success -- if this step fails for any
            # reason, that retrain simply falls back to whatever lag_ms was
            # already stored (or 0ms), same as before this existed; it isn't
            # worth blocking recording/training/calibration over.
            runner.run_cmd("Update CNN lag estimate", [PYTHON, str(SCRIPTS / "train_model.py")])

            for label, cmd in post_steps:
                if not runner.run_cmd(label, cmd):
                    return False
            while True:
                ok = runner.run_cmd("Quick calibrate (CNN)", [PYTHON, str(SCRIPTS / "quick_calibrate.py")])
                if not ok:
                    return False
                choice = runner.ask_calibration_checkpoint()
                if choice == "retry":
                    continue
                if choice == "cancel":
                    return False
                break
            return runner.run_cmd("Run (CNN)", [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py"), "--model", "cnn"])

        self._start(worker)

    def _on_calibrate_only(self) -> None:
        if not CNN_MODEL_PATH.exists():
            messagebox.showerror(
                "doomscroll launcher",
                f"No trained CNN model at {CNN_MODEL_PATH.name} yet -- use "
                "'Record, train & calibrate' first.",
            )
            return
        proceed = messagebox.askyesno(
            "doomscroll launcher",
            "Calibrating the CNN without recording & training only fits a small per-user "
            "correction on top of the existing model -- accuracy will likely not be as good as "
            "doing a full record & train. Continue anyway?",
            icon="warning",
        )
        if not proceed:
            return

        def worker(runner: CommandRunner) -> bool:
            while True:
                ok = runner.run_cmd("Quick calibrate (CNN)", [PYTHON, str(SCRIPTS / "quick_calibrate.py")])
                if not ok:
                    return False
                choice = runner.ask_calibration_checkpoint()
                if choice == "retry":
                    continue
                if choice == "cancel":
                    return False
                break
            return runner.run_cmd("Run (CNN)", [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py"), "--model", "cnn"])

        self._start(worker)

    def _on_clear_calibrations(self) -> None:
        proceed = messagebox.askyesno("doomscroll launcher", f"{CLEAR_CALIBRATIONS_HELP} Continue?", icon="warning")
        if not proceed:
            return

        self._append_log("\n=== Clear all calibrations ===\n")
        for label, path in (("math profile", MATH_PROFILE_PATH), ("CNN correction", CNN_CORRECTION_PATH)):
            if path.exists():
                path.unlink()
                self._append_log(f"Deleted {label}: {path}\n")
            else:
                self._append_log(f"No {label} found at {path} -- nothing to delete.\n")

        session_count = 0
        if SESSIONS_DIR.exists():
            for session_path in SESSIONS_DIR.glob("session_*.joblib"):
                session_path.unlink()
                session_count += 1
        self._append_log(f"Deleted {session_count} stored calibration session(s) from {SESSIONS_DIR}.\n")

        messagebox.showinfo("doomscroll launcher", "Calibrations cleared.")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
