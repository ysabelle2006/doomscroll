# Tkinter desktop launcher for doomscroll -- lets the user pick a
# calibration flow, run it, and start/stop the gaze-scroll subprocess.

from __future__ import annotations

import importlib.util
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from platformdirs import user_data_dir

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
RECORDINGS_DIR = ROOT / "data" / "recordings"

# Matches the session id record_session.py prints to its own stdout.
_SESSION_LINE_RE = re.compile(r"Session:\s*(\S+)")


# Returns the session id parsed from record_session.py's printed output.
def _extract_session_id(lines: list[str]) -> str | None:
    for line in lines:
        m = _SESSION_LINE_RE.search(line)
        if m:
            return m.group(1)
    return None


# Matches the calibration summary line printed by calibrate.py/quick_calibrate.py.
_CALIBRATION_SUMMARY_RE = re.compile(
    r"CALIBRATION_SUMMARY:\s*err_px=(?P<err_px>\S+)\s+err_pct=(?P<err_pct>\S+)\s+label=(?P<label>.+)"
)


# Returns (error px, error %, label) parsed from a calibration run's output.
def _extract_calibration_summary(lines: list[str]) -> tuple[float, float, str] | None:
    for line in reversed(lines):
        m = _CALIBRATION_SUMMARY_RE.search(line)
        if not m:
            continue
        try:
            return float(m.group("err_px")), float(m.group("err_pct")), m.group("label").strip()
        except ValueError:
            return None
    return None


# Validation error % at or above which the checkpoint dialog suggests retrying.
LOW_ACCURACY_THRESHOLD_PCT = 10.0

# Printed by calibrate.py/quick_calibrate.py when the user cancels mid-run.
_CANCELLED_MARKER = "CALIBRATION_CANCELLED"


# True if the cancellation marker appears anywhere in the output lines.
def _was_user_cancelled(lines: list[str]) -> bool:
    return any(_CANCELLED_MARKER in line for line in lines)

PYTHON = sys.executable

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CNN_MODEL_PATH = ROOT / "models" / "gaze_cnn.pt"

# Where the math-model calibration, CNN correction, and stored sessions live on disk.
MATH_PROFILE_PATH = Path(user_data_dir("gazescroll")) / "calibration" / "default.joblib"
CNN_CORRECTION_PATH = Path(user_data_dir("doomscroll")) / "calibration" / "affine_correction_cnn.json"
SESSIONS_DIR = MATH_PROFILE_PATH.parent / "sessions"

CLEAR_CALIBRATIONS_HELP = (
    "Deletes the saved math calibration profile, the saved CNN per-user correction, and all "
    "stored calibration sessions (used by --pool on the Quick Start box). Both pipelines will "
    "need a full recalibration before they'll run again, and --pool will have nothing to pool "
    "against until new sessions are captured."
)

DEV_VIEW_HELP = (
    "When on, every Run step (Quick Start, Record & train, Calibrate only) launches "
    "phase2_gaze_scroll.py with --dev-view, which draws the live OCR bounding-box overlay on "
    "top of whatever window you're reading -- useful for checking that OCR and gaze tracking "
    "are doing the right thing. It's excluded from this app's own OCR scan, but it's a real "
    "on-screen window like any other, so it IS visible to screen sharing/recording from other "
    "apps. When off (the default), scrolling behaves identically -- nothing is drawn on top of "
    "your screen."
)

# Light/dark color palettes applied by _apply_theme().
THEMES: dict[str, dict[str, str]] = {
    "light": {"bg": "#ffffff", "fg": "#000000", "border": "#000000", "muted": "#555555", "warn": "#aa3333"},
    "dark": {"bg": "#2b2b2b", "fg": "#ffffff", "border": "#ffffff", "muted": "#aaaaaa", "warn": "#ff6b6b"},
}


# Returns True if the OS's system-wide appearance is currently dark.
def _detect_system_dark_mode() -> bool:
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            return result.returncode == 0 and "dark" in result.stdout.strip().lower()
        # Windows: read the dark-mode setting from the registry.
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0
    except Exception:
        pass
    return False


# Lists existing per-person recording folders under data/raw.
def legacy_person_dirs() -> list[Path]:
    raw = ROOT / "data" / "raw"
    if not raw.is_dir():
        return []
    return sorted(p for p in raw.iterdir() if p.is_dir())


# Deiconifies, raises, and focuses the given Tk window.
def _bring_window_to_front(root: tk.Tk) -> None:
    root.deiconify()
    root.lift()
    root.focus_force()
    root.update()


# Runs launcher subprocess steps on a background thread and reports
# progress and dialogs back to the GUI thread via two queues.
class CommandRunner:

    def __init__(self, root: tk.Tk, log_queue: "queue.Queue[str]") -> None:
        self.root = root
        self.log_queue = log_queue
        self.action_queue: "queue.Queue[tuple[Callable[..., None], tuple]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self.silent_failure = False
        self.current_proc: subprocess.Popen | None = None

    # True while a worker thread is currently running.
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # Starts worker on a background thread; queues on_done with its result.
    def start(self, worker: Callable[["CommandRunner"], bool], on_done: Callable[[bool], None]) -> None:
        if self.running:
            return

        def _run() -> None:
            try:
                success = worker(self)
            except Exception as exc:
                self.log_queue.put(f"\nUnexpected error: {exc}\n")
                success = False
            self.action_queue.put((on_done, (success,)))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # Runs one subprocess step to completion; returns whether it exited 0.
    def run_cmd(self, label: str, cmd: list[str]) -> bool:
        ok, _lines = self.run_cmd_capture(label, cmd)
        return ok

    # Runs one subprocess step, streaming its output to the log and
    # collecting every printed line.
    def run_cmd_capture(
        self,
        label: str,
        cmd: list[str],
        on_started: Callable[[], None] | None = None,
        on_stopped: Callable[[], None] | None = None,
    ) -> tuple[bool, list[str]]:
        self.log_queue.put(f"\n=== {label} ===\n$ {' '.join(cmd)}\n")
        lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Unbuffered so the subprocess's prints stream to the log live.
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            self.log_queue.put(f"Failed to start: {exc}\n")
            return False, lines
        if on_started is not None:
            self.action_queue.put((on_started, ()))
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_queue.put(line)
                lines.append(line)
            code = proc.wait()
        finally:
            if on_stopped is not None:
                self.action_queue.put((on_stopped, ()))
        if code != 0:
            self.log_queue.put(f"\n'{label}' exited with code {code}.\n")
            return False, lines
        return True, lines

    # Runs the scrolling subprocess, tracked in current_proc so it can be
    # stopped, notifying the GUI when it starts and stops.
    def run_scroll(
        self,
        label: str,
        cmd: list[str],
        on_started: Callable[[], None],
        on_stopped: Callable[[], None],
    ) -> bool:
        self.log_queue.put(f"\n=== {label} ===\n$ {' '.join(cmd)}\n")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Unbuffered so the subprocess's prints stream to the log live.
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as exc:
            self.log_queue.put(f"Failed to start: {exc}\n")
            return False
        self.current_proc = proc
        self.action_queue.put((on_started, ()))
        code = 1
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self.log_queue.put(line)
            code = proc.wait()
        finally:
            self.current_proc = None
            self.action_queue.put((on_stopped, ()))
        if code != 0:
            self.log_queue.put(f"\n'{label}' exited with code {code}.\n")
            return False
        return True

    # Sends SIGINT to the running scrolling subprocess, if any.
    def stop_scroll(self) -> None:
        proc = self.current_proc
        if proc is None:
            return
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    # Shows a run/retry/cancel dialog after a calibration step; blocks until answered.
    def ask_calibration_checkpoint(self, summary: tuple[float, float, str] | None = None) -> str:
        result: dict[str, str] = {}
        done = threading.Event()

        if summary is not None:
            err_px, err_pct, label = summary
            accuracy_text = (
                f"Validation error ({label}): {err_px:.1f}px "
                f"({err_pct:.1f}% of screen diagonal).\n\n"
            )
            if err_pct >= LOW_ACCURACY_THRESHOLD_PCT:
                accuracy_text += (
                    "That's a high error relative to screen size -- accuracy is likely poor. "
                    "Consider retrying with better/more even lighting.\n\n"
                )
        else:
            accuracy_text = (
                "No validation error was captured for this calibration run (check the log "
                "above for why).\n\n"
            )

        def ask() -> None:
            _bring_window_to_front(self.root)
            choice = messagebox.askyesnocancel(
                "doomscroll launcher",
                accuracy_text
                + "Yes = launch now with this calibration.\n"
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

    # Shows the recording instructions dialog; blocks until Start is clicked or it's closed.
    def ask_start_recording(self, duration: float) -> bool:
        result: dict[str, bool] = {}
        done = threading.Event()

        def ask() -> None:
            _bring_window_to_front(self.root)
            dialog = tk.Toplevel(self.root)
            dialog.title("doomscroll launcher")
            dialog.resizable(False, False)
            dialog.transient(self.root)

            def _cancel() -> None:
                result["start"] = False
                dialog.destroy()
                done.set()

            def _start() -> None:
                result["start"] = True
                dialog.destroy()
                done.set()

            dialog.protocol("WM_DELETE_WINDOW", _cancel)
            ttk.Label(
                dialog,
                text=(
                    f"About to record a {duration:.0f}s session for training the CNN gaze "
                    "model.\n\n"
                    "Look AT the cursor the whole time and keep moving it -- sweep corners, "
                    "edges, zig-zags, slow and fast. Vary your head position (forward, turned, "
                    "tilted, leaning closer/further) across sessions, and avoid touching the "
                    "keyboard/mouse in ways that pull your eyes from the cursor.\n\n"
                    "Click Start when you're ready."
                ),
                wraplength=420,
                justify="left",
                padding=(20, 16, 20, 8),
            ).pack()
            ttk.Button(dialog, text="Start", command=_start).pack(pady=(0, 16))
            dialog.grab_set()
            dialog.lift()
            dialog.focus_force()

        self.action_queue.put((ask, ()))
        done.wait()
        return result.get("start", False)

    # Shows an OK-only info dialog and blocks until it's dismissed.
    def ask_acknowledge(self, title: str, message: str) -> None:
        done = threading.Event()

        def ask() -> None:
            _bring_window_to_front(self.root)
            messagebox.showinfo(title, message)
            done.set()

        self.action_queue.put((ask, ()))
        done.wait()


# The launcher's main window: builds the UI and wires its buttons to
# the calibration, recording, and run flows.
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("doomscroll launcher")
        root.geometry("800x700")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.runner = CommandRunner(root, self.log_queue)
        self.buttons: list[ttk.Button] = []
        self.boxed_frames: list[tk.Frame] = []
        self.help_badges: list[tk.Canvas] = []
        self.checkboxes: list[Callable[[], None]] = []
        self._recording_timer_win: tk.Toplevel | None = None
        self._recording_timer_label: ttk.Label | None = None
        self._recording_deadline: float = 0.0

        self._setup_styles()
        self._build_layout()
        self._apply_theme()
        self._poll_log()
        self._show_main_page()

    # Sets up fonts and the ttk theme engine used by _apply_theme().
    def _setup_styles(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        self.header_font = (default_font.actual("family"), 14, "bold")
        self.theme_var = tk.StringVar(value="dark" if _detect_system_dark_mode() else "light")
        self._theme_engine_set = False
        self.root.option_add("*TRadiobutton.takeFocus", 0)
        self._apply_theme()

    # Applies the current light/dark theme colors to every themed widget.
    def _apply_theme(self) -> None:
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

        style.configure(
            "TButton",
            borderwidth=1,
            relief="solid",
            bordercolor=theme["muted"],
            lightcolor=theme["muted"],
            darkcolor=theme["muted"],
        )
        style.map("TButton", background=[], foreground=[("disabled", theme["muted"])])
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

    # Adds a bold section-heading label.
    def _section_header(self, parent: tk.Widget, text: str) -> ttk.Label:
        label = ttk.Label(parent, text=text, font=self.header_font, style="Header.TLabel")
        label.pack(anchor="w", padx=10, pady=(10, 2))
        return label

    # Creates a bordered frame used to group a section's widgets.
    def _boxed_frame(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(
            parent,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.border_color,
            bg=self.bg_color,
        )
        self.boxed_frames.append(frame)
        return frame

    # Creates a small clickable '?' badge that opens a help popup.
    def _make_help_badge(self, parent: tk.Widget, command: Callable[[], None]) -> tk.Canvas:
        size = 22
        canvas = tk.Canvas(parent, width=size, height=size, highlightthickness=0, bd=0, bg=self.bg_color)
        canvas.create_oval(1, 1, size - 1, size - 1, fill="#555", outline="#333")
        canvas.create_text(
            size // 2, size // 2, text="?", font=(self.header_font[0], 10, "bold"), fill="white"
        )
        canvas.bind("<Button-1>", lambda _event: command())
        self.help_badges.append(canvas)
        return canvas

    # Creates a custom canvas-drawn checkbox bound to a BooleanVar.
    def _make_checkbox(self, parent: tk.Widget, text: str, variable: tk.BooleanVar) -> ttk.Frame:
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

    # Builds the main page and settings page frames.
    def _build_layout(self) -> None:
        self.main_page = ttk.Frame(self.root)
        self.settings_page = ttk.Frame(self.root)
        self._build_main_page(self.main_page)
        self._build_settings_page(self.settings_page)

    # Builds the main page: calibration/recording options plus the Settings and Stop buttons.
    def _build_main_page(self, parent: ttk.Frame) -> None:
        self._section_header(parent, "Standard Setup")
        quick = self._boxed_frame(parent)
        quick.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(
            quick,
            text="Calibrate your eye tracking and start using the app immediately. No training required.",
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 2))

        protocol_frame = ttk.Frame(quick)
        protocol_frame.pack(fill="x", padx=8, pady=(0, 2))
        ttk.Label(protocol_frame, text="Calibration Type:").pack(side="left")
        self.quick_protocol_var = tk.StringVar(value="trajectory")
        ttk.Radiobutton(
            protocol_frame, text="Fast Calibration (Recommended, faster)", variable=self.quick_protocol_var, value="trajectory"
        ).pack(side="left", padx=(8, 12))
        ttk.Radiobutton(
            protocol_frame, text="Precision calibration (more accurate)", variable=self.quick_protocol_var,
            value="dots",
        ).pack(side="left")
        ttk.Label(
            quick,
            text="Fast Calibration requires following a moving dot around the screen for ~15 seconds for each head pose. Precision Calibration requires looking at a series of fixed dots appearing in sequence around the screen for each head pose.",
            wraplength=740,
            justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        self.pool_var = tk.BooleanVar(value=False)
        self._make_checkbox(quick, "Use previous calibrations", self.pool_var).pack(
            anchor="w", padx=8, pady=(0, 2)
        )
        ttk.Label(
            quick,
            text="Combine this calibration with your recent calibrations to improve accuracy and stability. Recommended if you have calibrated before on the same screen",
            wraplength=740,
            justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", padx=8, pady=(0, 4))

        btn_quick = ttk.Button(quick, text="Start eye tracking", command=self._on_quick_start)
        btn_quick.pack(anchor="w", padx=8, pady=(0, 8))
        self.buttons.append(btn_quick)

        self._section_header(parent, "Personalised Setup")
        full = self._boxed_frame(parent)
        full.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(
            full,
            text="Create a custom eye-tracking model tailored to you. Follow your moving cursor for a few minutes while the app learns how your eyes move. Once recording is complete, calibration runs automatically.",
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

        duration_row = ttk.Frame(record_frame)
        duration_row.grid(row=1, column=0, sticky="w")
        ttk.Label(duration_row, text="Recording Time:").pack(side="left")
        self.duration_var = tk.StringVar(value="300")
        ttk.Entry(duration_row, textvariable=self.duration_var, width=8).pack(side="left", padx=(4, 0))

        btn_record = ttk.Button(record_frame, text="Create personal model", command=self._on_record_and_train)
        btn_record.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.buttons.append(btn_record)
        if not TORCH_AVAILABLE:
            btn_record.state(["disabled"])

        ttk.Separator(full).pack(fill="x", padx=8, pady=4)

        calib_frame = ttk.Frame(full)
        calib_frame.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(
            calib_frame,
            text="Quick personal calibration (originally calibrate only option). Fine tune eye tracking for your eyes and set ups in ~15 seonds, without the longer training process.",
            wraplength=740,
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            calib_frame,
            text="Important: Accuracy will generally be lower if you have not created a personal model. This option only performs quick calibration and does not train on your own eye-tracking data. For best results, use this after Create Personal Model instead.",
            wraplength=740,
            justify="left",
            style="Warn.TLabel",
        ).pack(anchor="w", pady=(4, 6))
        btn_calib_only = ttk.Button(calib_frame, text="Quick personal calibration", command=self._on_calibrate_only)
        btn_calib_only.pack(anchor="w")
        self.buttons.append(btn_calib_only)
        if not TORCH_AVAILABLE:
            btn_calib_only.state(["disabled"])

        bottom_row = ttk.Frame(parent)
        bottom_row.pack(fill="x", padx=10, pady=10)
        ttk.Button(bottom_row, text="Settings", command=self._show_settings_page).pack(side="left")
        self.btn_stop_scrolling = ttk.Button(
            bottom_row, text="Stop Scrolling", command=self._on_stop_scrolling
        )

    # Builds the settings page: log view, appearance, developer view, and reset controls.
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

        appearance_row = ttk.Frame(parent)
        appearance_row.pack(anchor="w", padx=10, pady=(16, 0))
        ttk.Label(appearance_row, text="Appearance:").pack(side="left")
        ttk.Radiobutton(
            appearance_row, text="Light", variable=self.theme_var, value="light", command=self._apply_theme
        ).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(
            appearance_row, text="Dark", variable=self.theme_var, value="dark", command=self._apply_theme
        ).pack(side="left")

        dev_view_row = ttk.Frame(parent)
        dev_view_row.pack(anchor="w", padx=10, pady=(16, 0))
        self.dev_view_var = tk.BooleanVar(value=False)
        self._make_checkbox(dev_view_row, "Developer view", self.dev_view_var).pack(
            side="left"
        )
        self._make_help_badge(dev_view_row, self._show_dev_view_help).pack(side="left", padx=(6, 0))

        reset_row = ttk.Frame(parent)
        reset_row.pack(anchor="w", padx=10, pady=(20, 10))
        btn_clear = ttk.Button(reset_row, text="Clear all calibrations", command=self._on_clear_calibrations)
        btn_clear.pack(side="left")
        self.buttons.append(btn_clear)
        self._make_help_badge(reset_row, self._show_clear_calibrations_help).pack(side="left", padx=(6, 0))

    # Switches the visible page to the main page.
    def _show_main_page(self) -> None:
        self.settings_page.pack_forget()
        self.main_page.pack(fill="both", expand=True)

    # Switches the visible page to the settings page.
    def _show_settings_page(self) -> None:
        self.main_page.pack_forget()
        self.settings_page.pack(fill="both", expand=True)

    # Shows what 'Clear all calibrations' deletes.
    def _show_clear_calibrations_help(self) -> None:
        messagebox.showinfo("Clear all calibrations", CLEAR_CALIBRATIONS_HELP)

    # Shows what the developer view checkbox does.
    def _show_dev_view_help(self) -> None:
        messagebox.showinfo("Developer view", DEV_VIEW_HELP)

    # Appends text to the log widget and scrolls to the end.
    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # Drains the log and action queues on the GUI thread; reschedules itself every 100ms.
    def _poll_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        try:
            while True:
                fn, fn_args = self.runner.action_queue.get_nowait()
                fn(*fn_args)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    # Enables or disables every tracked launcher button.
    def _set_buttons_enabled(self, enabled: bool) -> None:
        for b in self.buttons:
            try:
                b.state(["!disabled"] if enabled else ["disabled"])
            except tk.TclError as exc:
                self._append_log(f"\n_set_buttons_enabled: couldn't update a button: {exc}\n")

    # Disables buttons, runs a worker in the background, and reports its result.
    def _start(self, worker: Callable[[CommandRunner], bool]) -> None:
        self._set_buttons_enabled(False)

        def on_done(success: bool) -> None:
            self._append_log(f"\n=== step finished (success={success}); re-enabling buttons ===\n")
            self._set_buttons_enabled(True)
            if not success and not self.runner.silent_failure:
                _bring_window_to_front(self.root)
                messagebox.showerror("doomscroll launcher", "A step failed -- see the log for details.")
            self.runner.silent_failure = False

        self.runner.start(worker, on_done)

    # Stops the active scrolling subprocess.
    def _on_stop_scrolling(self) -> None:
        self.runner.stop_scroll()

    # Shows the Stop button and, outside developer view, an 'active' popup.
    def _on_scroll_started(self) -> None:
        self.btn_stop_scrolling.pack(side="left", padx=(8, 0))
        if not self.dev_view_var.get():
            _bring_window_to_front(self.root)
            messagebox.showinfo("doomscroll launcher", "Auto scrolling is active.")

    # Hides the Stop button once scrolling has stopped.
    def _on_scroll_stopped(self) -> None:
        try:
            self.btn_stop_scrolling.pack_forget()
        except tk.TclError:
            pass

    # Opens the recording countdown window and starts its timer.
    def _on_record_started(self, duration: float) -> None:
        self._recording_deadline = time.monotonic() + duration
        win = tk.Toplevel(self.root)
        win.title("doomscroll launcher")
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        self._recording_timer_label = ttk.Label(win, justify="center", padding=(28, 22))
        self._recording_timer_label.pack()
        win.lift()
        self._recording_timer_win = win
        self._tick_recording_timer()

    # Updates the countdown label once a second while recording.
    def _tick_recording_timer(self) -> None:
        if self._recording_timer_win is None or self._recording_timer_label is None:
            return
        remaining = max(0.0, self._recording_deadline - time.monotonic())
        mm, ss = divmod(int(remaining + 0.999), 60)
        try:
            self._recording_timer_label.config(text=f"Recording...\n\nTime remaining: {mm}:{ss:02d}")
        except tk.TclError:
            return
        self.root.after(200, self._tick_recording_timer)

    # Closes the recording countdown window.
    def _on_record_stopped(self) -> None:
        self._recording_timer_label = None
        if self._recording_timer_win is not None:
            try:
                self._recording_timer_win.destroy()
            except tk.TclError:
                pass
            self._recording_timer_win = None

    # Runs math-model calibration, then launches scrolling with it.
    def _on_quick_start(self) -> None:
        protocol = self.quick_protocol_var.get()
        pool = self.pool_var.get()
        dev_view = self.dev_view_var.get()

        def worker(runner: CommandRunner) -> bool:
            while True:
                cmd = [PYTHON, str(SCRIPTS / "calibrate.py"), "--protocol", protocol]
                if pool:
                    cmd.append("--pool")
                ok, lines = runner.run_cmd_capture(
                    f"Calibrate (math model, --protocol {protocol}{', --pool' if pool else ''})",
                    cmd,
                )
                if not ok:
                    if _was_user_cancelled(lines):
                        runner.silent_failure = True
                    return False
                choice = runner.ask_calibration_checkpoint(_extract_calibration_summary(lines))
                if choice == "retry":
                    continue
                if choice == "cancel":
                    runner.silent_failure = True
                    return False
                break
            run_cmd = [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py")]
            if dev_view:
                run_cmd.append("--dev-view")
            return runner.run_scroll(
                "Run (math model)", run_cmd, on_started=self._on_scroll_started, on_stopped=self._on_scroll_stopped
            )

        self._start(worker)

    # Records a session, builds training data, retrains the CNN, calibrates it, then launches scrolling.
    def _on_record_and_train(self) -> None:
        try:
            duration = float(self.duration_var.get().strip())
        except ValueError:
            messagebox.showwarning("doomscroll launcher", "Duration must be a number of seconds.")
            return
        dev_view = self.dev_view_var.get()

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
            if not runner.ask_start_recording(duration):
                runner.silent_failure = True
                return False
            ok, lines = runner.run_cmd_capture(
                "Record session",
                [PYTHON, str(SCRIPTS / "record_session.py"), "--duration", str(duration)],
                on_started=lambda: self._on_record_started(duration),
                on_stopped=self._on_record_stopped,
            )
            if not ok:
                return False
            session_id = _extract_session_id(lines)

            runner.ask_acknowledge(
                "doomscroll launcher",
                "Recording finished.\n\nTraining will start now -- check the Log (on the Settings "
                "page) for progress.",
            )

            if not runner.run_cmd("Build crops (new recording)", [PYTHON, str(SCRIPTS / "build_crops.py")]):
                return False
            if not runner.run_cmd("Build features (new recording)", [PYTHON, str(SCRIPTS / "build_dataset.py")]):
                return False

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

            runner.run_cmd("Update CNN lag estimate", [PYTHON, str(SCRIPTS / "train_model.py")])

            for label, cmd in post_steps:
                if not runner.run_cmd(label, cmd):
                    return False

            runner.ask_acknowledge(
                "doomscroll launcher",
                "Training finished.\n\nCalibration will start now.",
            )

            while True:
                ok, lines = runner.run_cmd_capture(
                    "Quick calibrate (CNN)", [PYTHON, str(SCRIPTS / "quick_calibrate.py")]
                )
                if not ok:
                    if _was_user_cancelled(lines):
                        runner.silent_failure = True
                    return False
                choice = runner.ask_calibration_checkpoint(_extract_calibration_summary(lines))
                if choice == "retry":
                    continue
                if choice == "cancel":
                    runner.silent_failure = True
                    return False
                break
            run_cmd = [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py"), "--model", "cnn"]
            if dev_view:
                run_cmd.append("--dev-view")
            return runner.run_scroll(
                "Run (CNN)", run_cmd, on_started=self._on_scroll_started, on_stopped=self._on_scroll_stopped
            )

        self._start(worker)

    # Calibrates the existing CNN model without retraining, then launches scrolling.
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
        dev_view = self.dev_view_var.get()

        def worker(runner: CommandRunner) -> bool:
            while True:
                ok, lines = runner.run_cmd_capture(
                    "Quick calibrate (CNN)", [PYTHON, str(SCRIPTS / "quick_calibrate.py")]
                )
                if not ok:
                    if _was_user_cancelled(lines):
                        runner.silent_failure = True
                    return False
                choice = runner.ask_calibration_checkpoint(_extract_calibration_summary(lines))
                if choice == "retry":
                    continue
                if choice == "cancel":
                    runner.silent_failure = True
                    return False
                break
            run_cmd = [PYTHON, str(SCRIPTS / "phase2_gaze_scroll.py"), "--model", "cnn"]
            if dev_view:
                run_cmd.append("--dev-view")
            return runner.run_scroll(
                "Run (CNN)", run_cmd, on_started=self._on_scroll_started, on_stopped=self._on_scroll_stopped
            )

        self._start(worker)

    # Deletes saved calibration files and stored sessions after confirmation.
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


# Creates the Tk root window and runs the launcher.
def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
