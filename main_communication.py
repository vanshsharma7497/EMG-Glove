"""
main_communication.py
---------------------
Communication Mode — user-facing interface for non-verbal users.

Runs a large, high-contrast Tkinter window designed to be readable from
a distance. The interface has two audiences:
  - The non-verbal user operating the glove (sees the full UI)
  - The person being communicated with (reads the large main display)

Layout
------
  Header bar          : mode title + context indicator (pen/glove)
  Preview panel       : live text being written — not yet confirmed
  Action buttons      : Confirm + Speak  |  Clear All
  Main display        : last confirmed message — very large text
  Volume bar          : current TTS volume level
  Lower section       : Quick Phrases (left)  |  Conversation History (right)
  Event log           : small developer log at bottom
  Simulation panel    : prototype controls (collapsed by default)
  Panic overlay       : full-window red alert — appears on top of everything

Run from the GLOVE/ root directory:
    python main_communication.py

Dependencies
------------
  pip install pyttsx3      (for text-to-speech — UI runs without it)
  All other dependencies are stdlib or already used by the project.
"""

import sys
import types
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from tkinter import simpledialog, messagebox

# ── Mock pyautogui ────────────────────────────────────────────────────────────
# WritingController imports pyautogui at module level. Since Communication Mode
# never calls typewrite/click, we mock it so the UI runs on any machine.
# Only inject if not already imported — avoids clobbering a real instance.
if "pyautogui" not in sys.modules:
    _pg = types.ModuleType("pyautogui")
    _pg.FAILSAFE    = True
    _pg.PAUSE       = 0
    _pg.position    = lambda: (500, 400)
    _pg.size        = lambda: (1920, 1080)
    _pg.moveTo      = lambda x, y, duration=0: None
    _pg.click       = lambda button="left": None
    _pg.doubleClick = lambda button="left": None
    _pg.press       = lambda k: None
    _pg.typewrite   = lambda t, interval=0: None
    _pg.scroll      = lambda x: None
    sys.modules["pyautogui"] = _pg

# ── Project imports ───────────────────────────────────────────────────────────
from modes.communication_mode import CommunicationController, DEFAULT_QUICK_PHRASES
from core.pen_simulator import (
    generate_pressure_writing,
    generate_pressure_idle,
    generate_pressure_light_tap,
    SAMPLE_RATE,
)
from core.letter_templates   import LETTER_TEMPLATES, _resample
from core.pen_stroke_processor import STROKE_SCALE

# ---------------------------------------------------------------------------
# Simulation helpers — same pattern as main_writing.py
# ---------------------------------------------------------------------------

SAMPLE_PERIOD = 1.0 / SAMPLE_RATE

def _template_to_gyro(letter, duration=0.5):
    """Derive gyro_x, gyro_y arrays from a letter template path."""
    path      = LETTER_TEMPLATES[letter]
    n         = int(duration * SAMPLE_RATE)
    resampled = _resample(path, n=n)
    dx = np.diff(resampled[:, 0], prepend=resampled[0, 0])
    dy = np.diff(resampled[:, 1], prepend=resampled[0, 1])
    gyro_x = (dx / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
    gyro_y = (dy / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
    return gyro_x, gyro_y


# Pre-built demo phrases — all letters now available (full alphabet templates).
DEMO_PHRASES = {
    "i am sure"  : list("i") + list("sure"),
    "i can"      : list("i") + list("can"),
    "is nice"    : list("is") + list("nice"),
    "no"         : list("no"),
    "i love"     : list("i") + list("love"),  # l, o, v, e all available
    "so real"    : list("so") + list("real"),
}

# ---------------------------------------------------------------------------
# Colours and fonts — defined once, used everywhere
# ---------------------------------------------------------------------------

COLOURS = {
    # Main background
    "bg"              : "#FFFFFF",
    "bg_secondary"    : "#F5F5F5",
    "bg_header"       : "#1A1A2E",

    # Text
    "text_primary"    : "#1A1A1A",
    "text_secondary"  : "#555555",
    "text_header"     : "#FFFFFF",
    "text_preview"    : "#333333",
    "text_confirmed"  : "#0A0A0A",
    "text_placeholder": "#AAAAAA",

    # Accent
    "accent"          : "#2563EB",   # blue — confirm button
    "accent_hover"    : "#1D4ED8",
    "danger"          : "#DC2626",   # red — clear + panic
    "danger_hover"    : "#B91C1C",
    "success"         : "#16A34A",   # green — phrase buttons
    "success_hover"   : "#15803D",
    "neutral"         : "#6B7280",   # grey — secondary buttons
    "neutral_hover"   : "#4B5563",

    # Panels
    "preview_bg"      : "#EFF6FF",   # light blue tint
    "preview_border"  : "#BFDBFE",
    "confirmed_bg"    : "#F0FDF4",   # light green tint
    "confirmed_border": "#BBF7D0",
    "log_bg"          : "#F9FAFB",
    "history_bg"      : "#FFFFFF",

    # Volume bar
    "volume_fill"     : "#2563EB",
    "volume_track"    : "#E5E7EB",

    # Panic
    "panic_bg"        : "#DC2626",
    "panic_text"      : "#FFFFFF",
}


# ---------------------------------------------------------------------------
# CommunicationApp — the main window class
# ---------------------------------------------------------------------------

class CommunicationApp:
    """
    The full Communication Mode UI.

    Structure
    ---------
    __init__           : create window, all frames and widgets
    _build_*           : one method per panel section
    _update_loop       : called every 50ms — reads controller state, refreshes UI
    _sim_*             : simulation helpers called by prototype buttons
    _on_*              : event handlers (button clicks, etc.)
    """

    UPDATE_INTERVAL_MS = 50   # UI refresh rate — 20 fps is smooth enough for text

    def __init__(self, root):
        self.root = root
        self.root.title("EMG Smart Glove — Communication Mode")
        self.root.configure(bg=COLOURS["bg"])
        self.root.minsize(900, 700)

        # ── Controller ────────────────────────────────────────────────────
        self.controller = CommunicationController()

        # ── Quick phrases list (mutable — user can add/remove) ────────────
        self.quick_phrases = list(DEFAULT_QUICK_PHRASES)

        # ── Simulation state ──────────────────────────────────────────────
        # The simulation runs in a background thread so it doesn't block
        # the Tkinter main loop. _sim_running prevents two simulations
        # overlapping if the user clicks buttons quickly.
        self._sim_running  = False
        self._sim_thread   = None

        # Track last confirmed message to detect changes for animation
        self._last_confirmed = ""

        # ── Scrollable container ─────────────────────────────────────────
        # All content lives inside a Canvas+Frame so the window is scrollable.
        # The header stays fixed outside the scroll area.
        self._build_header()   # header is fixed — outside the scroll canvas

        # Scrollable canvas covers the rest of the window
        self._scroll_canvas = tk.Canvas(self.root, bg=COLOURS["bg"], highlightthickness=0)
        self._scrollbar     = tk.Scrollbar(self.root, orient="vertical",
                                           command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scroll_canvas.grid(row=1, column=0, sticky="nsew")
        self._scrollbar.grid(row=1, column=1, sticky="ns")

        # Inner frame — all panels are children of this frame
        self._inner = tk.Frame(self._scroll_canvas, bg=COLOURS["bg"])
        self._canvas_window = self._scroll_canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )

        # Keep inner frame width in sync with canvas width
        def _on_canvas_resize(event):
            self._scroll_canvas.itemconfig(self._canvas_window, width=event.width)
        self._scroll_canvas.bind("<Configure>", _on_canvas_resize)

        # Update scroll region when inner frame content changes size
        self._inner.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")
            )
        )

        # ── Build all UI sections into self._inner ────────────────────────
        self._build_preview_panel()
        self._build_action_buttons()
        self._build_main_display()
        self._build_volume_bar()
        self._build_lower_section()    # quick phrases + history side by side
        self._build_event_log()
        self._build_simulation_panel()
        self._build_panic_overlay()    # built last so it sits on top

        # ── Configure root grid weights ───────────────────────────────────
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)   # scroll canvas expands

        # ── Inner frame grid weights ──────────────────────────────────────
        self._inner.columnconfigure(0, weight=1)
        self._inner.rowconfigure(4, weight=1)   # lower section expands

        # ── Bind mousewheel to scroll the main canvas ─────────────────────
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>",   self._on_mousewheel)
        self.root.bind_all("<Button-5>",   self._on_mousewheel)

        # ── Start the update loop ─────────────────────────────────────────
        self.root.after(self.UPDATE_INTERVAL_MS, self._update_loop)

        print("\n  CommunicationApp started")
        print("  Window ready — use simulation panel to test all features\n")


    # =========================================================================
    # Panel builders — called once during __init__
    # =========================================================================

    def _build_header(self):
        """
        Dark header bar spanning the full width.
        Shows the mode name on the left and the current context (pen/glove) on right.
        """
        header = tk.Frame(self.root, bg=COLOURS["bg_header"], height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)          # keep fixed height even when empty
        header.columnconfigure(1, weight=1)   # middle spacer expands

        # Mode title
        tk.Label(
            header,
            text="⌨  EMG Smart Glove — Communication Mode",
            font=("Helvetica", 15, "bold"),
            bg=COLOURS["bg_header"],
            fg=COLOURS["text_header"],
            padx=20,
        ).grid(row=0, column=0, sticky="w", pady=10)

        # Spacer column
        tk.Label(header, bg=COLOURS["bg_header"]).grid(row=0, column=1, sticky="ew")

        # Context indicator — updates each loop to show pen/glove mode
        self._ctx_label = tk.Label(
            header,
            text="● PEN MODE",
            font=("Helvetica", 11, "bold"),
            bg=COLOURS["bg_header"],
            fg="#60A5FA",   # light blue
            padx=20,
        )
        self._ctx_label.grid(row=0, column=2, sticky="e", pady=10)


    def _build_preview_panel(self):
        """
        The preview panel shows live text as it's being written.
        Text appears here BEFORE confirmation. Light blue background to
        signal "not yet sent".
        """
        outer = tk.Frame(self._inner, bg=COLOURS["bg"], padx=16, pady=4)
        outer.grid(row=1, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)

        tk.Label(
            outer,
            text="Writing Preview",
            font=("Helvetica", 10),
            bg=COLOURS["bg"],
            fg=COLOURS["text_secondary"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        # The preview box itself
        preview_frame = tk.Frame(
            outer,
            bg=COLOURS["preview_bg"],
            highlightbackground=COLOURS["preview_border"],
            highlightthickness=2,
        )
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        preview_frame.columnconfigure(0, weight=1)

        self._preview_label = tk.Label(
            preview_frame,
            text="Start writing with the pen...",
            font=("Helvetica", 22),
            bg=COLOURS["preview_bg"],
            fg=COLOURS["text_placeholder"],
            anchor="w",
            padx=16,
            pady=14,
            wraplength=820,       # wrap long sentences
            justify="left",
        )
        self._preview_label.grid(row=0, column=0, sticky="ew")


    def _build_action_buttons(self):
        """
        Two large buttons: Confirm + Speak, and Clear All.
        These are the primary interaction points for the non-verbal user.
        """
        btn_row = tk.Frame(self._inner, bg=COLOURS["bg"], padx=16, pady=6)
        btn_row.grid(row=2, column=0, sticky="ew")

        # Confirm + Speak — the most important button, takes 2/3 of the space
        self._confirm_btn = tk.Button(
            btn_row,
            text="✓  Confirm + Speak  🔊",
            font=("Helvetica", 14, "bold"),
            bg=COLOURS["accent"],
            fg="white",
            activebackground=COLOURS["accent_hover"],
            activeforeground="white",
            relief="flat",
            padx=24,
            pady=12,
            cursor="hand2",
            command=self._on_confirm,
        )
        self._confirm_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))

        # Clear All
        self._clear_btn = tk.Button(
            btn_row,
            text="✕  Clear All",
            font=("Helvetica", 14),
            bg=COLOURS["danger"],
            fg="white",
            activebackground=COLOURS["danger_hover"],
            activeforeground="white",
            relief="flat",
            padx=24,
            pady=12,
            cursor="hand2",
            command=self._on_clear,
        )
        self._clear_btn.pack(side="left", padx=(0, 0))


    def _build_main_display(self):
        """
        The main display panel — large text showing the last confirmed message.
        This is what the other person in the conversation reads.
        Minimum height so it's always prominent even when empty.
        """
        outer = tk.Frame(self._inner, bg=COLOURS["bg"], padx=16, pady=2)
        outer.grid(row=3, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)

        tk.Label(
            outer,
            text="Confirmed Message",
            font=("Helvetica", 10),
            bg=COLOURS["bg"],
            fg=COLOURS["text_secondary"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        display_frame = tk.Frame(
            outer,
            bg=COLOURS["confirmed_bg"],
            highlightbackground=COLOURS["confirmed_border"],
            highlightthickness=2,
            height=90,
        )
        display_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        display_frame.columnconfigure(0, weight=1)
        display_frame.grid_propagate(False)   # keep minimum height

        self._confirmed_label = tk.Label(
            display_frame,
            text="—",
            font=("Helvetica", 36, "bold"),
            bg=COLOURS["confirmed_bg"],
            fg=COLOURS["text_placeholder"],
            anchor="w",
            padx=16,
            pady=12,
            wraplength=820,
            justify="left",
        )
        self._confirmed_label.grid(row=0, column=0, sticky="nsew")


    def _build_volume_bar(self):
        """
        Volume indicator — shows current TTS volume as a canvas bar.
        Also has + / - buttons for manual adjustment via mouse.
        The bar fills left-to-right; percentage shown as text.
        """
        vol_row = tk.Frame(self._inner, bg=COLOURS["bg"], padx=16, pady=6)
        vol_row.grid(row=4, column=0, sticky="ew")

        tk.Label(
            vol_row,
            text="🔊 Volume",
            font=("Helvetica", 10),
            bg=COLOURS["bg"],
            fg=COLOURS["text_secondary"],
        ).pack(side="left", padx=(0, 10))

        # Minus button
        tk.Button(
            vol_row,
            text="−",
            font=("Helvetica", 12, "bold"),
            bg=COLOURS["neutral"],
            fg="white",
            activebackground=COLOURS["neutral_hover"],
            activeforeground="white",
            relief="flat",
            width=2,
            cursor="hand2",
            command=lambda: self._on_volume_change(-0.08),
        ).pack(side="left", padx=(0, 4))

        # The bar — drawn on a Canvas widget
        # Canvas lets us draw rectangles with precise control over colour and size
        self._vol_canvas = tk.Canvas(
            vol_row,
            height=22,
            width=300,
            bg=COLOURS["volume_track"],
            highlightthickness=1,
            highlightbackground="#D1D5DB",
        )
        self._vol_canvas.pack(side="left", padx=(0, 8))

        # The filled rectangle — we'll resize it in the update loop
        # x0=0, y0=0, x1=initial_width, y1=22
        self._vol_bar_rect = self._vol_canvas.create_rectangle(
            0, 0, 210, 22,   # 70% of 300px = 210
            fill=COLOURS["volume_fill"],
            outline="",
        )

        # Plus button
        tk.Button(
            vol_row,
            text="+",
            font=("Helvetica", 12, "bold"),
            bg=COLOURS["neutral"],
            fg="white",
            activebackground=COLOURS["neutral_hover"],
            activeforeground="white",
            relief="flat",
            width=2,
            cursor="hand2",
            command=lambda: self._on_volume_change(+0.08),
        ).pack(side="left", padx=(0, 8))

        self._vol_pct_label = tk.Label(
            vol_row,
            text="70%",
            font=("Helvetica", 10, "bold"),
            bg=COLOURS["bg"],
            fg=COLOURS["text_secondary"],
            width=4,
            anchor="w",
        )
        self._vol_pct_label.pack(side="left")


    def _build_lower_section(self):
        """
        Two-column section:
          Left  (30%) : Quick phrases panel
          Right (70%) : Conversation history
        """
        lower = tk.Frame(self._inner, bg=COLOURS["bg"], padx=16, pady=4)
        lower.grid(row=5, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=3)   # quick phrases — narrower
        lower.columnconfigure(1, weight=7)   # history — wider
        lower.rowconfigure(0, weight=1)

        self._build_quick_phrases(lower)
        self._build_history_panel(lower)


    def _build_quick_phrases(self, parent):
        """
        Left panel: scrollable list of phrase buttons.
        Each phrase fires immediately when clicked (bypasses writing pipeline).
        User can add custom phrases or remove existing ones.
        """
        frame = tk.Frame(
            parent,
            bg=COLOURS["bg_secondary"],
            highlightbackground="#E5E7EB",
            highlightthickness=1,
        )
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        tk.Label(
            frame,
            text="Quick Phrases",
            font=("Helvetica", 11, "bold"),
            bg=COLOURS["bg_secondary"],
            fg=COLOURS["text_primary"],
            anchor="w",
            padx=10,
            pady=8,
        ).grid(row=0, column=0, sticky="ew")

        # Scrollable frame for phrase buttons
        # Tkinter doesn't have a native scrollable frame — the standard trick
        # is to put a Canvas inside the frame, then place a Frame on the canvas,
        # and attach a Scrollbar to the canvas.
        canvas_frame = tk.Frame(frame, bg=COLOURS["bg_secondary"])
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self._phrases_canvas = tk.Canvas(
            canvas_frame,
            bg=COLOURS["bg_secondary"],
            highlightthickness=0,
            height=160,
        )
        scrollbar = tk.Scrollbar(
            canvas_frame, orient="vertical",
            command=self._phrases_canvas.yview
        )
        self._phrases_canvas.configure(yscrollcommand=scrollbar.set)

        self._phrases_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # The actual frame that holds the buttons lives inside the canvas
        self._phrases_inner = tk.Frame(
            self._phrases_canvas,
            bg=COLOURS["bg_secondary"]
        )
        self._phrases_canvas_window = self._phrases_canvas.create_window(
            (0, 0), window=self._phrases_inner, anchor="nw"
        )

        # When the inner frame resizes, update the canvas scroll region
        self._phrases_inner.bind(
            "<Configure>",
            lambda e: self._phrases_canvas.configure(
                scrollregion=self._phrases_canvas.bbox("all")
            )
        )
        # Make canvas width match the frame width
        self._phrases_canvas.bind(
            "<Configure>",
            lambda e: self._phrases_canvas.itemconfig(
                self._phrases_canvas_window, width=e.width
            )
        )

        # Add/Remove buttons at the bottom
        btn_frame = tk.Frame(frame, bg=COLOURS["bg_secondary"])
        btn_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=6)

        tk.Button(
            btn_frame,
            text="+ Add Phrase",
            font=("Helvetica", 9),
            bg=COLOURS["neutral"],
            fg="white",
            activebackground=COLOURS["neutral_hover"],
            activeforeground="white",
            relief="flat",
            padx=8, pady=4,
            cursor="hand2",
            command=self._on_add_phrase,
        ).pack(side="left", padx=(0, 4))

        tk.Button(
            btn_frame,
            text="− Remove Last",
            font=("Helvetica", 9),
            bg=COLOURS["neutral"],
            fg="white",
            activebackground=COLOURS["neutral_hover"],
            activeforeground="white",
            relief="flat",
            padx=8, pady=4,
            cursor="hand2",
            command=self._on_remove_phrase,
        ).pack(side="left")

        # Render initial phrase buttons
        self._refresh_phrase_buttons()


    def _refresh_phrase_buttons(self):
        """
        Rebuild the phrase buttons from self.quick_phrases.
        Called on init and whenever phrases are added/removed.
        """
        # Clear existing buttons
        for widget in self._phrases_inner.winfo_children():
            widget.destroy()

        for phrase in self.quick_phrases:
            # Each phrase button needs its own closure variable.
            # Without `p=phrase` in the lambda default, all buttons would
            # reference the last value of `phrase` from the loop.
            btn = tk.Button(
                self._phrases_inner,
                text=phrase,
                font=("Helvetica", 11),
                bg=COLOURS["success"],
                fg="white",
                activebackground=COLOURS["success_hover"],
                activeforeground="white",
                relief="flat",
                padx=12, pady=8,
                cursor="hand2",
                anchor="w",
                command=lambda p=phrase: self._on_quick_phrase(p),
            )
            btn.pack(fill="x", padx=8, pady=3)


    def _build_history_panel(self, parent):
        """
        Right panel: scrollable conversation history log.
        Each confirmed message appears here with a timestamp.
        """
        frame = tk.Frame(
            parent,
            bg=COLOURS["history_bg"],
            highlightbackground="#E5E7EB",
            highlightthickness=1,
        )
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        header_row = tk.Frame(frame, bg=COLOURS["history_bg"])
        header_row.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        header_row.columnconfigure(0, weight=1)

        tk.Label(
            header_row,
            text="Conversation History",
            font=("Helvetica", 11, "bold"),
            bg=COLOURS["history_bg"],
            fg=COLOURS["text_primary"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Button(
            header_row,
            text="Clear History",
            font=("Helvetica", 9),
            bg=COLOURS["neutral"],
            fg="white",
            activebackground=COLOURS["neutral_hover"],
            activeforeground="white",
            relief="flat",
            padx=8, pady=2,
            cursor="hand2",
            command=self._on_clear_history,
        ).grid(row=0, column=1, sticky="e")

        # Text widget for history — scrollable, read-only
        # We use a Text widget (not Label) because it supports scrolling and
        # multiple lines with different formatting (timestamps vs messages).
        text_frame = tk.Frame(frame, bg=COLOURS["history_bg"])
        text_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self._history_text = tk.Text(
            text_frame,
            font=("Helvetica", 12),
            bg=COLOURS["history_bg"],
            fg=COLOURS["text_primary"],
            relief="flat",
            wrap="word",
            state="disabled",   # read-only — we insert programmatically
            cursor="arrow",
            height=8,
        )
        hist_scroll = tk.Scrollbar(
            text_frame, orient="vertical",
            command=self._history_text.yview
        )
        self._history_text.configure(yscrollcommand=hist_scroll.set)
        self._history_text.grid(row=0, column=0, sticky="nsew")
        hist_scroll.grid(row=0, column=1, sticky="ns")

        # Text tags for styling timestamp vs message text
        self._history_text.tag_configure(
            "timestamp",
            font=("Helvetica", 9),
            foreground="#9CA3AF",
        )
        self._history_text.tag_configure(
            "message",
            font=("Helvetica", 13),
            foreground=COLOURS["text_primary"],
            spacing1=2,   # padding above line
            spacing3=6,   # padding below line
        )

        self._history_len = 0   # track how many entries we've rendered


    def _build_event_log(self):
        """
        Small developer log at the bottom — shows recent controller events.
        This is a debugging / awareness tool, not user-facing.
        """
        log_frame = tk.Frame(
            self.root,
            bg=COLOURS["log_bg"],
            highlightbackground="#E5E7EB",
            highlightthickness=1,
        )
        log_frame.grid(row=6, column=0, sticky="ew", padx=16, pady=(4, 0))
        log_frame.columnconfigure(1, weight=1)

        tk.Label(
            log_frame,
            text="Event Log",
            font=("Helvetica", 8, "bold"),
            bg=COLOURS["log_bg"],
            fg=COLOURS["text_secondary"],
            padx=8, pady=4,
        ).grid(row=0, column=0, sticky="w")

        self._log_label = tk.Label(
            log_frame,
            text="",
            font=("Helvetica", 8),
            bg=COLOURS["log_bg"],
            fg=COLOURS["text_secondary"],
            anchor="w",
            padx=4,
        )
        self._log_label.grid(row=0, column=1, sticky="ew")


    def _build_simulation_panel(self):
        """
        Prototype simulation controls — collapsible section at the bottom.

        Organised into four groups:
          1. Write phrases  : buttons that feed pen strokes into the pipeline
          2. Pen gestures   : double-tap, triple-tap, tilt, panic (5 taps)
          3. Glove gestures : all 8 mapped gestures
          4. Context switch : toggle between pen and glove mode display
        """
        # Outer container
        outer = tk.Frame(
            self.root,
            bg="#1E293B",
        )
        outer.grid(row=7, column=0, sticky="ew", pady=(8, 0))
        outer.columnconfigure(0, weight=1)

        # Toggle button — shows/hides the panel body
        self._sim_visible = tk.BooleanVar(value=True)

        def toggle_sim():
            if self._sim_visible.get():
                sim_body.grid_remove()
                toggle_btn.config(text="▶  Simulation Controls (show)")
                self._sim_visible.set(False)
            else:
                sim_body.grid()
                toggle_btn.config(text="▼  Simulation Controls (hide)")
                self._sim_visible.set(True)

        toggle_btn = tk.Button(
            outer,
            text="▼  Simulation Controls (hide)",
            font=("Helvetica", 9, "bold"),
            bg="#1E293B",
            fg="#94A3B8",
            activebackground="#334155",
            activeforeground="#CBD5E1",
            relief="flat",
            anchor="w",
            padx=12, pady=6,
            cursor="hand2",
            command=toggle_sim,
        )
        toggle_btn.grid(row=0, column=0, sticky="ew")

        # Panel body
        sim_body = tk.Frame(outer, bg="#1E293B", padx=12, pady=8)
        sim_body.grid(row=1, column=0, sticky="ew")
        sim_body.columnconfigure((0, 1, 2, 3), weight=1)   # 4 equal columns

        def sim_label(parent, text, col):
            tk.Label(
                parent, text=text,
                font=("Helvetica", 8, "bold"),
                bg="#1E293B", fg="#64748B",
                anchor="w",
            ).grid(row=0, column=col, sticky="w", padx=4, pady=(0, 4))

        def sim_btn(parent, text, cmd, row, col, colour="#334155"):
            b = tk.Button(
                parent, text=text,
                font=("Helvetica", 8),
                bg=colour, fg="#E2E8F0",
                activebackground="#475569",
                activeforeground="white",
                relief="flat",
                padx=6, pady=3,
                cursor="hand2",
                command=cmd,
            )
            b.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            return b

        # ── Column 0: Write phrases ───────────────────────────────────────
        sim_label(sim_body, "✏ Write Phrase", 0)
        for i, (label, _) in enumerate(DEMO_PHRASES.items()):
            sim_btn(sim_body, label, lambda l=label: self._sim_write(l), i + 1, 0)

        # ── Column 1: Pen gestures ────────────────────────────────────────
        sim_label(sim_body, "✋ Pen Gestures", 1)
        sim_btn(sim_body, "Double Tap (undo word)",  self._sim_double_tap, 1, 1)
        sim_btn(sim_body, "Triple Tap (clear all)",  self._sim_triple_tap, 2, 1)
        sim_btn(sim_body, "Tilt Right (vol +)",      self._sim_tilt_up,    3, 1)
        sim_btn(sim_body, "Tilt Left  (vol −)",      self._sim_tilt_down,  4, 1)
        sim_btn(sim_body, "5 Taps (panic alert)",    self._sim_five_taps,  5, 1,
                colour="#7F1D1D")

        # ── Column 2: Glove gestures ──────────────────────────────────────
        sim_label(sim_body, "🧤 Glove Gestures", 2)
        glove_gestures = [
            ("thumb+index → Yes",           "thumb_index_tap"),
            ("thumb+middle → No",           "thumb_middle_tap"),
            ("thumb+ring → Thank you",      "thumb_ring_tap"),
            ("thumb+pinky → I need help",   "thumb_pinky_tap"),
            ("index×2 → Please wait",       "index_double_tap"),
            ("mid+idx → Vol −",             "middle_index_tap"),
            ("pinky+ring → Vol +",          "pinky_ring_tap"),
            ("idx+ring → PANIC",            "index_ring_tap"),
        ]
        for i, (label, gesture) in enumerate(glove_gestures):
            colour = "#7F1D1D" if "PANIC" in label else "#334155"
            sim_btn(
                sim_body, label,
                lambda g=gesture: self.controller.process_glove_gesture(g),
                i + 1, 2, colour=colour,
            )

        # ── Column 3: Context + misc ──────────────────────────────────────
        sim_label(sim_body, "⚙ Controls", 3)

        # Auto-demo: writes a full sentence then stops — user presses Confirm+Speak
        # This is specifically to test TTS: write → confirm → hear it spoken
        self._demo_btn = sim_btn(
            sim_body, "▶ Auto Demo (write sentence)",
            self._sim_auto_demo, 1, 3, colour="#1E3A5F"
        )
        self._demo_status = tk.Label(
            sim_body, text="",
            font=("Helvetica", 7), bg="#1E293B", fg="#94A3B8",
        )
        self._demo_status.grid(row=2, column=3, sticky="w", padx=4)

        sim_btn(sim_body, "Dismiss Panic",       self.controller.dismiss_panic, 3, 3)
        sim_btn(sim_body, "Clear History",        self._on_clear_history,        4, 3)
        sim_btn(sim_body, "Simulate Panic (UI)", self._on_panic_btn,             5, 3,
                colour="#7F1D1D")


    def _on_mousewheel(self, event):
        """
        Mousewheel scrolls the main window canvas.
        If the mouse is over the history Text or phrases Canvas, scroll those instead.
        """
        if event.num == 4:      # Linux scroll up
            delta = -1
        elif event.num == 5:    # Linux scroll down
            delta = 1
        else:                   # Windows/Mac — event.delta is ±120
            delta = -1 if event.delta > 0 else 1

        # Check if mouse is over the history text or phrases canvas — scroll those
        try:
            widget = event.widget
            while widget and widget != self.root:
                if isinstance(widget, tk.Text):
                    widget.yview_scroll(delta, "units")
                    return
                if widget is self._phrases_canvas:
                    widget.yview_scroll(delta, "units")
                    return
                widget = widget.master
        except Exception:
            pass

        # Default: scroll the main window canvas
        self._scroll_canvas.yview_scroll(delta, "units")

    def _build_panic_overlay(self):
        """
        Full-window red overlay that appears when panic is triggered.
        Built using place() geometry manager — this lets us position a
        frame to cover the entire window regardless of the grid layout.

        Initially hidden (place_forget). Revealed by _update_loop when
        panic_active becomes True.
        """
        self._panic_overlay = tk.Frame(
            self.root,
            bg=COLOURS["panic_bg"],
        )
        # We don't call .place() yet — we do that when panic fires.

        # Inner content — centred in the overlay
        inner = tk.Frame(self._panic_overlay, bg=COLOURS["panic_bg"])
        inner.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            inner,
            text="⚠",
            font=("Helvetica", 72),
            bg=COLOURS["panic_bg"],
            fg=COLOURS["panic_text"],
        ).pack()

        tk.Label(
            inner,
            text="A L E R T",
            font=("Helvetica", 56, "bold"),
            bg=COLOURS["panic_bg"],
            fg=COLOURS["panic_text"],
        ).pack()

        tk.Label(
            inner,
            text="I NEED IMMEDIATE HELP",
            font=("Helvetica", 24),
            bg=COLOURS["panic_bg"],
            fg=COLOURS["panic_text"],
        ).pack(pady=(8, 32))

        tk.Button(
            inner,
            text="Dismiss Alert",
            font=("Helvetica", 14),
            bg="#FFFFFF",
            fg=COLOURS["panic_bg"],
            activebackground="#FEE2E2",
            relief="flat",
            padx=20, pady=10,
            cursor="hand2",
            command=self._on_dismiss_panic,
        ).pack()

        self._panic_overlay_shown = False


    # =========================================================================
    # Update loop — runs every 50ms, reads controller state, refreshes UI
    # =========================================================================

    def _update_loop(self):
        """
        The heartbeat of the UI. Called every UPDATE_INTERVAL_MS milliseconds.

        Pattern:
          1. Read state from controller
          2. Update each widget if the relevant value has changed
          3. Schedule the next call

        We only update widgets when their value has changed — this avoids
        unnecessary redraws and keeps the UI responsive.
        """
        state = self.controller._state_snapshot()

        # ── Preview panel ──────────────────────────────────────────────────
        preview = state["preview_text"]
        if preview:
            self._preview_label.config(
                text=preview,
                fg=COLOURS["text_preview"],
            )
        else:
            self._preview_label.config(
                text="Start writing with the pen...",
                fg=COLOURS["text_placeholder"],
            )

        # ── Main display ───────────────────────────────────────────────────
        confirmed = state["confirmed_message"]
        if confirmed != self._last_confirmed:
            self._last_confirmed = confirmed
            if confirmed:
                self._confirmed_label.config(
                    text=confirmed.upper(),
                    fg=COLOURS["text_confirmed"],
                    font=("Helvetica", 36, "bold"),
                )
            else:
                self._confirmed_label.config(
                    text="—",
                    fg=COLOURS["text_placeholder"],
                    font=("Helvetica", 36),
                )

        # ── Conversation history ───────────────────────────────────────────
        history = state["conversation_history"]
        if len(history) > self._history_len:
            # New entries arrived — append them to the Text widget
            self._history_text.config(state="normal")
            for i in range(self._history_len, len(history)):
                ts, msg = history[i]
                self._history_text.insert("end", f"{ts}   ", "timestamp")
                self._history_text.insert("end", f"{msg}\n", "message")
            self._history_text.config(state="disabled")
            self._history_text.see("end")   # scroll to bottom
            self._history_len = len(history)

        # ── Volume bar ─────────────────────────────────────────────────────
        vol = state["volume"]
        bar_width = int(vol * 300)
        self._vol_canvas.coords(self._vol_bar_rect, 0, 0, bar_width, 22)
        self._vol_pct_label.config(text=f"{int(vol * 100)}%")

        # ── Event log ──────────────────────────────────────────────────────
        log = state["event_log"]
        if log:
            # Show the 3 most recent events on one line
            self._log_label.config(text="  |  ".join(log[-3:]))

        # ── Panic overlay ──────────────────────────────────────────────────
        panic = state["panic_active"]
        if panic and not self._panic_overlay_shown:
            self._show_panic_overlay()
        elif not panic and self._panic_overlay_shown:
            self._hide_panic_overlay()

        # Schedule next update
        self.root.after(self.UPDATE_INTERVAL_MS, self._update_loop)


    # =========================================================================
    # Event handlers — called by button clicks
    # =========================================================================

    def _on_confirm(self):
        """Confirm + Speak button pressed."""
        self.controller.confirm_and_speak()

    def _on_clear(self):
        """Clear All button pressed."""
        self.controller.clear_all()

    def _on_volume_change(self, delta):
        """Volume + / - button pressed."""
        self.controller.set_volume(self.controller.volume + delta)

    def _on_quick_phrase(self, phrase):
        """A quick phrase button was clicked."""
        self.controller._broadcast_quick_phrase(phrase)

    def _on_add_phrase(self):
        """Open a dialog to add a new custom phrase."""
        phrase = simpledialog.askstring(
            "Add Phrase",
            "Enter new quick phrase:",
            parent=self.root,
        )
        if phrase and phrase.strip():
            self.quick_phrases.append(phrase.strip())
            self._refresh_phrase_buttons()

    def _on_remove_phrase(self):
        """Remove the last phrase in the list."""
        if len(self.quick_phrases) > 1:
            removed = self.quick_phrases.pop()
            self._refresh_phrase_buttons()
            self.controller._log(f"Removed phrase: \"{removed}\"")
        else:
            messagebox.showinfo(
                "Cannot Remove",
                "At least one quick phrase must remain.",
                parent=self.root,
            )

    def _on_clear_history(self):
        """Clear the conversation history."""
        self.controller.conversation_history.clear()
        self._history_len = 0
        self._history_text.config(state="normal")
        self._history_text.delete("1.0", "end")
        self._history_text.config(state="disabled")
        self.controller._log("Conversation history cleared")

    def _on_panic_btn(self):
        """Simulate panic via UI button."""
        self.controller.trigger_panic_simulation()

    def _on_dismiss_panic(self):
        """Dismiss the panic overlay."""
        self.controller.dismiss_panic()

    def _show_panic_overlay(self):
        """Make the panic overlay cover the entire window."""
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self._panic_overlay.place(x=0, y=0, width=w, height=h)
        self._panic_overlay.lift()   # bring to front above all other widgets
        self._panic_overlay_shown = True

    def _hide_panic_overlay(self):
        """Remove the panic overlay."""
        self._panic_overlay.place_forget()
        self._panic_overlay_shown = False


    # =========================================================================
    # Simulation methods — run in a background thread
    # =========================================================================

    def _sim_write(self, phrase_label):
        """
        Simulate writing a phrase by feeding pen strokes into the controller.
        Runs in a background thread so the UI stays responsive while "writing".
        """
        if self._sim_running:
            return
        letters = DEMO_PHRASES[phrase_label]
        thread = threading.Thread(
            target=self._sim_write_thread,
            args=(letters,),
            daemon=True,
        )
        thread.start()

    def _sim_write_thread(self, letters):
        """
        Background thread: feed one letter at a time into the controller,
        with idle gaps between letters exactly as main_writing.py does.
        """
        self._sim_running = True
        try:
            for letter in letters:
                if letter not in LETTER_TEMPLATES:
                    continue
                # Generate stroke
                pressure       = generate_pressure_writing(0.5)
                gyro_x, gyro_y = _template_to_gyro(letter, 0.5)
                n = min(len(pressure), len(gyro_x))

                for i in range(n):
                    self.controller.process_pen_frame(
                        pressure[i], gyro_x[i], gyro_y[i]
                    )
                    time.sleep(SAMPLE_PERIOD)

                # Idle between letters
                idle = generate_pressure_idle(0.2)
                for p in idle:
                    self.controller.process_pen_frame(p, 0.0, 0.0)
                    time.sleep(SAMPLE_PERIOD)

            # Longer idle to trigger word break detection
            idle = generate_pressure_idle(1.5)
            for p in idle:
                self.controller.process_pen_frame(p, 0.0, 0.0)
                time.sleep(SAMPLE_PERIOD)

        finally:
            self._sim_running = False

    def _sim_double_tap(self):
        """Simulate a double-tap (undo last word)."""
        if self._sim_running:
            return
        threading.Thread(
            target=self._sim_tap_thread, args=(2,), daemon=True
        ).start()

    def _sim_triple_tap(self):
        """Simulate a triple-tap (clear all)."""
        if self._sim_running:
            return
        threading.Thread(
            target=self._sim_tap_thread, args=(3,), daemon=True
        ).start()

    def _sim_five_taps(self):
        """Simulate 5 rapid taps (panic alert)."""
        if self._sim_running:
            return
        threading.Thread(
            target=self._sim_tap_thread, args=(5,), daemon=True
        ).start()

    def _sim_tap_thread(self, n_taps):
        """Feed n tap pressure samples into the controller."""
        self._sim_running = True
        try:
            taps = generate_pressure_light_tap(n_taps=n_taps, tap_gap=0.12)
            for p in taps:
                self.controller.process_pen_frame(p, 0.0, 0.0)
                time.sleep(SAMPLE_PERIOD)
            # Small idle after taps
            idle = generate_pressure_idle(0.2)
            for p in idle:
                self.controller.process_pen_frame(p, 0.0, 0.0)
                time.sleep(SAMPLE_PERIOD)
        finally:
            self._sim_running = False

    def _sim_tilt_up(self):
        """Simulate pen tilting right → volume up."""
        threading.Thread(target=self._sim_tilt_thread, args=(1.5,), daemon=True).start()

    def _sim_tilt_down(self):
        """Simulate pen tilting left → volume down."""
        threading.Thread(target=self._sim_tilt_thread, args=(-1.5,), daemon=True).start()

    def _sim_tilt_thread(self, gyro_z):
        """Feed 80 tilt samples (0.4s worth) into the controller."""
        idle = generate_pressure_idle(0.4)
        for p in idle:
            self.controller.process_pen_frame(p, 0.0, 0.0, gyro_z=gyro_z)
            time.sleep(SAMPLE_PERIOD)

    def _sim_auto_demo(self):
        """
        Auto Demo — writes a complete sentence letter by letter, then stops.

        The sentence is: 'i can nice' — a short readable demo phrase.
        Once writing finishes, the preview panel is full and the demo status
        label tells the user to press Confirm + Speak to hear TTS.

        This is the primary way to test that TTS works correctly:
          1. Click Auto Demo
          2. Watch letters appear in the preview panel
          3. Press Confirm + Speak  → text moves to main display AND is spoken
          4. Click a Quick Phrase  → also spoken (tests repeated TTS calls)
        """
        if self._sim_running:
            self._demo_status.config(text="Already running...")
            return

        # Sentence: 'i can nice' — uses only i, c, a, n, e, i (all available)
        demo_letters = list('i') + list('can') + list('nice')

        def _run():
            self._sim_running = True
            self._demo_btn.config(state="disabled", text="Writing...")
            self._demo_status.config(text="Simulating pen strokes...")
            try:
                for letter in demo_letters:
                    if letter not in LETTER_TEMPLATES:
                        continue
                    # Write the letter
                    pressure       = generate_pressure_writing(0.5)
                    gyro_x, gyro_y = _template_to_gyro(letter, 0.5)
                    n = min(len(pressure), len(gyro_x))
                    for i in range(n):
                        self.controller.process_pen_frame(
                            pressure[i], gyro_x[i], gyro_y[i]
                        )
                        time.sleep(SAMPLE_PERIOD)

                    # Brief lift between letters
                    idle = generate_pressure_idle(0.2)
                    for p in idle:
                        self.controller.process_pen_frame(p, 0.0, 0.0)
                        time.sleep(SAMPLE_PERIOD)

                # Longer idle to trigger word-break on the last word
                idle = generate_pressure_idle(1.5)
                for p in idle:
                    self.controller.process_pen_frame(p, 0.0, 0.0)
                    time.sleep(SAMPLE_PERIOD)

                # Update status — tell user what to do next
                self._demo_status.config(
                    text="✓ Done — press Confirm + Speak!"
                )
            finally:
                self._sim_running = False
                self._demo_btn.config(state="normal", text="▶ Auto Demo (write sentence)")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()

    # Set a comfortable starting size — not maximised, but spacious
    root.geometry("980x980")
    root.resizable(True, True)

    app = CommunicationApp(root)
    root.mainloop()