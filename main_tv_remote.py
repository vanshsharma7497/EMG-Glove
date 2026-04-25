"""
main_tv_remote.py
------------------
TV Remote Mode — simulated TV interface for the EMG Smart Glove prototype.

This is the entry point. Run this file to launch the TV UI:
    python main_tv_remote.py

What this file does
--------------------
1. Creates the TVCommandSimulator (TV state + fake phone layer)
2. Creates the TVRemoteController (gesture → command translator)
3. Starts the EMG + IMU simulation loop in a background thread
4. Builds the Tkinter UI and runs it

Layout
------
  Left (main area, 80%):  TV screen — Home Screen or Player Screen
  Right (dev panel, 20%): Last gesture, last command, command log

Home Screen:
  Header bar — title + volume/mute indicator
  3×3 content grid — tiles with highlight cursor
  Overlay — menu/info popup when triggered

Player Screen:
  Header bar — title + back button hint
  Now Playing — content title + genre
  Progress bar — animated, updates every second
  Playback status — play/pause indicator
  Volume bar
  Controls legend — gesture reference card

Simulation
----------
The EMG + IMU loop runs in a background thread (same pattern as main.py).
Buttons in the dev panel let you fire gestures manually for testing.

Run from the GLOVE/ root directory:
    python main_tv_remote.py
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont

# ── Project imports ───────────────────────────────────────────────────────────
# tv_simulator.py and tv_remote_mode.py live at the root level
from tv_simulator import TVCommandSimulator
from modes.tv_remote_mode import TVRemoteController

# ── Colours ───────────────────────────────────────────────────────────────────
# Dark TV-style palette — looks like a real TV interface, not a dev dashboard

C = {
    # Backgrounds
    "tv_bg"          : "#0A0A0A",    # very dark — TV background
    "header_bg"      : "#111111",    # slightly lighter — header bar
    "tile_bg"        : "#1A1A2E",    # dark blue — content tile
    "tile_selected"  : "#16213E",    # brighter tile for highlighted
    "tile_border"    : "#0F3460",    # border colour for normal tile
    "tile_sel_border": "#E94560",    # bright red/pink — selected tile border
    "player_bg"      : "#0D0D0D",    # player screen background
    "progress_track" : "#2A2A2A",    # progress bar background
    "progress_fill"  : "#E94560",    # progress bar fill — matches accent
    "volume_track"   : "#2A2A2A",
    "volume_fill"    : "#4CAF50",    # green volume bar
    "overlay_bg"     : "#1A1A2E",    # menu/info overlay background
    "dev_bg"         : "#111827",    # dev panel background

    # Text
    "text_primary"   : "#FFFFFF",
    "text_secondary" : "#AAAAAA",
    "text_accent"    : "#E94560",    # bright accent — titles, highlights
    "text_genre"     : "#7B8CDE",    # genre label colour
    "text_dev"       : "#6EE7B7",    # green — dev panel text
    "text_cmd"       : "#FCD34D",    # yellow — command names

    # Buttons (dev panel)
    "btn_bg"         : "#374151",
    "btn_hover"      : "#4B5563",
    "btn_text"       : "#FFFFFF",
}

# ── Fonts — defined once, used everywhere ─────────────────────────────────────
# We use a lambda so fonts are created after Tk() is initialised

def make_fonts():
    return {
        "header"      : tkfont.Font(family="Helvetica", size=16, weight="bold"),
        "tile_title"  : tkfont.Font(family="Helvetica", size=11, weight="bold"),
        "tile_genre"  : tkfont.Font(family="Helvetica", size=9),
        "now_playing" : tkfont.Font(family="Helvetica", size=28, weight="bold"),
        "subtitle"    : tkfont.Font(family="Helvetica", size=14),
        "label"       : tkfont.Font(family="Helvetica", size=11),
        "small"       : tkfont.Font(family="Helvetica", size=9),
        "dev_title"   : tkfont.Font(family="Courier",   size=10, weight="bold"),
        "dev_text"    : tkfont.Font(family="Courier",   size=9),
        "time_display": tkfont.Font(family="Courier",   size=13, weight="bold"),
        "overlay"     : tkfont.Font(family="Helvetica", size=13, weight="bold"),
    }


# ── UI Update Rate ────────────────────────────────────────────────────────────
UPDATE_MS = 100    # refresh UI every 100ms (10fps — plenty for a TV interface)

# ── Simulation loop rate ──────────────────────────────────────────────────────
SIM_INTERVAL = 0.05   # 50ms per EMG frame (20 fps)


# =============================================================================
# TVRemoteApp — the main window
# =============================================================================

class TVRemoteApp:
    """
    Full TV Remote Mode UI.

    Structure
    ---------
    __init__          : window + layout setup, create TV + controller
    _build_left       : main TV area (contains home + player frames)
    _build_home       : home screen — header + tile grid
    _build_player     : player screen — header + now playing + controls
    _build_dev_panel  : right-side developer panel
    _update_loop      : runs every UPDATE_MS — reads tv state, refreshes UI
    _show_home/player : switches which screen is visible
    _draw_grid        : redraws the tile grid canvas
    _draw_progress    : redraws the progress bar canvas
    _draw_volume      : redraws the volume bar canvas
    _sim_gesture      : manually fires a gesture for testing
    """

    def __init__(self, root):
        self.root = root
        self.root.title("EMG Smart Glove — TV Remote Mode")
        self.root.configure(bg=C["tv_bg"])
        self.root.geometry("1100x680")
        self.root.minsize(900, 600)

        # ── Create TV simulator and controller ────────────────────────────
        self.tv = TVCommandSimulator()
        self.controller = TVRemoteController(self.tv)

        # ── Fonts ─────────────────────────────────────────────────────────
        self.fonts = make_fonts()

        # ── Simulation state ──────────────────────────────────────────────
        self._last_screen  = None   # track screen changes to trigger swap

        # ── Build layout ──────────────────────────────────────────────────
        # Two columns: TV area (weight 4) + dev panel (weight 1)
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_left()
        self._build_dev_panel()

        # ── Show initial screen ───────────────────────────────────────────
        self._show_home()

        # ── Start update loop ─────────────────────────────────────────────
        self._update_loop()

        print("TV Remote UI started. Use the dev panel buttons to test gestures.")


    # =========================================================================
    # Layout builders
    # =========================================================================

    def _build_left(self):
        """Main TV area — a container that holds both screens as stacked frames."""
        self.left = tk.Frame(self.root, bg=C["tv_bg"])
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.rowconfigure(0, weight=1)
        self.left.columnconfigure(0, weight=1)

        # Both screens are children of left, placed in same grid cell.
        # Only one is visible at a time — controlled by _show_home/_show_player.
        self._build_home()
        self._build_player()


    # ── Home Screen ───────────────────────────────────────────────────────────

    def _build_home(self):
        """Home screen: header bar + 3×3 content grid."""
        self.home_frame = tk.Frame(self.left, bg=C["tv_bg"])
        self.home_frame.grid(row=0, column=0, sticky="nsew")
        self.home_frame.rowconfigure(1, weight=1)
        self.home_frame.columnconfigure(0, weight=1)

        # ── Header bar ────────────────────────────────────────────────────
        header = tk.Frame(self.home_frame, bg=C["header_bg"], height=52)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.columnconfigure(1, weight=1)

        tk.Label(header, text="📺  EMG Glove TV",
                 font=self.fonts["header"],
                 fg=C["text_accent"], bg=C["header_bg"],
                 padx=16).grid(row=0, column=0, sticky="w", pady=10)

        # Volume + mute indicator (right side of header)
        self.home_vol_label = tk.Label(header, text="",
                                       font=self.fonts["label"],
                                       fg=C["text_secondary"], bg=C["header_bg"],
                                       padx=16)
        self.home_vol_label.grid(row=0, column=2, sticky="e", pady=10)

        # ── Content grid canvas ───────────────────────────────────────────
        # We use a Canvas so we can draw rectangles with custom borders.
        # The grid is redrawn every update cycle.
        grid_container = tk.Frame(self.home_frame, bg=C["tv_bg"], padx=20, pady=20)
        grid_container.grid(row=1, column=0, sticky="nsew")
        grid_container.rowconfigure(0, weight=1)
        grid_container.columnconfigure(0, weight=1)

        self.grid_canvas = tk.Canvas(grid_container,
                                     bg=C["tv_bg"],
                                     highlightthickness=0)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")

        # Bind resize so grid redraws when window is resized
        self.grid_canvas.bind("<Configure>", lambda e: self._draw_grid())

        # ── Overlay frame (menu / info) ───────────────────────────────────
        # Placed over the grid when show_menu or show_info is True.
        self.overlay_frame = tk.Frame(self.home_frame,
                                      bg=C["overlay_bg"],
                                      padx=30, pady=20)
        # Not visible initially — shown by _update_loop when needed

        tk.Label(self.overlay_frame, text="⚙  OPTIONS",
                 font=self.fonts["overlay"],
                 fg=C["text_accent"], bg=C["overlay_bg"]).pack(anchor="w")

        self.overlay_body = tk.Label(self.overlay_frame,
                                     text="",
                                     font=self.fonts["label"],
                                     fg=C["text_primary"], bg=C["overlay_bg"],
                                     justify="left")
        self.overlay_body.pack(anchor="w", pady=(10, 0))

        tk.Label(self.overlay_frame,
                 text="[ thumb_middle_tap = back to dismiss ]",
                 font=self.fonts["small"],
                 fg=C["text_secondary"], bg=C["overlay_bg"]).pack(anchor="w", pady=(16,0))


    # ── Player Screen ─────────────────────────────────────────────────────────

    def _build_player(self):
        """Player screen: now playing info + progress bar + controls legend."""
        self.player_frame = tk.Frame(self.left, bg=C["player_bg"])
        self.player_frame.grid(row=0, column=0, sticky="nsew")
        self.player_frame.columnconfigure(0, weight=1)
        self.player_frame.rowconfigure(2, weight=1)

        # ── Header bar ────────────────────────────────────────────────────
        p_header = tk.Frame(self.player_frame, bg=C["header_bg"], height=52)
        p_header.grid(row=0, column=0, sticky="ew")
        p_header.grid_propagate(False)
        p_header.columnconfigure(1, weight=1)

        tk.Label(p_header, text="▶  Now Playing",
                 font=self.fonts["header"],
                 fg=C["text_accent"], bg=C["header_bg"],
                 padx=16).grid(row=0, column=0, sticky="w", pady=10)

        tk.Label(p_header, text="thumb_middle_tap = back",
                 font=self.fonts["small"],
                 fg=C["text_secondary"], bg=C["header_bg"],
                 padx=16).grid(row=0, column=2, sticky="e", pady=10)

        # ── Now Playing info ──────────────────────────────────────────────
        info_frame = tk.Frame(self.player_frame, bg=C["player_bg"], pady=30)
        info_frame.grid(row=1, column=0, sticky="ew", padx=40)

        self.player_title_label = tk.Label(info_frame,
                                           text="",
                                           font=self.fonts["now_playing"],
                                           fg=C["text_primary"],
                                           bg=C["player_bg"])
        self.player_title_label.pack(anchor="w")

        self.player_genre_label = tk.Label(info_frame,
                                           text="",
                                           font=self.fonts["subtitle"],
                                           fg=C["text_genre"],
                                           bg=C["player_bg"])
        self.player_genre_label.pack(anchor="w", pady=(4, 0))

        # ── Progress bar ──────────────────────────────────────────────────
        progress_frame = tk.Frame(self.player_frame, bg=C["player_bg"], padx=40)
        progress_frame.grid(row=2, column=0, sticky="ew")

        # Time labels
        time_row = tk.Frame(progress_frame, bg=C["player_bg"])
        time_row.pack(fill="x")

        self.time_elapsed = tk.Label(time_row, text="00:00",
                                     font=self.fonts["time_display"],
                                     fg=C["text_primary"], bg=C["player_bg"])
        self.time_elapsed.pack(side="left")

        self.play_status_label = tk.Label(time_row, text="",
                                          font=self.fonts["label"],
                                          fg=C["text_accent"], bg=C["player_bg"])
        self.play_status_label.pack(side="left", padx=20)

        self.time_total = tk.Label(time_row, text="00:00",
                                   font=self.fonts["time_display"],
                                   fg=C["text_secondary"], bg=C["player_bg"])
        self.time_total.pack(side="right")

        # Progress bar canvas
        self.progress_canvas = tk.Canvas(progress_frame,
                                         bg=C["player_bg"],
                                         height=12,
                                         highlightthickness=0)
        self.progress_canvas.pack(fill="x", pady=(8, 0))
        self.progress_canvas.bind("<Configure>", lambda e: self._draw_progress())

        # ── Volume bar ────────────────────────────────────────────────────
        vol_frame = tk.Frame(self.player_frame, bg=C["player_bg"], padx=40)
        vol_frame.grid(row=3, column=0, sticky="ew", pady=(20, 0))

        vol_row = tk.Frame(vol_frame, bg=C["player_bg"])
        vol_row.pack(fill="x")

        self.vol_icon_label = tk.Label(vol_row, text="🔊",
                                       font=self.fonts["label"],
                                       fg=C["text_secondary"], bg=C["player_bg"])
        self.vol_icon_label.pack(side="left")

        self.vol_pct_label = tk.Label(vol_row, text="",
                                      font=self.fonts["label"],
                                      fg=C["text_primary"], bg=C["player_bg"],
                                      padx=8)
        self.vol_pct_label.pack(side="left")

        self.volume_canvas = tk.Canvas(vol_frame,
                                       bg=C["player_bg"],
                                       height=8,
                                       highlightthickness=0)
        self.volume_canvas.pack(fill="x", pady=(6, 0))
        self.volume_canvas.bind("<Configure>", lambda e: self._draw_volume())

        # ── Controls legend ───────────────────────────────────────────────
        legend_frame = tk.Frame(self.player_frame, bg=C["player_bg"], padx=40)
        legend_frame.grid(row=4, column=0, sticky="ew", pady=(24, 0))

        tk.Label(legend_frame, text="GESTURE CONTROLS",
                 font=self.fonts["small"],
                 fg=C["text_secondary"], bg=C["player_bg"]).grid(
                 row=0, column=0, columnspan=4, sticky="w", pady=(0,8))

        controls = [
            ("thumb_index_tap ×2", "Play / Pause"),
            ("thumb_middle_tap",   "← Back"),
            ("thumb_ring_tap",     "Vol +"),
            ("thumb_pinky_tap",    "Vol -"),
            ("index_fold",         "Mute"),
            ("thumb_ring_tap ×2",  "Skip +10s"),
            ("thumb_pinky_tap ×2", "Skip -10s"),
            ("flex",               "Stop"),
        ]

        for i, (gesture, action) in enumerate(controls):
            col = (i % 4) * 2
            row = 1 + (i // 4)
            tk.Label(legend_frame,
                     text=f"{gesture}",
                     font=self.fonts["small"],
                     fg=C["text_cmd"], bg=C["player_bg"],
                     anchor="w").grid(row=row, column=col, sticky="w", padx=(0,4))
            tk.Label(legend_frame,
                     text=f"→ {action}",
                     font=self.fonts["small"],
                     fg=C["text_secondary"], bg=C["player_bg"],
                     anchor="w").grid(row=row, column=col+1, sticky="w", padx=(0,20))


    # ── Dev Panel ─────────────────────────────────────────────────────────────

    def _build_dev_panel(self):
        """Right-side developer panel — gesture buttons + command log."""
        self.dev_panel = tk.Frame(self.root, bg=C["dev_bg"])
        self.dev_panel.grid(row=0, column=1, sticky="nsew")
        self.dev_panel.columnconfigure(0, weight=1)

        tk.Label(self.dev_panel,
                 text="DEV PANEL",
                 font=self.fonts["dev_title"],
                 fg=C["text_dev"], bg=C["dev_bg"],
                 pady=8).pack(fill="x")

        # ── Last gesture + command ────────────────────────────────────────
        tk.Label(self.dev_panel, text="Last gesture:",
                 font=self.fonts["dev_text"],
                 fg=C["text_secondary"], bg=C["dev_bg"]).pack(anchor="w", padx=8)

        self.last_gesture_label = tk.Label(self.dev_panel, text="—",
                                           font=self.fonts["dev_text"],
                                           fg=C["text_dev"], bg=C["dev_bg"],
                                           wraplength=160, justify="left")
        self.last_gesture_label.pack(anchor="w", padx=8, pady=(0,4))

        tk.Label(self.dev_panel, text="Last command:",
                 font=self.fonts["dev_text"],
                 fg=C["text_secondary"], bg=C["dev_bg"]).pack(anchor="w", padx=8)

        self.last_cmd_label = tk.Label(self.dev_panel, text="—",
                                       font=self.fonts["dev_text"],
                                       fg=C["text_cmd"], bg=C["dev_bg"],
                                       wraplength=160, justify="left")
        self.last_cmd_label.pack(anchor="w", padx=8, pady=(0,8))

        # Divider
        tk.Frame(self.dev_panel, bg="#374151", height=1).pack(fill="x", padx=8)

        # ── Gesture buttons ───────────────────────────────────────────────
        tk.Label(self.dev_panel, text="FIRE GESTURE:",
                 font=self.fonts["dev_text"],
                 fg=C["text_secondary"], bg=C["dev_bg"],
                 pady=6).pack(anchor="w", padx=8)

        # All 16 classifier gestures — assigned ones fire TV commands,
        # unassigned ones are silently ignored (useful to verify in testing)
        gesture_buttons = [
            "thumb_index_tap",
            "thumb_middle_tap",
            "thumb_ring_tap",
            "thumb_pinky_tap",
            "index_ring_tap",
            "middle_pinky_tap",
            "victory",
            "thumbs_up",
            "flex",
            "index_fold",
            "index_double_tap",
            "middle_double_tap",
            "ring_double_tap",
            "fist",
            "spiderman",
        ]

        for gesture in gesture_buttons:
            # Shorten label to fit the narrow panel
            label = gesture.replace("_tap", "↓").replace("_", " ")
            btn = tk.Button(self.dev_panel,
                            text=label,
                            font=self.fonts["dev_text"],
                            fg=C["btn_text"], bg=C["btn_bg"],
                            activebackground=C["btn_hover"],
                            relief="flat", cursor="hand2",
                            command=lambda g=gesture: self._sim_gesture(g))
            btn.pack(fill="x", padx=8, pady=1)

        # Navigation buttons
        tk.Frame(self.dev_panel, bg="#374151", height=1).pack(fill="x", padx=8, pady=4)
        tk.Label(self.dev_panel, text="NAVIGATE:",
                 font=self.fonts["dev_text"],
                 fg=C["text_secondary"], bg=C["dev_bg"]).pack(anchor="w", padx=8)

        nav_frame = tk.Frame(self.dev_panel, bg=C["dev_bg"])
        nav_frame.pack(padx=8, pady=2)

        tk.Button(nav_frame, text="↑", width=3,
                  font=self.fonts["dev_text"],
                  fg=C["btn_text"], bg=C["btn_bg"],
                  relief="flat",
                  command=lambda: self.tv.send_command("nav_up")).grid(row=0, column=1)
        tk.Button(nav_frame, text="←", width=3,
                  font=self.fonts["dev_text"],
                  fg=C["btn_text"], bg=C["btn_bg"],
                  relief="flat",
                  command=lambda: self.tv.send_command("nav_left")).grid(row=1, column=0)
        tk.Button(nav_frame, text="→", width=3,
                  font=self.fonts["dev_text"],
                  fg=C["btn_text"], bg=C["btn_bg"],
                  relief="flat",
                  command=lambda: self.tv.send_command("nav_right")).grid(row=1, column=2)
        tk.Button(nav_frame, text="↓", width=3,
                  font=self.fonts["dev_text"],
                  fg=C["btn_text"], bg=C["btn_bg"],
                  relief="flat",
                  command=lambda: self.tv.send_command("nav_down")).grid(row=2, column=1)

        # ── Command log ───────────────────────────────────────────────────
        tk.Frame(self.dev_panel, bg="#374151", height=1).pack(fill="x", padx=8, pady=4)
        tk.Label(self.dev_panel, text="COMMAND LOG:",
                 font=self.fonts["dev_text"],
                 fg=C["text_secondary"], bg=C["dev_bg"]).pack(anchor="w", padx=8)

        self.log_text = tk.Text(self.dev_panel,
                                font=self.fonts["dev_text"],
                                fg=C["text_dev"],
                                bg="#0D1117",
                                relief="flat",
                                state="disabled",
                                wrap="word",
                                height=12)
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(2, 8))


    # =========================================================================
    # Screen switching
    # =========================================================================

    def _show_home(self):
        """Bring home screen to front, hide player screen."""
        self.player_frame.grid_remove()
        self.home_frame.grid(row=0, column=0, sticky="nsew")
        self._draw_grid()

    def _show_player(self):
        """Bring player screen to front, hide home screen."""
        self.home_frame.grid_remove()
        self.player_frame.grid(row=0, column=0, sticky="nsew")


    # =========================================================================
    # Canvas drawing
    # =========================================================================

    def _draw_grid(self):
        """
        Redraws the entire 3×3 content tile grid on the canvas.

        Called every update cycle (100ms) AND on window resize.
        Each tile is a rounded rectangle with title + genre text.
        The selected tile gets a bright coloured border.
        """
        canvas = self.grid_canvas
        canvas.delete("all")

        w = canvas.winfo_width()
        h = canvas.winfo_height()

        if w < 10 or h < 10:
            return   # canvas not yet sized — skip

        # Tile layout
        cols = 3
        rows = 3
        pad  = 12     # gap between tiles
        tile_w = (w - (cols + 1) * pad) / cols
        tile_h = (h - (rows + 1) * pad) / rows

        for row in range(rows):
            for col in range(cols):
                # Pixel position of this tile's top-left corner
                x1 = pad + col * (tile_w + pad)
                y1 = pad + row * (tile_h + pad)
                x2 = x1 + tile_w
                y2 = y1 + tile_h

                is_selected = (row == self.tv.cursor_row and
                               col == self.tv.cursor_col)

                # Tile background
                fill   = C["tile_selected"] if is_selected else C["tile_bg"]
                border = C["tile_sel_border"] if is_selected else C["tile_border"]
                bwidth = 3 if is_selected else 1

                canvas.create_rectangle(x1, y1, x2, y2,
                                        fill=fill,
                                        outline=border,
                                        width=bwidth)

                # Content info
                idx = row * cols + col
                if idx < len(self.tv.content_grid[row]):
                    content = self.tv.content_grid[row][col]
                    title   = content["title"]
                    genre   = content["genre"]
                    dur     = self.tv.get_time_string(content["duration"])

                    cx = (x1 + x2) / 2   # centre x of tile
                    cy = (y1 + y2) / 2   # centre y of tile

                    # Selected tile gets a small "▶ SELECT" hint at top
                    if is_selected:
                        canvas.create_text(cx, y1 + 10,
                                           text="▶  SELECT",
                                           font=self.fonts["small"],
                                           fill=C["text_accent"],
                                           anchor="n")

                    # Title
                    canvas.create_text(cx, cy - 14,
                                       text=title,
                                       font=self.fonts["tile_title"],
                                       fill=C["text_primary"],
                                       anchor="center",
                                       width=tile_w - 16)

                    # Genre
                    canvas.create_text(cx, cy + 10,
                                       text=genre,
                                       font=self.fonts["tile_genre"],
                                       fill=C["text_genre"],
                                       anchor="center")

                    # Duration
                    canvas.create_text(cx, cy + 26,
                                       text=dur,
                                       font=self.fonts["small"],
                                       fill=C["text_secondary"],
                                       anchor="center")


    def _draw_progress(self):
        """Redraws the playback progress bar."""
        canvas = self.progress_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4:
            return

        # Track (background)
        canvas.create_rectangle(0, 0, w, h,
                                 fill=C["progress_track"],
                                 outline="")

        # Fill (progress)
        filled = int(w * self.tv.progress_fraction)
        if filled > 0:
            canvas.create_rectangle(0, 0, filled, h,
                                     fill=C["progress_fill"],
                                     outline="")

        # Playhead dot
        if filled > 0:
            canvas.create_oval(filled - 6, -2, filled + 6, h + 2,
                               fill=C["progress_fill"],
                               outline=C["text_primary"],
                               width=1)


    def _draw_volume(self):
        """Redraws the volume bar."""
        canvas = self.volume_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 4:
            return

        canvas.create_rectangle(0, 0, w, h,
                                 fill=C["volume_track"], outline="")

        fraction = self.tv.volume / 100
        filled   = int(w * fraction)
        if filled > 0:
            canvas.create_rectangle(0, 0, filled, h,
                                     fill=C["volume_fill"], outline="")


    # =========================================================================
    # Update loop — runs every UPDATE_MS milliseconds
    # =========================================================================

    def _update_loop(self):
        """
        The heartbeat of the UI. Called every 100ms by Tkinter's after().

        Reads current TV state and updates every widget that might have changed.
        This is simpler than event-driven updates because we don't need to
        track which state changed — we just redraw everything every tick.
        At 100ms intervals this is imperceptible to the user.
        """
        tv = self.tv

        # ── Screen switch ─────────────────────────────────────────────────
        if tv.current_screen != self._last_screen:
            if tv.current_screen == "home":
                self._show_home()
            else:
                self._show_player()
            self._last_screen = tv.current_screen

        # ── Home screen updates ───────────────────────────────────────────
        if tv.current_screen == "home":
            # Volume indicator in header
            mute_str = "🔇 MUTED" if tv.is_muted else f"🔊 {tv.volume}%"
            self.home_vol_label.config(text=mute_str)

            # Redraw tile grid (cursor may have moved)
            self._draw_grid()

            # Overlay visibility
            if tv.show_menu or tv.show_info:
                if tv.show_menu:
                    self.overlay_body.config(
                        text="• thumb_index_tap  →  Select\n"
                             "• victory          →  Dismiss\n"
                             "• thumb_ring_tap   →  Volume +\n"
                             "• thumb_pinky_tap  →  Volume -\n"
                             "• index_fold       →  Mute\n"
                             "• flex             →  Stop"
                    )
                else:
                    content = tv.content_grid[tv.cursor_row][tv.cursor_col]
                    self.overlay_body.config(
                        text=f"Title:    {content['title']}\n"
                             f"Genre:    {content['genre']}\n"
                             f"Duration: {tv.get_time_string(content['duration'])}"
                    )
                # Place overlay over the grid
                self.overlay_frame.place(relx=0.1, rely=0.15,
                                         relwidth=0.8, relheight=0.6)
                self.overlay_frame.lift()
            else:
                self.overlay_frame.place_forget()

        # ── Player screen updates ─────────────────────────────────────────
        if tv.current_screen == "player" and tv.selected_content:
            content = tv.selected_content

            # Title + genre
            self.player_title_label.config(text=content["title"])
            self.player_genre_label.config(text=content["genre"].upper())

            # Play/pause indicator
            status = "▶  PLAYING" if tv.is_playing else "⏸  PAUSED"
            self.play_status_label.config(text=status)

            # Time labels
            self.time_elapsed.config(text=tv.position_str)
            self.time_total.config(text=f"/ {tv.duration_str}")

            # Progress bar
            self._draw_progress()

            # Volume bar + icon
            icon = "🔇" if tv.is_muted else "🔊"
            self.vol_icon_label.config(text=icon)
            vol_text = "MUTED" if tv.is_muted else f"{tv.volume}%"
            self.vol_pct_label.config(text=vol_text)
            self._draw_volume()

        # ── Dev panel updates ─────────────────────────────────────────────
        # Last gesture
        self.last_gesture_label.config(
            text=self.controller.last_gesture or "—")

        # Last command
        self.last_cmd_label.config(
            text=tv.last_command or "—")

        # Command log — rebuild from tv.command_log
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        for ts, cmd in reversed(tv.command_log):   # newest first
            self.log_text.insert("end", f"[{ts}]  {cmd}\n")
        self.log_text.config(state="disabled")

        # ── Schedule next update ──────────────────────────────────────────
        self.root.after(UPDATE_MS, self._update_loop)


    # =========================================================================
    # Simulation helpers
    # =========================================================================

    def _sim_gesture(self, gesture_name):
        """
        Fires a single gesture through the controller, as if the EMG
        classifier had just detected it. Called by dev panel buttons.

        Runs in a short background thread so the button press returns
        immediately and the UI stays responsive.
        """
        def _fire():
            # Brief rest before gesture (simulates finger lifting)
            self.controller.process_frame("rest", 0, 0)
            time.sleep(0.05)
            self.controller.process_frame(gesture_name, 0, 0)
            self.last_gesture_label.config(text=gesture_name)

        threading.Thread(target=_fire, daemon=True).start()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = TVRemoteApp(root)
    root.mainloop()