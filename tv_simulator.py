"""
tv_simulator.py
----------------
The TV Command Simulator — the "fake phone layer" for the TV Remote prototype.

In a real product, the glove would send commands to a smartphone app over
Bluetooth, and the phone would relay them to the TV over WiFi (e.g. HDMI-CEC,
Google Cast) or IR blaster. This file simulates that entire chain.

Role in the architecture
-------------------------
  TVRemoteController  →  TVCommandSimulator  →  TV UI reads state
  (gesture logic)        (this file)             (Tkinter drawing)

TVCommandSimulator does two things:
  1. Holds all TV state (volume, screen, cursor position, playback, etc.)
  2. Updates that state correctly when send_command() is called

The UI never calls send_command() — it only reads state variables.
The controller never reads state — it only calls send_command().

Content Grid
------------
The home screen shows a 3-column × 3-row grid of content tiles.
Cursor position is tracked as (row, col) — both zero-indexed.
Grid wraps at edges: moving right from column 2 goes to column 0.

Playback
--------
A background thread advances playback_position by 1 every second
when is_playing is True. This makes the progress bar animate
automatically without the UI needing a timer of its own.

Called by: main_tv_remote.py
"""

import time
import threading


# ── Content Library ───────────────────────────────────────────────────────────
# Each entry: (title, genre, duration_in_seconds)
# Duration is intentionally short for demo purposes so progress bar moves visibly.

CONTENT_LIBRARY = [
    ("Interstellar",       "Sci-Fi",    300),
    ("The Dark Knight",    "Action",    280),
    ("Inception",          "Thriller",  260),
    ("Breaking Bad S1",    "Drama",     180),
    ("Planet Earth II",    "Nature",    240),
    ("The Office S3",      "Comedy",    140),
    ("Dune",               "Sci-Fi",    320),
    ("Stranger Things",    "Drama",     200),
    ("Oppenheimer",        "History",   350),
]

# Grid dimensions — must match len(CONTENT_LIBRARY)
GRID_COLS = 3
GRID_ROWS = 3   # 3 × 3 = 9 tiles

# ── Volume Settings ───────────────────────────────────────────────────────────
VOLUME_MIN  = 0
VOLUME_MAX  = 100
VOLUME_STEP = 10    # each volume_up/down command changes by this much

# ── Skip Settings ─────────────────────────────────────────────────────────────
SKIP_SECONDS = 10   # skip_forward / skip_back jumps by this many seconds


class TVCommandSimulator:
    """
    Simulates a TV + smart device (the 'fake phone layer').

    All state is stored as plain attributes so the UI can read them directly:

        tv.volume           → int 0–100
        tv.is_muted         → bool
        tv.current_screen   → "home" or "player"
        tv.cursor_row       → int 0–2
        tv.cursor_col       → int 0–2
        tv.selected_content → dict or None
        tv.is_playing       → bool
        tv.playback_position→ float (seconds)
        tv.show_menu        → bool
        tv.show_info        → bool
        tv.last_command     → str (last command received, for UI log)
        tv.command_log      → list of (timestamp, command) tuples
    """

    def __init__(self):
        # ── Global state ─────────────────────────────────────────────────────
        self.volume         = 50        # start at 50%
        self.is_muted       = False
        self.current_screen = "home"    # "home" or "player"

        # ── Home screen state ─────────────────────────────────────────────────
        self.cursor_row  = 0
        self.cursor_col  = 0
        self.show_menu   = False        # victory gesture → options overlay
        self.show_info   = False        # middle_pinky_tap → info overlay

        # ── Content grid ─────────────────────────────────────────────────────
        # Convert flat library list into a 2D grid for easy row/col access
        self.content_grid = []
        for row in range(GRID_ROWS):
            grid_row = []
            for col in range(GRID_COLS):
                idx = row * GRID_COLS + col
                if idx < len(CONTENT_LIBRARY):
                    title, genre, duration = CONTENT_LIBRARY[idx]
                    grid_row.append({
                        "title":    title,
                        "genre":    genre,
                        "duration": duration,
                    })
            self.content_grid.append(grid_row)

        # ── Player screen state ───────────────────────────────────────────────
        self.selected_content    = None     # dict from content_grid, or None
        self.is_playing          = False
        self.playback_position   = 0.0      # seconds elapsed
        self.playback_duration   = 0        # total seconds of content

        # ── Command log ───────────────────────────────────────────────────────
        self.last_command = ""
        self.command_log  = []              # list of (time_str, command) tuples

        # ── Playback thread ───────────────────────────────────────────────────
        # A daemon thread that ticks every second and advances playback.
        # Daemon = it dies automatically when the main program exits.
        self._playback_thread = threading.Thread(
            target=self._playback_ticker, daemon=True)
        self._playback_thread.start()

        print("  TV Simulator initialized")
        print(f"  Content library: {len(CONTENT_LIBRARY)} titles in {GRID_ROWS}×{GRID_COLS} grid")
        print(f"  Volume: {self.volume}%  |  Screen: {self.current_screen}\n")


    # ── Playback Thread ───────────────────────────────────────────────────────

    def _playback_ticker(self):
        """
        Runs in the background. Every second:
          - If playing: advance playback_position by 1
          - If position reaches duration: auto-stop (content ended)

        This runs independently of the UI refresh rate, so the progress
        bar moves smoothly at 1-second intervals regardless of frame rate.
        """
        while True:
            time.sleep(1.0)
            if self.is_playing and self.selected_content is not None:
                self.playback_position += 1.0
                if self.playback_position >= self.playback_duration:
                    self.playback_position = self.playback_duration
                    self.is_playing = False
                    self._log("content_ended")


    # ── Command Logging ───────────────────────────────────────────────────────

    def _log(self, command):
        """Records a command with timestamp. Keeps last 20 entries."""
        timestamp = time.strftime("%H:%M:%S")
        self.last_command = command
        self.command_log.append((timestamp, command))
        if len(self.command_log) > 20:
            self.command_log.pop(0)


    # ── Main Command Router ───────────────────────────────────────────────────

    def send_command(self, command):
        """
        Receives a command string from TVRemoteController and updates state.

        This is the only public method the controller calls. Think of it
        as the TV's remote-control receiver — it accepts a button press
        and decides what to do based on current state.

        Context-awareness: some commands behave differently depending on
        which screen is active. For example, 'back' on the player screen
        goes to home; 'back' on the home screen does nothing.
        """
        self._log(command)

        # ── Volume commands (work on any screen) ──────────────────────────────
        if command == "volume_up":
            if not self.is_muted:
                self.volume = min(VOLUME_MAX, self.volume + VOLUME_STEP)

        elif command == "volume_down":
            if not self.is_muted:
                self.volume = max(VOLUME_MIN, self.volume - VOLUME_STEP)

        elif command == "mute_toggle":
            self.is_muted = not self.is_muted

        # ── Navigation commands (home screen cursor movement) ──────────────────
        elif command == "nav_right":
            if self.current_screen == "home":
                # Wrap around: col 2 → col 0
                self.cursor_col = (self.cursor_col + 1) % GRID_COLS

        elif command == "nav_left":
            if self.current_screen == "home":
                self.cursor_col = (self.cursor_col - 1) % GRID_COLS

        elif command == "nav_down":
            if self.current_screen == "home":
                self.cursor_row = (self.cursor_row + 1) % GRID_ROWS

        elif command == "nav_up":
            if self.current_screen == "home":
                self.cursor_row = (self.cursor_row - 1) % GRID_ROWS

        # ── Select (context-dependent) ────────────────────────────────────────
        elif command == "select":
            if self.current_screen == "home":
                self._select_content()      # go to player screen
            elif self.current_screen == "player":
                self._toggle_play_pause()   # select on player = play/pause

        # ── Back (context-dependent) ──────────────────────────────────────────
        elif command == "back":
            if self.current_screen == "player":
                self._go_home()
            elif self.current_screen == "home":
                # Dismiss overlay if one is open; otherwise do nothing
                if self.show_menu:
                    self.show_menu = False
                elif self.show_info:
                    self.show_info = False

        # ── Home (always goes to home screen) ─────────────────────────────────
        elif command == "home":
            self._go_home()

        # ── Playback commands (player screen only) ────────────────────────────
        elif command == "play_pause":
            if self.current_screen == "player":
                self._toggle_play_pause()

        elif command == "stop":
            if self.current_screen == "player":
                self.is_playing = False
                self._go_home()

        elif command == "skip_forward":
            if self.current_screen == "player":
                self.playback_position = min(
                    self.playback_duration,
                    self.playback_position + SKIP_SECONDS
                )

        elif command == "skip_back":
            if self.current_screen == "player":
                self.playback_position = max(
                    0,
                    self.playback_position - SKIP_SECONDS
                )

        # ── Overlay commands (home screen) ────────────────────────────────────
        elif command == "menu":
            self.show_menu = not self.show_menu   # toggle
            self.show_info = False                # dismiss info if open

        elif command == "info":
            self.show_info = not self.show_info   # toggle
            self.show_menu = False                # dismiss menu if open

        # ── Internal (fired by playback thread) ───────────────────────────────
        elif command == "content_ended":
            pass   # state already updated by thread; UI will notice is_playing=False


    # ── Private State Transitions ─────────────────────────────────────────────

    def _select_content(self):
        """
        Called when 'select' fires on the home screen.
        Reads whichever tile the cursor is currently on,
        loads it into player state, and switches to player screen.
        """
        content = self.content_grid[self.cursor_row][self.cursor_col]
        self.selected_content  = content
        self.playback_position = 0.0
        self.playback_duration = content["duration"]
        self.is_playing        = True       # auto-play on select
        self.current_screen    = "player"
        print(f"  [TV] Selected: {content['title']} ({content['duration']}s)")

    def _go_home(self):
        """Returns to home screen. Pauses playback but preserves position."""
        self.is_playing     = False
        self.current_screen = "home"
        self.show_menu      = False
        self.show_info      = False

    def _toggle_play_pause(self):
        """Flips is_playing. Only works if content is loaded."""
        if self.selected_content is not None:
            self.is_playing = not self.is_playing


    # ── Helper: readable playback time ───────────────────────────────────────

    def get_time_string(self, seconds):
        """Converts seconds to MM:SS string. Used by the UI for display."""
        seconds = int(seconds)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    @property
    def position_str(self):
        return self.get_time_string(self.playback_position)

    @property
    def duration_str(self):
        return self.get_time_string(self.playback_duration)

    @property
    def progress_fraction(self):
        """Returns playback progress as 0.0 to 1.0. Used by progress bar."""
        if self.playback_duration == 0:
            return 0.0
        return self.playback_position / self.playback_duration


# ── Quick Standalone Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    tv = TVCommandSimulator()

    print("── Test 1: Volume ───────────────────────────────────────")
    print(f"  Start volume: {tv.volume}")
    tv.send_command("volume_up")
    tv.send_command("volume_up")
    print(f"  After 2× volume_up: {tv.volume}")
    tv.send_command("mute_toggle")
    print(f"  After mute_toggle: is_muted={tv.is_muted}")
    tv.send_command("volume_up")
    print(f"  Volume_up while muted (should stay same): {tv.volume}")
    tv.send_command("mute_toggle")
    print(f"  After unmute: is_muted={tv.is_muted}")

    print("\n── Test 2: Grid navigation ──────────────────────────────")
    print(f"  Start: row={tv.cursor_row}, col={tv.cursor_col}")
    tv.send_command("nav_right")
    tv.send_command("nav_right")
    print(f"  After 2× nav_right: col={tv.cursor_col}")
    tv.send_command("nav_down")
    print(f"  After nav_down: row={tv.cursor_row}")
    tv.send_command("nav_right")
    print(f"  After nav_right (should wrap to 0): col={tv.cursor_col}")

    print("\n── Test 3: Select content → player screen ───────────────")
    tv.cursor_row = 0
    tv.cursor_col = 0
    tv.send_command("select")
    print(f"  Screen: {tv.current_screen}")
    print(f"  Content: {tv.selected_content['title']}")
    print(f"  Playing: {tv.is_playing}")
    print(f"  Duration: {tv.duration_str}")

    print("\n── Test 4: Playback controls ────────────────────────────")
    tv.send_command("play_pause")
    print(f"  After play_pause: is_playing={tv.is_playing}")
    tv.send_command("skip_forward")
    print(f"  After skip_forward: position={tv.position_str}")
    tv.send_command("skip_forward")
    tv.send_command("skip_forward")
    print(f"  After 2 more skips: position={tv.position_str}")
    tv.send_command("skip_back")
    print(f"  After skip_back: position={tv.position_str}")

    print("\n── Test 5: Back to home ─────────────────────────────────")
    tv.send_command("back")
    print(f"  Screen: {tv.current_screen}")
    print(f"  is_playing: {tv.is_playing}")

    print("\n── Test 6: Command log ───────────────────────────────────")
    for ts, cmd in tv.command_log:
        print(f"  [{ts}] {cmd}")

    print("\nAll tests complete.")