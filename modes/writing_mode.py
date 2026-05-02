"""
modes/writing_mode.py
----------------------
Writing mode controller — the equivalent of mouse_mode.py for pen input.

Receives pressure + gyro readings each frame and routes them through:
  - StrokeProcessor  : stroke segmentation + (x,y) path reconstruction
  - recognise_letter : template matching → letter
  - LanguageModel    : letter accumulation + autocorrect + word completion

Also handles Slider 1 (pen cursor mode) where the pen replaces the
thumb IMU for cursor movement, and taps become mouse clicks.

Called by ModeManager (or directly in standalone testing) once per frame.
"""

import time
import numpy as np
import pyautogui
from collections import deque

from core.pen_stroke_processor import StrokeProcessor
from core.letter_templates      import recognise_letter
from core.pen_language_model    import LanguageModel
from core.pen_simulator         import (
    PEN_DOWN_THRESHOLD, FIRM_PRESS_THRESHOLD,
    imu_to_cursor_velocity, imu_to_scroll, SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# How many recent pressure samples to keep for tap classification.
# 0.5s * 200Hz = 100 samples — enough to see a double or triple tap.
TAP_WINDOW_SAMPLES = int(0.5 * SAMPLE_RATE)

# Minimum time between tap actions to prevent repeated firing.
TAP_COOLDOWN = 0.4   # seconds

# Confidence threshold — if the letter recogniser is less confident than
# this, we still add the letter but flag it as uncertain in the log.
CONFIDENCE_THRESHOLD = 0.25

# How long (seconds) of pen-up time signals end of a word (not just a stroke).
# Within a word, brief lifts separate strokes of multi-stroke letters.
# A longer pause means the user has finished the word.
WORD_BREAK_DURATION = 1.2   # seconds

# ── Scroll intent state machine ───────────────────────────────────────────────
#
# Scroll is a two-phase gesture:
#   Phase 1 (IDLE → CONFIRMING): pen tip plants on surface and stays still.
#       "Still" = total gyro magnitude below SCROLL_STILL_THRESHOLD.
#       Must remain still for SCROLL_CONFIRM_SAMPLES consecutive samples.
#   Phase 2 (CONFIRMING → SCROLL_READY): confirmation complete.
#       Now gyro_z tilt drives scroll. Mode stays active until pen lifts
#       or pen moves (gyro_x/y exceeds SCROLL_STILL_THRESHOLD).
#
# This prevents casual pen handling or cursor movement from triggering scroll.

SCROLL_STILL_THRESHOLD  = 0.3    # rad/s — total gyro magnitude below this = "still"
SCROLL_CONFIRM_DURATION = 0.5    # seconds the pen must stay still to confirm intent
SCROLL_CONFIRM_SAMPLES  = int(SCROLL_CONFIRM_DURATION * SAMPLE_RATE)  # = 100 samples


# ---------------------------------------------------------------------------
# WritingController
# ---------------------------------------------------------------------------

class WritingController:
    """
    Main controller for pen writing mode.

    Usage (same pattern as MouseController):
        controller = WritingController()
        controller.process_frame(pressure, gyro_x, gyro_y, slider_position)
    """

    # Minimum samples pen must be continuously down before we classify it as
    # a drag (vs a tap that hasn't lifted yet). 100ms * 200Hz = 20 samples.
    DRAG_CONFIRM_SAMPLES = 20

    # Minimum stroke duration to be treated as a written letter rather than a tap.
    # A tap lasts ~80ms. Anything shorter than 150ms is classified as a tap action
    # and routed to _handle_writing_action() instead of the letter recogniser.
    MIN_LETTER_DURATION = 0.15   # seconds

    def __init__(self):
        self.stroke_processor = StrokeProcessor(sample_rate=SAMPLE_RATE)
        self.language_model   = LanguageModel()

        # Rolling pressure window for tap detection
        self.pressure_window  = deque(maxlen=TAP_WINDOW_SAMPLES)

        # Cursor movement accumulator (Slider 1 mode, same as mouse_mode.py)
        self.accum_x = 0.0
        self.accum_y = 0.0

        # Scroll accumulator — tilt scroll fires in integer units like cursor pixels
        self.accum_scroll = 0.0

        # ── Scroll intent state machine ────────────────────────────────────
        # States: "IDLE" → "CONFIRMING" → "SCROLL_READY"
        #   IDLE         : pen moving or up — scroll cannot trigger
        #   CONFIRMING   : pen planted and still, counting down to confirmation
        #   SCROLL_READY : confirmed — gyro_z tilt now drives scroll
        self._scroll_state        = "IDLE"
        self._scroll_still_count  = 0   # consecutive still samples counted so far

        # Drag state — True while mouseDown is held for click-and-drag
        self.dragging = False

        # Counter: how many consecutive samples the pen has been down.
        # Used to distinguish a brief tap from a sustained hold (drag).
        self._pen_down_samples = 0

        # Timing
        self.last_tap_time    = 0.0
        self.last_pen_up_time = time.time()

        # State tracking
        self.pen_was_down     = False
        self.last_action      = None

        # Sample counter for word-break detection (avoids wall-clock timing issues)
        self.pen_up_samples   = 0
        self.word_break_samples = int(WORD_BREAK_DURATION * SAMPLE_RATE)

        # Output log for dashboard display
        self.action_log = []

        print("  WritingController initialized")
        print(f"  Word break threshold : {WORD_BREAK_DURATION}s")
        print(f"  Tap cooldown         : {TAP_COOLDOWN}s")
        print(f"  Confidence threshold : {CONFIDENCE_THRESHOLD}\n")


    # ── Main entry point ──────────────────────────────────────────────────

    def process_frame(self, pressure, gyro_x, gyro_y, slider_position=2, **kwargs):
        """
        Called once per sensor sample.

        pressure         : float 0..1
        gyro_x, gyro_y  : float rad/s
        slider_position  : int 1 (cursor) or 2 (writing)

        Returns a status dict the dashboard can display:
            {
              'mode'         : 'cursor' | 'writing',
              'pen_down'     : bool,
              'current_word' : str,
              'output_text'  : str,
              'completions'  : list,
              'last_action'  : str or None,
              'dx'           : float,
              'dy'           : float,
            }
        """
        if slider_position == 1:
            gyro_z = kwargs.get("gyro_z", 0.0)
            return self._process_cursor_mode(pressure, gyro_x, gyro_y, gyro_z)
        return self._process_writing_mode(pressure, gyro_x, gyro_y)


    # ── Slider 1: cursor mode ─────────────────────────────────────────────

    def _process_cursor_mode(self, pressure, gyro_x, gyro_y, gyro_z=0.0):
        """
        Full pen cursor mode:
          gyro_x/y → cursor movement (rolling)
          gyro_z   → scroll via two-phase intent gesture (see below)
          pressure → tap classification → clicks / drag

        Scroll state machine (prevents accidental scroll during cursor use):
          IDLE         pen is up, or pen is moving → scroll cannot trigger
          CONFIRMING   pen is planted + still for up to 0.5s → intent building
          SCROLL_READY confirmation complete → gyro_z tilt now drives scroll
                       exits back to IDLE when pen lifts or moves again
        """
        pen_is_down = pressure >= PEN_DOWN_THRESHOLD

        # ── Total gyro magnitude — used for "still" detection ──────────────
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        is_still = gyro_mag < SCROLL_STILL_THRESHOLD

        # ── Scroll state machine ───────────────────────────────────────────
        scroll_amount = 0

        if not pen_is_down:
            # Pen is up — always reset to IDLE
            if self._scroll_state != "IDLE":
                self._log("Scroll mode cancelled (pen lifted)")
            self._scroll_state       = "IDLE"
            self._scroll_still_count = 0
            self.accum_scroll        = 0.0

        elif self._scroll_state == "IDLE":
            # Pen just went down (or was already down but moving)
            if is_still:
                # Start counting toward confirmation
                self._scroll_state = "CONFIRMING"
                self._scroll_still_count = 1

        elif self._scroll_state == "CONFIRMING":
            if is_still:
                self._scroll_still_count += 1
                if self._scroll_still_count >= SCROLL_CONFIRM_SAMPLES:
                    # Held still long enough — scroll intent confirmed
                    self._scroll_state = "SCROLL_READY"
                    self._log("Scroll ready ↕")
            else:
                # Pen moved — restart the still countdown
                self._scroll_state       = "IDLE"
                self._scroll_still_count = 0

        elif self._scroll_state == "SCROLL_READY":
            if not is_still and (abs(gyro_x) > SCROLL_STILL_THRESHOLD
                                 or abs(gyro_y) > SCROLL_STILL_THRESHOLD):
                # Pen is rolling (cursor movement) — exit scroll mode
                self._scroll_state       = "IDLE"
                self._scroll_still_count = 0
                self.accum_scroll        = 0.0
                self._log("Scroll mode cancelled (pen moved)")
            else:
                # Still in scroll mode — process tilt
                scroll_amount = self._apply_tilt_scroll(gyro_z)

        # ── Cursor movement (only when not in scroll mode) ─────────────────
        # Suppress cursor movement while SCROLL_READY so the planted pen
        # doesn't jitter the cursor during intentional tilting.
        if self._scroll_state != "SCROLL_READY" and pen_is_down:
            dx, dy = imu_to_cursor_velocity(gyro_x, gyro_y)
            self._move_cursor(dx, dy)
        else:
            dx, dy = 0.0, 0.0

        # ── Pressure / tap handling ────────────────────────────────────────
        self.pressure_window.append(pressure)

        # Release drag when pen lifts
        if self.dragging and not pen_is_down:
            pyautogui.mouseUp(button="left")
            self.dragging = False
            self._log("DRAG END")

        # Suppress tap/drag classification while scroll confirmation is
        # in progress or scroll mode is active — the pen being planted
        # and still for scroll intent must not trigger a drag start.
        scroll_active = self._scroll_state in ("CONFIRMING", "SCROLL_READY")

        if self.pen_was_down and not pen_is_down and not scroll_active:
            self._pen_down_samples = 0
            action = self.stroke_processor.classify_action(
                list(self.pressure_window)
            )
            self._execute_click_action(action)
        elif pen_is_down and not self.dragging and not scroll_active:
            self._pen_down_samples += 1
            if self._pen_down_samples >= self.DRAG_CONFIRM_SAMPLES:
                action = self.stroke_processor.classify_action(
                    list(self.pressure_window)
                )
                if action == "tap_and_hold":
                    self._execute_click_action(action)
        elif not pen_is_down:
            self._pen_down_samples = 0

        self.pen_was_down = pen_is_down

        return {
            "mode"         : "cursor",
            "pen_down"     : pen_is_down,
            "current_word" : "",
            "output_text"  : "",
            "completions"  : [],
            "last_action"  : self.last_action,
            "dx"           : dx,
            "dy"           : dy,
            "scroll"       : scroll_amount,
            "gyro_z"       : gyro_z,
            "gyro_mag"     : gyro_mag,
            "dragging"     : self.dragging,
            "scroll_state" : self._scroll_state,
            "scroll_progress": min(self._scroll_still_count / SCROLL_CONFIRM_SAMPLES, 1.0),
        }


    def _apply_tilt_scroll(self, gyro_z):
        """
        Converts tilt (gyro_z) to integer scroll events with sub-unit accumulation.
        Only called when scroll state machine is in SCROLL_READY.
        Sub-unit accumulation works identically to cursor sub-pixel movement:
          fractional units build up each frame and fire as integers.
        Returns the scroll amount fired this frame (0 if nothing fired).
        """
        raw = imu_to_scroll(gyro_z)
        if raw == 0.0:
            return 0

        self.accum_scroll += raw
        units = int(self.accum_scroll)

        if units != 0:
            self.accum_scroll -= units
            # pyautogui: positive = scroll up, negative = scroll down
            # our convention: gyro_z > 0 = tilt forward = scroll down
            pyautogui.scroll(-units)
            direction = "SCROLL DOWN" if units > 0 else "SCROLL UP"
            self._log(direction)
            return -units

        return 0


    def _move_cursor(self, dx, dy):
        """Sub-pixel accumulation cursor movement — identical to mouse_mode.py."""
        if dx == 0.0 and dy == 0.0:
            return
        self.accum_x += dx
        self.accum_y += dy
        move_x = int(self.accum_x)
        move_y = int(self.accum_y)
        if move_x != 0 or move_y != 0:
            self.accum_x -= move_x
            self.accum_y -= move_y
            current_x, current_y = pyautogui.position()
            screen_w, screen_h   = pyautogui.size()
            new_x = int(np.clip(current_x + move_x, 0, screen_w - 1))
            new_y = int(np.clip(current_y + move_y, 0, screen_h - 1))
            pyautogui.moveTo(new_x, new_y, duration=0)


    def _execute_click_action(self, action):
        """
        Map tap classification to mouse action — Slider 1 only.
        tap_and_hold now properly holds mouseDown for drag operations.
        """
        now = time.time()
        if (now - self.last_tap_time) < TAP_COOLDOWN:
            return

        if action == "tap":
            pyautogui.click(button="left")
            self._log("LEFT CLICK")

        elif action == "double_tap":
            pyautogui.doubleClick(button="left")
            self._log("DOUBLE CLICK")

        elif action == "firm_press":
            pyautogui.click(button="right")
            self._log("RIGHT CLICK")

        elif action == "tap_and_hold":
            # Hold the mouse button down — it stays down until pen lifts
            # _process_cursor_mode watches pen_is_down and calls mouseUp
            if not self.dragging:
                pyautogui.mouseDown(button="left")
                self.dragging = True
                self._log("DRAG START")

        else:
            return

        self.last_action   = action
        self.last_tap_time = now


    def _process_writing_mode(self, pressure, gyro_x, gyro_y):
        self.pressure_window.append(pressure)
        pen_is_down = pressure >= PEN_DOWN_THRESHOLD

        # Feed sample into stroke processor
        stroke = self.stroke_processor.update(pressure, gyro_x, gyro_y)

        if stroke is not None:
            # ── Duration gate ─────────────────────────────────────────────
            # Short strokes are tap gestures, not letters.
            # The tap classifier runs on the pressure window; the stroke
            # path is discarded so it doesn't produce a phantom letter.
            if stroke["duration"] >= WritingController.MIN_LETTER_DURATION:
                self._handle_completed_stroke(stroke)
            else:
                # It was a tap — classify what kind and act on it
                action = self.stroke_processor.classify_action(
                    list(self.pressure_window)
                )
                self._handle_writing_action(action)

        elif self.pen_was_down and not pen_is_down:
            # Pen lifted without completing a stroke yet — reset counter
            self.pen_up_samples = 0

        # Word break detection — counted in samples, not wall-clock time.
        # This works correctly in both real-time and simulation.
        # WORD_BREAK_SAMPLES = 1.2s * 200Hz = 240 samples of pen-up time.
        if not pen_is_down and self.language_model.current_word:
            self.pen_up_samples += 1
            if self.pen_up_samples >= self.word_break_samples:
                self._log("Word break detected")
                self.language_model.commit_word()
                self.pen_up_samples = 0
        elif pen_is_down:
            self.pen_up_samples = 0

        self.pen_was_down = pen_is_down

        return {
            "mode"        : "writing",
            "pen_down"    : pen_is_down,
            "current_word": self.language_model.current_word,
            "output_text" : self.language_model.output_text,
            "completions" : self.language_model.completions,
            "last_action" : self.last_action,
            "dx"          : 0.0,
            "dy"          : 0.0,
        }


    def _handle_completed_stroke(self, stroke):
        letter, confidence, _ = recognise_letter(stroke)
        flag = "" if confidence >= CONFIDENCE_THRESHOLD else " [low confidence]"
        self._log(f"Stroke → '{letter}' (conf={confidence:.2f}){flag}")
        self.language_model.add_letter(letter)
        self.last_action = f"letter:{letter}"


    def _handle_writing_action(self, action):
        now = time.time()
        if (now - self.last_tap_time) < TAP_COOLDOWN:
            return
        if action == "double_tap":
            if self.language_model.current_word:
                committed = self.language_model.commit_word()
                self._log(f"COMMIT → '{committed}'")
                self._type_word(committed)
            self.last_tap_time = now
        elif action == "triple_tap":
            self.language_model.backspace()
            self._log("BACKSPACE")
            pyautogui.press("backspace")
            self.last_tap_time = now
        elif action == "firm_press":
            self._log("FIRM PRESS (selection — not yet implemented)")
            self.last_tap_time = now


    def _type_word(self, word):
        if word:
            pyautogui.typewrite(word + " ", interval=0.02)


    def _log(self, message):
        self.action_log.append(message)
        if len(self.action_log) > 8:
            self.action_log.pop(0)
        print(f"  [Writing] {message}")


    @property
    def display_text(self):
        return self.language_model.full_text

    @property
    def current_word(self):
        return self.language_model.current_word

    @property
    def completions(self):
        return self.language_model.completions


# ---------------------------------------------------------------------------
# Self-test — simulates writing "love" and "is" using template-derived gyro
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from core.letter_templates import LETTER_TEMPLATES, _resample
    from core.pen_simulator import (
        generate_pressure_writing, generate_pressure_idle,
        generate_pressure_light_tap, SAMPLE_RATE,
    )
    from core.pen_stroke_processor import STROKE_SCALE

    SAMPLE_PERIOD = 1.0 / SAMPLE_RATE

    def template_to_gyro(letter, duration=0.6):
        """
        Derives realistic gyro values directly from a letter template.
        Reverses the integration step so the stroke processor reconstructs
        a path that closely matches the template — guaranteeing recognition.
        """
        path      = LETTER_TEMPLATES[letter]
        n_samples = int(duration * SAMPLE_RATE)
        resampled = _resample(path, n=n_samples)
        dx = np.diff(resampled[:, 0], prepend=resampled[0, 0])
        dy = np.diff(resampled[:, 1], prepend=resampled[0, 1])
        gyro_x = (dx / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
        gyro_y = (dy / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
        return gyro_x, gyro_y

    def feed_letter(controller, letter, duration=0.5):
        """Feed one complete letter stroke into the controller."""
        print(f"\n  ── Stroke: '{letter}' ──")
        pressure       = generate_pressure_writing(duration)
        gyro_x, gyro_y = template_to_gyro(letter, duration)
        n = min(len(pressure), len(gyro_x))
        for i in range(n):
            controller.process_frame(pressure[i], gyro_x[i], gyro_y[i],
                                     slider_position=2)

    def feed_idle(controller, duration=0.15):
        """Feed idle samples (pen lifted) between letters."""
        pressure = generate_pressure_idle(duration)
        for p in pressure:
            controller.process_frame(p, 0.0, 0.0, slider_position=2)

    # ── Run test ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("  WritingController Self-Test")
    print("=" * 55 + "\n")

    controller = WritingController()

    # Write "love"
    print("Simulating writing 'love'...\n")
    for letter in "love":
        feed_letter(controller, letter)
        feed_idle(controller, 0.2)

    # Double tap to commit
    print("\n  ── Double tap to commit ──")
    tap = generate_pressure_light_tap(n_taps=2)
    for p in tap:
        controller.process_frame(p, 0.0, 0.0, slider_position=2)

    feed_idle(controller, 0.3)

    # Write "is"
    print("\nSimulating writing 'is'...\n")
    for letter in "is":
        feed_letter(controller, letter)
        feed_idle(controller, 0.2)

    # Reset cooldown timer — in the real system, the user pauses naturally
    # between words. In simulation, time.time() doesn't advance between
    # tight-loop samples, so we reset the timer explicitly for the test.
    controller.last_tap_time = 0.0

    print("\n  ── Double tap to commit ──")
    tap = generate_pressure_light_tap(n_taps=2)
    for p in tap:
        controller.process_frame(p, 0.0, 0.0, slider_position=2)

    # ── Results ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  Output text : '{controller.language_model.output_text.strip()}'")
    print(f"  Expected    : 'love is'")
    result = controller.language_model.output_text.strip()
    print(f"  Result      : {'✓ PASS' if result == 'love is' else '✗ FAIL — see log'}")
    print(f"\n  Action log:")
    for entry in controller.action_log:
        print(f"    {entry}")
    print("=" * 55)