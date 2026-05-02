"""
modes/communication_mode.py
----------------------------
Communication Mode controller for the EMG Smart Glove.

Designed for non-verbal users who retain hand control. The pen is used
to write sentences; the glove gestures trigger quick phrases, volume
control, and the panic alert.

Architecture
------------
This controller WRAPS WritingController — it owns one internally and
calls process_frame() on it every tick, exactly as the dashboard does.
The key difference: instead of letting WritingController typewrite output
to the OS, we intercept the text and route it to our own state.

We also override tap behaviour:
  WritingController:  double tap = commit word (typewrite)
  CommunicationMode:  double tap = delete last word (undo)
  CommunicationMode:  triple tap = clear everything
  CommunicationMode:  5 taps within 3s = panic alert

This interception works by suppressing tap events before they reach the
WritingController's internal _handle_writing_action(). We do this by
subclassing WritingController and overriding that method.

Two-layer state
---------------
  preview_text      : the sentence being written right now (not yet confirmed)
  confirmed_message : the last message the user pressed Confirm+Speak on

The UI reads both and displays them in separate panels.

Pen context gestures
---------------------
  Write strokes   → letter recognition → preview_text grows
  Double tap      → delete last word from preview_text
  Triple tap      → clear preview_text entirely
  Single tap      → nothing (intentionally ignored)
  Tilt (gyro_z)   → TTS volume up/down
  5 taps / 3s     → panic alert

Glove context gestures (pen docked)
-------------------------------------
  thumb_index_tap   → quick phrase: "Yes"
  thumb_middle_tap  → quick phrase: "No"
  thumb_ring_tap    → quick phrase: "Thank you"
  thumb_pinky_tap   → quick phrase: "I need help"
  index_double_tap  → quick phrase: "Please wait a moment"
  thumbs_up         → volume up
  index_fold        → volume down
  fist              → panic alert

Called by: main_communication.py (the Tkinter UI)
"""

import time
import threading
from collections import deque

from .writing_mode import WritingController
from core.pen_simulator import SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Volume range — pyttsx3 accepts 0.0 to 1.0
VOLUME_MIN  = 0.0
VOLUME_MAX  = 1.0
VOLUME_STEP = 0.08   # how much each tilt tick changes volume

# Tilt threshold — gyro_z below this is treated as "flat" (no volume change).
# Same philosophy as the dead zone in cursor and scroll modes.
TILT_DEAD_ZONE = 0.25   # rad/s

# How much gyro_z beyond the dead zone changes volume per sample.
# Small value because process_frame() is called ~200 times per second —
# a large number would make volume jump to min/max almost instantly.
TILT_VOLUME_RATE = 0.0003   # volume units per sample per rad/s beyond dead zone

# Panic gesture: 5 taps within this time window triggers the alert.
PANIC_TAP_COUNT  = 5
PANIC_TIME_WINDOW = 3.0   # seconds

# Minimum time between any tap event being registered.
# Prevents one physical tap from counting as multiple taps.
TAP_COOLDOWN = 0.3   # seconds

# Quick phrases mapped to glove gestures.
# The UI also displays these as clickable buttons for prototype use.
DEFAULT_QUICK_PHRASES = [
    "Yes",
    "No",
    "Thank you",
    "I need help",
    "Please wait a moment",
]

# Glove gesture → quick phrase mapping.
GLOVE_GESTURE_PHRASES = {
    "thumb_index_tap"  : "Yes",
    "thumb_middle_tap" : "No",
    "thumb_ring_tap"   : "Thank you",
    "thumb_pinky_tap"  : "I need help",
    "index_double_tap" : "Please wait a moment",
}

GLOVE_GESTURE_VOLUME = {
    "thumbs_up"   : +VOLUME_STEP,   # thumb up → volume up
    "index_fold"  : -VOLUME_STEP,   # index fold → volume down
}

GLOVE_GESTURE_PANIC = "fist"   # strong closed fist → panic alert


# ---------------------------------------------------------------------------
# Intercepting subclass — suppresses WritingController's tap output
# ---------------------------------------------------------------------------

class _SilentWritingController(WritingController):
    """
    WritingController with tap actions re-routed to CommunicationController.

    In Writing Mode, double-tap calls pyautogui.typewrite() and triple-tap
    calls pyautogui.press("backspace"). We don't want either of those here.

    Instead of silencing tap actions entirely, we forward them via a callback
    so CommunicationController can handle double-tap (undo) and triple-tap
    (clear). The callback is set by CommunicationController after construction.

    The stroke recognition pipeline (letters, word building, autocorrect)
    is completely unchanged — only the tap output destination changes.
    """

    def __init__(self):
        super().__init__()
        # Set by CommunicationController after construction.
        # Called with the action string: "double_tap", "triple_tap", etc.
        self.on_tap_action = None

    def _handle_writing_action(self, action):
        # Forward to CommunicationController instead of calling pyautogui
        if self.on_tap_action is not None:
            self.on_tap_action(action)

    def _type_word(self, word):
        # Intentionally suppressed — we never typewrite in this mode
        pass


# ---------------------------------------------------------------------------
# CommunicationController
# ---------------------------------------------------------------------------

class CommunicationController:
    """
    Top-level controller for Communication Mode.

    Usage:
        controller = CommunicationController()

        # Each sensor frame (pen context, 200Hz):
        controller.process_pen_frame(pressure, gyro_x, gyro_y, gyro_z)

        # Glove gesture event (pen docked — not every frame, only on gesture):
        controller.process_glove_gesture("thumb_index_tap")

        # User pressed Confirm + Speak button on screen:
        controller.confirm_and_speak()

        # Read state for display:
        controller.preview_text        → str
        controller.confirmed_message   → str
        controller.conversation_history → list of (time_str, message) tuples
        controller.volume              → float 0.0–1.0
        controller.panic_active        → bool
        controller.event_log           → list of str (recent events)
    """

    def __init__(self):
        print("\n  CommunicationController initialising")

        # ── Inner writing pipeline ─────────────────────────────────────────
        # We use the silent subclass so tap events don't reach pyautogui.
        self._writer = _SilentWritingController()
        # Wire the tap callback — now tap actions come here instead of pyautogui
        self._writer.on_tap_action = self._on_tap_action

        # ── Text state ────────────────────────────────────────────────────
        # preview_text is rebuilt from the writing controller's output each
        # frame — it's the live text the user is currently composing.
        # confirmed_message is set only when the user presses Confirm+Speak.
        self.preview_text        = ""
        self.confirmed_message   = ""
        self.conversation_history = []   # list of (time_str, message)

        # ── Volume ────────────────────────────────────────────────────────
        self.volume = 0.7   # start at 70%

        # ── Tap tracking ─────────────────────────────────────────────────
        # We maintain our own tap counter, separate from WritingController,
        # because we've silenced that controller's tap handling.
        self._tap_times    = deque()   # timestamps of recent taps
        self._last_tap_time = 0.0

        # ── Panic state ───────────────────────────────────────────────────
        self.panic_active = False

        # ── TTS ───────────────────────────────────────────────────────────
        # We do NOT keep a persistent engine instance — pyttsx3 is re-created
        # fresh on each speak call. See _speak() for the full explanation.
        # _tts_available is None (unchecked), True, or False.
        self._tts_available = None
        self._tts_busy      = False
        self._tts_thread    = None

        # ── Event log ────────────────────────────────────────────────────
        # Short human-readable log of recent actions, shown in the UI.
        self.event_log = []

        print("  CommunicationController ready\n")


    # =========================================================================
    # Public API — called by the UI each frame
    # =========================================================================

    def process_pen_frame(self, pressure, gyro_x, gyro_y, gyro_z=0.0):
        """
        Called once per sensor sample while pen is undocked (~200Hz).

        pressure       : float 0..1   — pen tip force
        gyro_x, gyro_y : float rad/s  — rolling motion (used for letter strokes)
        gyro_z         : float rad/s  — tilt left/right (used for volume)

        This method:
          1. Feeds the frame into the inner writing controller (letters only)
          2. Reads the resulting text back and stores it as preview_text
          3. Checks for tap events and handles double/triple/panic logic
          4. Applies tilt → volume change
        """
        from core.pen_simulator import PEN_DOWN_THRESHOLD

        # ── 1. Feed into writing pipeline ─────────────────────────────────
        status = self._writer.process_frame(
            pressure, gyro_x, gyro_y, slider_position=2
        )

        # ── 2. Rebuild preview text ───────────────────────────────────────
        # output_text = committed words (each followed by a space).
        # current_word = letters accumulated for the word being written.
        # We combine them carefully to ensure a visible space between the
        # last committed word and the letters being written right now.
        committed = self._writer.language_model.output_text   # e.g. "hello "
        building  = self._writer.language_model.current_word  # e.g. "wor"

        if committed and building:
            # Both present — committed already ends with a space, just join
            self.preview_text = committed + building
        elif committed:
            # Word break just happened — show committed text, trimmed
            self.preview_text = committed.strip()
        elif building:
            # First word, not yet committed
            self.preview_text = building
        else:
            self.preview_text = ""

        # ── 3. Tap detection ──────────────────────────────────────────────
        # Tap actions are handled via the on_tap_action callback wired in
        # __init__. When WritingController classifies a tap gesture, it calls
        # self._writer.on_tap_action(action) which routes here directly.
        # Nothing to do in this frame — callbacks fire asynchronously.

        # ── 4. Volume via tilt ────────────────────────────────────────────
        self._update_volume_from_tilt(gyro_z)

        return self._state_snapshot()


    def process_glove_gesture(self, gesture_name):
        """
        Called when a glove gesture is detected (pen docked context).

        gesture_name : str — one of the keys in GLOVE_GESTURE_PHRASES,
                             GLOVE_GESTURE_VOLUME, or GLOVE_GESTURE_PANIC.

        In the prototype, the UI has buttons that simulate each gesture.
        On real hardware, this is called by ModeManager when the EMG
        classifier returns one of these gesture labels.
        """
        if gesture_name == GLOVE_GESTURE_PANIC:
            self._trigger_panic(f"Glove gesture: {gesture_name}")
            return

        if gesture_name in GLOVE_GESTURE_VOLUME:
            delta = GLOVE_GESTURE_VOLUME[gesture_name]
            self._change_volume(delta)
            return

        if gesture_name in GLOVE_GESTURE_PHRASES:
            phrase = GLOVE_GESTURE_PHRASES[gesture_name]
            self._broadcast_quick_phrase(phrase)
            return

        self._log(f"Unknown gesture: {gesture_name}")


    def confirm_and_speak(self):
        """
        Called when the user presses the Confirm + Speak button on screen.

        Takes whatever is in preview_text, moves it to confirmed_message,
        adds it to conversation history, speaks it aloud, and resets the
        writing pipeline for the next message.
        """
        text = self.preview_text.strip()
        if not text:
            self._log("Confirm pressed — nothing to confirm")
            return

        self.confirmed_message = text
        self._add_to_history(text)
        self._reset_writing_pipeline()
        self._log(f"Confirmed: \"{text}\"")
        self._speak(text)


    def dismiss_panic(self):
        """Called when the user dismisses the panic alert overlay."""
        self.panic_active = False
        self._log("Panic alert dismissed")


    def delete_last_word(self):
        """
        Removes the last committed word from the writing pipeline's output.
        Called by double-tap gesture and exposed publicly so the UI can also
        offer an Undo button.
        """
        lm = self._writer.language_model

        # If there's a word currently being built, clear it first.
        # This is the most natural "undo" — get rid of what you're writing.
        if lm.current_word:
            lm.current_word = ""
            lm.completions  = []
            self._refresh_preview()
            self._log("Deleted current word (in progress)")
            return

        # Otherwise remove the last committed word from output_text.
        # output_text is a space-separated string ending with a space.
        # e.g. "hello world " → we want to remove "world " → "hello "
        words = lm.output_text.strip().split()
        if words:
            words.pop()
            lm.output_text = (" ".join(words) + " ") if words else ""
            self._refresh_preview()
            self._log(f"Deleted last word. Remaining: \"{lm.output_text.strip()}\"")
        else:
            self._log("Nothing to undo")


    def clear_all(self):
        """
        Clears preview_text entirely. Called by triple-tap gesture and
        the Clear All button on screen.
        """
        self._reset_writing_pipeline()
        self._log("Cleared all (triple tap)")


    def set_volume(self, value):
        """Set volume directly (called by UI volume slider). value: 0.0–1.0."""
        self.volume = max(VOLUME_MIN, min(VOLUME_MAX, value))
        self._apply_volume_to_tts()


    def trigger_panic_simulation(self):
        """Called by the UI's 'Simulate Panic' button for prototype demo."""
        self._trigger_panic("Simulated via UI button")


    # =========================================================================
    # Private helpers
    # =========================================================================

    def _on_tap_action(self, action):
        """
        Called by _SilentWritingController when a tap gesture is detected.

        The WritingController's stroke processor classifies tap patterns as
        "tap", "double_tap", or "triple_tap" — we receive those here and
        map them to Communication Mode actions.

        The 5-tap panic counter works differently: we count each classified
        ACTION (not each physical tap), so a double_tap counts as 1 action,
        a triple_tap counts as 1 action. Five separate tap actions in 3s = panic.

        WritingController action → Communication Mode action:
          tap         → nothing (intentional) — 1 panic counter tick
          double_tap  → delete last word      — 1 panic counter tick
          triple_tap  → clear all             — 1 panic counter tick, then reset
        """
        now = time.time()

        # Each classified gesture (regardless of tap count) is one panic tick
        self._tap_times.append(now)
        cutoff = now - PANIC_TIME_WINDOW
        while self._tap_times and self._tap_times[0] < cutoff:
            self._tap_times.popleft()

        action_count = len(self._tap_times)
        self._log(f"Tap action: {action} (panic counter: {action_count}/{PANIC_TAP_COUNT})")

        # Panic overrides everything — check first
        if action_count >= PANIC_TAP_COUNT:
            self._tap_times.clear()
            self._trigger_panic("5-tap gesture")
            return

        # Map to communication mode behaviour — clear counter after acting
        if action == "double_tap":
            self._tap_times.clear()
            self.delete_last_word()
        elif action == "triple_tap":
            self._tap_times.clear()
            self.clear_all()
        # single tap → do nothing (intentional)

    def _update_volume_from_tilt(self, gyro_z):
        """
        Adjusts TTS volume based on pen tilt (gyro_z axis).
        Tilt left  → gyro_z negative → volume down
        Tilt right → gyro_z positive → volume up
        """
        if abs(gyro_z) < TILT_DEAD_ZONE:
            return   # within dead zone — no change

        # Amount beyond dead zone, scaled to a volume delta
        excess = abs(gyro_z) - TILT_DEAD_ZONE
        delta  = excess * TILT_VOLUME_RATE
        if gyro_z < 0:
            delta = -delta

        self._change_volume(delta)

    def _change_volume(self, delta):
        """Apply a volume delta and clamp to valid range."""
        self.volume = max(VOLUME_MIN, min(VOLUME_MAX, self.volume + delta))
        self._apply_volume_to_tts()

    def _apply_volume_to_tts(self):
        """Volume is applied at speak-time — nothing to do here."""
        pass   # fresh engine created per call, volume set inside _speak()/_speak_panic()

    def _broadcast_quick_phrase(self, phrase):
        """
        Immediately confirm and speak a quick phrase, bypassing the writing
        pipeline entirely. This is the same as typing a phrase and pressing
        Confirm+Speak, but in one action.
        """
        self.confirmed_message = phrase
        self._add_to_history(phrase)
        self._log(f"Quick phrase: \"{phrase}\"")
        self._speak(phrase)

    def _add_to_history(self, message):
        """Append a message to the conversation history with a timestamp."""
        timestamp = time.strftime("%H:%M")
        self.conversation_history.append((timestamp, message))

    def _trigger_panic(self, source):
        """Activate the panic alert."""
        self.panic_active = True
        self._log(f"⚠ PANIC ALERT triggered ({source})")
        self._speak_panic()

    def _reset_writing_pipeline(self):
        """Clear the inner writing controller's text state completely."""
        lm = self._writer.language_model
        lm.current_word = ""
        lm.output_text  = ""
        lm.completions  = []
        self._writer.pen_up_samples = 0
        self.preview_text = ""

    def _refresh_preview(self):
        """Recompute preview_text from the language model's current state."""
        committed = self._writer.language_model.output_text
        building  = self._writer.language_model.current_word
        self.preview_text = (committed + building).strip()

    def _log(self, message):
        """Add to the event log shown in the UI."""
        self.event_log.append(message)
        if len(self.event_log) > 12:
            self.event_log.pop(0)
        print(f"  [Comm] {message}")

    def _state_snapshot(self):
        """Return a dict of current state for the UI to read."""
        return {
            "preview_text"         : self.preview_text,
            "confirmed_message"    : self.confirmed_message,
            "conversation_history" : list(self.conversation_history),
            "volume"               : self.volume,
            "panic_active"         : self.panic_active,
            "event_log"            : list(self.event_log),
        }

    # =========================================================================
    # Text-to-speech
    # =========================================================================

    def _check_tts_available(self):
        """
        Returns True if pyttsx3 is importable.
        Does NOT keep an engine instance — we create one fresh per call.

        Why fresh per call?
        pyttsx3's runAndWait() drives an internal OS event loop. On most
        platforms, once that loop exits, the engine's driver is no longer
        in a state that accepts new say() calls. Re-initialising is the
        standard fix and is cheap — pyttsx3 init takes ~50ms.
        """
        if self._tts_available is None:
            try:
                import pyttsx3
                # Quick smoke-test init to confirm the driver works
                engine = pyttsx3.init()
                engine.stop()
                self._tts_available = True
                self._log("TTS engine confirmed available")
            except ImportError:
                self._tts_available = False
                self._log("pyttsx3 not installed — run: pip install pyttsx3")
            except Exception as e:
                self._tts_available = False
                self._log(f"TTS unavailable: {e}")
        return self._tts_available

    def _speak(self, text):
        """
        Speak text aloud. Creates a fresh pyttsx3 engine instance for each
        call — this is the correct pattern for pyttsx3 in a background thread.

        Reusing the same engine across calls causes silent failures on the
        second invocation because runAndWait() leaves the internal driver
        in a closed state. A new init() takes ~50ms and reliably works.
        """
        if not self._check_tts_available():
            return

        if self._tts_busy:
            self._log("TTS busy — previous message still speaking")
            return

        def _run():
            self._tts_busy = True
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("volume", self.volume)
                engine.setProperty("rate", 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                self._log(f"TTS error: {e}")
            finally:
                self._tts_busy = False

        self._tts_thread = threading.Thread(target=_run, daemon=True)
        self._tts_thread.start()

    def _speak_panic(self):
        """
        Speak the panic alert. Ignores _tts_busy — panic always interrupts.
        Also uses a fresh engine instance for the same reason as _speak().
        """
        if not self._check_tts_available():
            return

        def _run():
            self._tts_busy = True
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("volume", 1.0)   # always full volume for alert
                engine.setProperty("rate", 130)     # slightly slower for clarity
                engine.say("Alert. Alert. I need immediate help.")
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                self._log(f"TTS panic error: {e}")
            finally:
                self._tts_busy = False

        self._tts_thread = threading.Thread(target=_run, daemon=True)
        self._tts_thread.start()