"""
modes/tv_remote_mode.py
------------------------
TV Remote Mode controller for the EMG Smart Glove.

Architecture
------------
This file is the TV equivalent of mouse_mode.py. It sits between the
EMG classifier and the TV. Its only job is:

  1. Receive a gesture name + IMU dx/dy every frame (from mode_manager)
  2. Decide what TV command that gesture means in the current context
  3. Send that command to the TVCommandSimulator (the fake phone layer)

It never draws anything to the screen. It never talks to a real TV.
It just translates gestures into command strings.

Gesture → Command Mapping
--------------------------
Single taps:
  thumb_index_tap   → select          (OK / confirm)
  thumb_middle_tap  → back            (go back one screen)
  thumb_ring_tap    → volume_up
  thumb_pinky_tap   → volume_down
  index_ring_tap    → home            (go to home screen)
  middle_pinky_tap  → info            (show info overlay)
  victory           → menu            (options menu)
  thumbs_up         → volume_up       (same as thumb_ring_tap, extra option)
  flex              → stop            (stop playback, return to home)
  index_fold        → mute_toggle

Double taps (same gesture fired twice within 300ms):
  thumb_index_tap × 2  → play_pause
  thumb_middle_tap × 2 → home
  thumb_ring_tap × 2   → skip_forward
  thumb_pinky_tap × 2  → skip_back

IMU navigation (thumb roll → grid movement):
  dx > threshold   → nav_right
  dx < -threshold  → nav_left
  dy > threshold   → nav_down
  dy < -threshold  → nav_up

Called by: main_tv_remote.py (the Tkinter UI entry point)
"""

import time

# ── Gesture → Single Tap Command ─────────────────────────────────────────────
# All 16 classifier gestures are listed explicitly.
# None = unassigned in this mode — gesture is ignored.

SINGLE_TAP_COMMANDS = {
    "rest":               None,
    # ── Assigned ──────────────────────────────────────────────────────────────
    "thumb_index_tap":    "select",
    "thumb_middle_tap":   "back",
    "thumb_ring_tap":     "volume_up",
    "thumb_pinky_tap":    "volume_down",
    "index_ring_tap":     "home",
    "middle_pinky_tap":   "info",
    "victory":            "menu",
    "thumbs_up":          "volume_up",    # same as thumb_ring_tap, extra option
    "flex":               "stop",
    "index_fold":         "mute_toggle",
    # ── Unassigned ────────────────────────────────────────────────────────────
    "index_double_tap":   None,           # raw classifier gesture — ignored here
    "middle_double_tap":  None,
    "ring_double_tap":    None,
    "fist":               None,
    "spiderman":          None,
}

# ── Gesture → Double Tap Command ─────────────────────────────────────────────
# What each gesture does when fired TWICE within DOUBLE_TAP_WINDOW seconds.
# Only gestures listed here have double-tap behaviour.
# Gestures not listed here: double tap = same as single tap.
#
# How double taps work in practice:
#   - thumb_index_tap × 2  → play_pause  (most used in player)
#   - thumb_middle_tap × 2 → home        (back twice = go all the way home)
#   - thumb_ring_tap × 2   → skip_fwd    (vol+ twice = skip forward)
#   - thumb_pinky_tap × 2  → skip_back   (vol- twice = skip back)

DOUBLE_TAP_COMMANDS = {
    "thumb_index_tap":  "play_pause",
    "thumb_middle_tap": "home",
    "thumb_ring_tap":   "skip_forward",
    "thumb_pinky_tap":  "skip_back",
    # Removed: thumb_middle_double_tap, thumb_ring_double_tap, thumb_pinky_double_tap
    # — not in GESTURE_PROFILES; dedicated classifier gestures were never added
}

# ── Timing Constants ──────────────────────────────────────────────────────────
DOUBLE_TAP_WINDOW = 0.35    # seconds — two taps within this time = double tap
GESTURE_COOLDOWN  = 0.30    # seconds — minimum time before same gesture fires again
                             # prevents one physical tap from firing 3-4 times

# ── Navigation Constants ──────────────────────────────────────────────────────
# The IMU produces small dx/dy floats each frame.
# We accumulate them and only fire a nav command when the total crosses this threshold.
# Lower = more sensitive thumb roll. Higher = requires more deliberate movement.
NAV_THRESHOLD = 12.0        # accumulated IMU units before one grid step fires


class TVRemoteController:
    """
    Translates gesture names + IMU movement into TV commands.

    Usage (called by mode_manager or main_tv_remote every frame):
        controller = TVRemoteController(tv_simulator)
        controller.process_frame(gesture_name, dx, dy)

    The tv_simulator is the fake phone layer — it receives command strings
    and updates the TV state accordingly.
    """

    def __init__(self, tv_simulator):
        """
        tv_simulator : TVCommandSimulator instance
                       Must have a .send_command(command_string) method.
        """
        self.tv = tv_simulator

        # ── Gesture tracking ─────────────────────────────────────────────────
        self.last_gesture      = "rest"    # what gesture fired last frame
        self.last_action_time  = 0.0       # when did we last fire an action

        # ── Double tap tracking ───────────────────────────────────────────────
        # To detect a double tap we remember the last tap gesture and when it fired.
        # If the same gesture fires again before DOUBLE_TAP_WINDOW expires → double tap.
        self.last_tap_gesture  = None      # which gesture was the first tap
        self.last_tap_time     = 0.0       # when was the first tap

        # ── IMU accumulator ───────────────────────────────────────────────────
        # We don't fire a nav command every frame — we accumulate dx/dy
        # and fire only when the total crosses NAV_THRESHOLD.
        self.accum_x = 0.0
        self.accum_y = 0.0

        print("  TV Remote controller initialized")
        print(f"  Double-tap window : {DOUBLE_TAP_WINDOW}s")
        print(f"  Nav threshold     : {NAV_THRESHOLD} IMU units\n")


    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate(self, dx, dy):
        """
        Converts IMU thumb roll into grid navigation commands.

        dx > 0 = thumb rolling right  → nav_right
        dx < 0 = thumb rolling left   → nav_left
        dy > 0 = thumb rolling down   → nav_down
        dy < 0 = thumb rolling up     → nav_up

        We accumulate dx/dy across frames and only fire when the
        accumulated total crosses NAV_THRESHOLD. This means:
          - Small tremor movements are ignored (they never reach the threshold)
          - Deliberate rolls reliably fire after a short movement
          - Fast rolls can fire multiple steps in quick succession
        """
        if dx == 0.0 and dy == 0.0:
            return

        self.accum_x += dx
        self.accum_y += dy

        # Check horizontal threshold
        if self.accum_x > NAV_THRESHOLD:
            self.tv.send_command("nav_right")
            self.accum_x -= NAV_THRESHOLD       # keep remainder, don't reset to 0

        elif self.accum_x < -NAV_THRESHOLD:
            self.tv.send_command("nav_left")
            self.accum_x += NAV_THRESHOLD

        # Check vertical threshold
        if self.accum_y > NAV_THRESHOLD:
            self.tv.send_command("nav_down")
            self.accum_y -= NAV_THRESHOLD

        elif self.accum_y < -NAV_THRESHOLD:
            self.tv.send_command("nav_up")
            self.accum_y += NAV_THRESHOLD


    # ── Gesture Execution ─────────────────────────────────────────────────────

    def execute_gesture(self, gesture_name):
        """
        Decides what command to fire for a detected gesture.

        The logic flow:
          1. Ignore 'rest' and gestures not in our mapping.
          2. Check cooldown — don't fire if too soon after last action.
          3. Check if gesture changed (finger lifted and re-tapped).
          4. Check for double tap — same gesture within DOUBLE_TAP_WINDOW?
          5. Fire the appropriate command (double or single).
          6. Update tracking variables.
        """
        # Step 1 — ignore rest and unmapped gestures
        if SINGLE_TAP_COMMANDS.get(gesture_name) is None and \
           gesture_name not in DOUBLE_TAP_COMMANDS:
            self.last_gesture = gesture_name
            return

        # Step 2 & 3 — cooldown + gesture change check
        # (same logic as MouseController — prevents one tap firing 4 times)
        now = time.time()
        gesture_changed  = (gesture_name != self.last_gesture)
        cooldown_passed  = (now - self.last_action_time) > GESTURE_COOLDOWN

        if not (gesture_changed and cooldown_passed):
            self.last_gesture = gesture_name
            return

        # Step 4 — double tap detection
        # Is this the same gesture as the last tap, within the time window?
        time_since_last_tap = now - self.last_tap_time
        is_double_tap = (
            gesture_name == self.last_tap_gesture and
            time_since_last_tap < DOUBLE_TAP_WINDOW and
            gesture_name in DOUBLE_TAP_COMMANDS
        )

        # Step 5 — fire the command
        if is_double_tap:
            command = DOUBLE_TAP_COMMANDS[gesture_name]
            print(f"  → DOUBLE TAP: {gesture_name} → {command}")
            self.tv.send_command(command)
            # Reset tap tracking so a third tap doesn't trigger again
            self.last_tap_gesture = None
            self.last_tap_time    = 0.0

        else:
            command = SINGLE_TAP_COMMANDS.get(gesture_name)
            if command:
                print(f"  → {gesture_name} → {command}")
                self.tv.send_command(command)
            # Record this tap in case a second tap follows quickly
            self.last_tap_gesture = gesture_name
            self.last_tap_time    = now

        # Step 6 — update tracking
        self.last_action_time = now
        self.last_gesture     = gesture_name


    # ── Main Entry Point ──────────────────────────────────────────────────────

    def process_frame(self, gesture_name, dx, dy):
        """
        Called every frame from the main loop.

        gesture_name : string from EMG classifier (e.g. 'thumb_index_tap')
        dx, dy       : floats from IMU thumb simulator
                       positive dx = thumb rolling right
                       positive dy = thumb rolling down

        This is the ONLY method the outside world needs to call.
        It handles navigation and gesture actions together.
        """
        self.navigate(dx, dy)
        self.execute_gesture(gesture_name)


# ── Quick Standalone Test ─────────────────────────────────────────────────────
# Run this file directly to verify the controller logic without any UI.
# It creates a fake TV simulator that just prints commands to the terminal.

if __name__ == "__main__":

    class FakeTVSimulator:
        """Minimal fake — just prints commands so we can see what fires."""
        def send_command(self, cmd):
            print(f"    [TV] received: {cmd}")

    fake_tv = FakeTVSimulator()
    controller = TVRemoteController(fake_tv)

    print("── Test 1: Basic single taps ────────────────────────────")
    for gesture in ["thumb_index_tap", "rest", "thumb_ring_tap",
                    "rest", "victory", "rest", "index_fold", "rest"]:
        print(f"  gesture: {gesture}")
        controller.process_frame(gesture, 0, 0)
        time.sleep(0.4)

    print("\n── Test 2: Double taps ──────────────────────────────────")
    double_tap_tests = [
        ("thumb_index_tap",  "play_pause"),
        ("thumb_middle_tap", "home"),
        ("thumb_ring_tap",   "skip_forward"),
        ("thumb_pinky_tap",  "skip_back"),
        # Removed: thumb_middle_double_tap, thumb_ring_double_tap, thumb_pinky_double_tap
        # — these are not in GESTURE_PROFILES and were removed from the command maps
    ]
    for gesture, expected in double_tap_tests:
        print(f"  Testing {gesture} × 2 → expected: {expected}")
        controller.process_frame(gesture, 0, 0)
        time.sleep(0.15)   # within double tap window
        controller.process_frame("rest", 0, 0)
        controller.process_frame(gesture, 0, 0)
        time.sleep(0.35)   # past cooldown before next test

    print("\n── Test 3: IMU navigation ───────────────────────────────")
    print("  Simulating thumb rolling right (dx=5 per frame, 4 frames)...")
    for i in range(4):
        controller.process_frame("rest", 5.0, 0.0)  # 4 × 5 = 20 > threshold of 12
        print(f"    accum_x after frame {i+1}: {controller.accum_x:.1f}")

    print("\n── Test 4: Unmapped gesture (index_double_tap) ──────────")
    print("  gesture: index_double_tap (should do nothing in TV mode)")
    controller.process_frame("index_double_tap", 0, 0)

    print("\nAll tests complete.")