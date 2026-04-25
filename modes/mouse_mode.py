"""
modes/mouse_mode.py
-------------------
Mouse Mode controller — translates EMG gestures + IMU movement into OS mouse events.

Maps 6 of the 16 gestures to mouse actions (left click, right click, scroll up/down,
double click). The remaining 10 gestures are unassigned (None) in this mode —
they are available for the AI agent layer to remap contextually.
Cursor movement comes from the thumb IMU via sub-pixel accumulation.

Architecture:
  Called by: mode_manager.py (routes EMG frames here when glove_mode="computer_mouse")
  Calls:     pyautogui for all OS mouse events
  Exports:   MouseController, GESTURE_ACTIONS, SCROLL_AMOUNT, CLICK_COOLDOWN
"""
import pyautogui
import time
import numpy as np

# ── Safety Settings ───────────────────────────────────────────────────────────
# pyautogui has a built-in safety feature — if you move the mouse to the
# top-left corner of the screen it raises an exception and stops the program.
# This is your emergency stop if something goes wrong.
pyautogui.FAILSAFE = True

# Removes the small delay pyautogui adds between actions by default.
# We handle timing ourselves for better real-time feel.
pyautogui.PAUSE = 0

# ── Gesture → Action Mapping ──────────────────────────────────────────────────
# All 16 classifier gestures are listed explicitly.
# None = unassigned in this mode — available for the AI agent to remap.
GESTURE_ACTIONS = {
    "rest":               None,             # baseline — do nothing
    # ── Assigned ──────────────────────────────────────────────────────────────
    "thumb_index_tap":    "left_click",
    "thumb_middle_tap":   "right_click",
    "thumb_ring_tap":     "scroll_up",
    "thumb_pinky_tap":    "scroll_down",
    "index_double_tap":   "double_click",
    # ── Unassigned — available for AI agent remapping ─────────────────────────
    "index_ring_tap":     None,
    "middle_pinky_tap":   None,
    "middle_double_tap":  None,
    "ring_double_tap":    None,
    "victory":            None,
    "thumbs_up":          None,
    "flex":               None,
    "index_fold":         None,
    "fist":               None,
    "spiderman":          None,
}

# ── Action → Keyboard Shortcut Mapping ───────────────────────────────────────
# Maps agent-supplied action names to pyautogui.hotkey() key tuples.
# execute_gesture() unpacks the tuple as hotkey(*keys).
# Actions NOT listed here keep print-only behaviour (no standard shortcut).
ACTION_HOTKEYS = {

    # ── Video / Media (YouTube, Netflix, Spotify) ─────────────────────────────
    "fullscreen_toggle":   ("f",),
    "pause_play":          ("space",),     # caution: also fires in text fields
    "seek_forward_10s":    ("l",),
    "seek_back_10s":       ("j",),
    "volume_up":           ("up",),
    "volume_down":         ("down",),
    "mute_toggle":         ("m",),         # video standard (M); Discord uses ctrl+shift+m
    "theatre_mode":        ("t",),
    "captions_toggle":     ("c",),
    "next_video":          ("shift", "n"),
    "next_track":          ("ctrl", "right"),
    "previous_track":      ("ctrl", "left"),
    "shuffle_toggle":      ("ctrl", "shift", "s"),
    "repeat_toggle":       ("ctrl", "r"),

    # ── Navigation ────────────────────────────────────────────────────────────
    "go_back":             ("alt", "left"),
    "go_forward":          ("alt", "right"),
    "go_up":               ("alt", "up"),
    "new_folder":          ("ctrl", "shift", "n"),
    "rename":              ("f2",),
    "delete_file":         ("delete",),
    "properties":          ("alt", "enter"),
    "address_bar":         ("alt", "d"),

    # ── VS Code ───────────────────────────────────────────────────────────────
    "go_to_definition":    ("f12",),
    "find_all_references": ("shift", "f12"),
    "open_terminal":       ("ctrl", "grave"),   # ctrl+`
    "command_palette":     ("ctrl", "shift", "p"),
    "find_in_file":        ("ctrl", "f"),
    "format_document":     ("shift", "alt", "f"),
    "undo":                ("ctrl", "z"),
    "redo":                ("ctrl", "y"),

    # ── Google Docs / Sheets ──────────────────────────────────────────────────
    "comment":             ("ctrl", "alt", "m"),
    "find_and_replace":    ("ctrl", "h"),
    "go_to_top":           ("ctrl", "home"),
    "go_to_bottom":        ("ctrl", "end"),
    "bold_toggle":         ("ctrl", "b"),
    "italic_toggle":       ("ctrl", "i"),
    "word_count":          ("ctrl", "shift", "c"),
    "go_to_cell":          ("ctrl", "g"),

    # ── Browser / GitHub ──────────────────────────────────────────────────────
    "focus_search":        ("slash",),          # / key (GitHub shortcut)
    "open_file_finder":    ("ctrl", "p"),       # general; GitHub uses T key natively
    "keyboard_shortcuts":  ("question",),       # ? key

    # ── Discord ───────────────────────────────────────────────────────────────
    "deafen_toggle":       ("ctrl", "shift", "d"),
    "next_unread_channel": ("alt", "shift", "down"),
    "prev_channel":        ("alt", "up"),
    "jump_to_inbox":       ("ctrl", "i"),

    # ── General ───────────────────────────────────────────────────────────────
    "select_all":          ("ctrl", "a"),
    "copy":                ("ctrl", "c"),
    "paste":               ("ctrl", "v"),
    "save":                ("ctrl", "s"),

    # ── Not listed: like_video, like_track, next_episode, insert_row, ─────────
    # ── star_repo, push_to_talk, new_conversation, copy_last_response, ────────
    # ── focus_input — no standard shortcut, kept as print-only. ───────────────
}

# ── Scroll Settings ───────────────────────────────────────────────────────────
SCROLL_AMOUNT = 3       # how many units to scroll per detected gesture
                        # increase if scrolling feels too slow

# ── Click Cooldown ────────────────────────────────────────────────────────────
# Prevents accidental double-clicks if the same gesture lingers for
# multiple windows. A tap gesture must "rest" before firing again.
CLICK_COOLDOWN = 0.4    # seconds between allowed clicks

class MouseController:
    def __init__(self, context_agent=None):
        self.context_agent = context_agent   # ContextAgent instance, or None for fallback
        self._last_action_source = "static"  # "agent" or "static" — set on every action

        self.last_action_time = 0
        self.last_gesture = "rest"
        self.cursor_x, self.cursor_y = pyautogui.position()

        # Accumulator for sub-pixel cursor movements
        # Without this, small dx/dy values get lost due to integer rounding
        self.accum_x = 0.0
        self.accum_y = 0.0

        # Cursor speed multiplier — applied to every dx/dy in move_cursor().
        # Set by the Settings panel via apply_settings command in main.py.
        # 1.0 = default speed, 3.0 = three times faster.
        self.speed = 1.0

        print("  Mouse controller initialized")
        print(f"  Starting cursor position: ({self.cursor_x}, {self.cursor_y})")
        print("  FAILSAFE: move mouse to top-left corner to emergency stop\n")


    def move_cursor(self, dx, dy):
        """
        Moves the cursor by (dx, dy) pixels from current position.
        Uses sub-pixel accumulation so small movements aren't lost.
        """
        if dx == 0.0 and dy == 0.0:
            return

        # Apply speed multiplier — controlled by the Settings panel
        dx *= self.speed
        dy *= self.speed

        # Accumulate fractional pixels
        self.accum_x += dx
        self.accum_y += dy

        # Only move when we've accumulated at least 1 pixel
        move_x = int(self.accum_x)
        move_y = int(self.accum_y)

        if move_x != 0 or move_y != 0:
            self.accum_x -= move_x
            self.accum_y -= move_y

            # Get current position and compute new position
            current_x, current_y = pyautogui.position()
            screen_w, screen_h = pyautogui.size()

            # Clamp to screen boundaries
            new_x = int(np.clip(current_x + move_x, 0, screen_w - 1))
            new_y = int(np.clip(current_y + move_y, 0, screen_h - 1))

            pyautogui.moveTo(new_x, new_y, duration=0)


    def execute_gesture(self, gesture_name):
        """
        Executes the mouse action mapped to the detected gesture.
        Asks the context agent first; falls back to the static GESTURE_ACTIONS
        dict if the agent is inactive, unavailable, or has no remapping.
        Respects cooldown to prevent repeated firing.
        """
        # ── Action lookup — agent first, static fallback ───────────────────────
        # Ask the agent first — returns None if agent is inactive,
        # unavailable, or has no remapping for this gesture
        action = None
        if self.context_agent is not None:
            action = self.context_agent.get_action(gesture_name)
            if action is not None:
                self._last_action_source = "agent"

        # Fall back to static mapping if agent returned nothing
        if action is None:
            action = GESTURE_ACTIONS.get(gesture_name)
            self._last_action_source = "static"

        if action is None:
            self.last_gesture = gesture_name
            return

        # ── Cooldown check ────────────────────────────────────────────────────
        # Only fire if enough time has passed AND the gesture changed
        # (finger was lifted and re-tapped)
        now = time.time()
        gesture_changed = gesture_name != self.last_gesture
        cooldown_passed = (now - self.last_action_time) > CLICK_COOLDOWN

        if gesture_changed and cooldown_passed:
            # src suffix appended to terminal output so AI remaps are visible
            src = " [AI]" if self._last_action_source == "agent" else ""

            if action == "left_click":
                pyautogui.click(button="left")
                print(f"  → LEFT CLICK{src}")

            elif action == "right_click":
                pyautogui.click(button="right")
                print(f"  → RIGHT CLICK{src}")

            elif action == "scroll_up":
                pyautogui.scroll(SCROLL_AMOUNT)
                print(f"  → SCROLL UP{src}")

            elif action == "scroll_down":
                pyautogui.scroll(-SCROLL_AMOUNT)
                print(f"  → SCROLL DOWN{src}")

            elif action == "double_click":
                pyautogui.doubleClick(button="left")
                print(f"  → DOUBLE CLICK{src}")

            else:
                # Agent-supplied custom action — look up keyboard shortcut
                keys = ACTION_HOTKEYS.get(action)
                if keys is not None:
                    try:
                        pyautogui.hotkey(*keys)
                        print(f"  → {action.upper().replace('_', ' ')}{src}")
                    except Exception as e:
                        print(f"  → {action.upper().replace('_', ' ')} [hotkey error: {e}]{src}")
                else:
                    # No standard shortcut defined — log only
                    print(f"  → {action.upper().replace('_', ' ')}{src}")

            self.last_action_time = now

        self.last_gesture = gesture_name


    def process_frame(self, gesture_name, dx, dy):
        """
        Main entry point called every frame from the realtime loop.
        Handles both cursor movement and gesture actions together.
        
        gesture_name : predicted gesture from EMG classifier
        dx, dy       : cursor velocity from IMU thumb simulator
        """
        self.move_cursor(dx, dy)
        self.execute_gesture(gesture_name)


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from core.imu_simulator import generate_imu_sample, imu_to_cursor_velocity

    controller = MouseController()
    screen_w, screen_h = pyautogui.size()
    print(f"  Screen size: {screen_w} x {screen_h}")
    print("\n  Running cursor movement test (5 seconds)...")
    print("  Watch your cursor — it should drift right then up\n")

    # Test cursor movement
    for i in range(100):
        state = "roll_right" if i < 50 else "roll_up"
        sample = generate_imu_sample(state)
        dx, dy = imu_to_cursor_velocity(sample)
        controller.process_frame("rest", dx, dy)
        time.sleep(0.05)

    print("\n  Testing clicks...")
    time.sleep(0.5)
    controller.process_frame("thumb_index_tap", 0, 0)
    time.sleep(0.5)
    controller.process_frame("thumb_middle_tap", 0, 0)
    time.sleep(0.5)
    controller.process_frame("thumb_ring_tap", 0, 0)
    time.sleep(0.5)
    controller.process_frame("thumb_pinky_tap", 0, 0)

    print("\n  Test complete!")