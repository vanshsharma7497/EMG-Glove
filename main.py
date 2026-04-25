"""
main.py
-------
Mouse Mode entry point — runs the real-time EMG glove dashboard.

Displays a 6-panel matplotlib dashboard showing:
  - Live EMG signal (4 channels)
  - Thumb IMU bar chart
  - Feature vector
  - Gesture history timeline
  - Cursor trail
  - Action log

Two execution paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE)
  True  → hardware loop  (reads from ESP32 via HardwareReader)

Run from GLOVE/ root:
    python main.py

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pyautogui  # used in both hardware and simulator loops for cursor position

from core.emg_filter import filter_all_channels
from core.emg_features import extract_features
from mode_manager import ModeManager, USE_HARDWARE

# ── Simulator imports (only used when USE_HARDWARE = False) ──────────────────
if not USE_HARDWARE:
    from core.emg_simulator import generate_emg_window
    from core.imu_simulator import generate_imu_sample, imu_to_cursor_velocity, THUMB_STATES

# ── Initialize mode manager ───────────────────────────────────────────────────
manager = ModeManager(mode="computer_mouse")

# ── Context agent — gesture remapping layer ───────────────────────────────────
# Runs as a daemon thread; reads ANTHROPIC_API_KEY from environment.
# Falls back silently if the key is absent or any API call fails.
from agent.context_agent import ContextAgent
context_agent = ContextAgent()
context_agent.start()

# Wire agent to the mouse controller so execute_gesture() can consult it
manager.glove_controller.context_agent = context_agent

# ── Gesture display config ────────────────────────────────────────────────────
GESTURE_COLORS = {
    "rest":               "#4fc3f7",
    "thumb_index_tap":    "#ef5350",
    "thumb_middle_tap":   "#ffa726",
    "thumb_ring_tap":     "#66bb6a",
    "thumb_pinky_tap":    "#ab47bc",
    "index_double_tap":   "#26c6da",
    "index_ring_tap":     "#f06292",
    "middle_pinky_tap":   "#80cbc4",
    "victory":            "#fff176",
    "thumbs_up":          "#a5d6a7",
    "flex":               "#ce93d8",
    "index_fold":         "#ffcc80",
    "middle_double_tap":  "#4db6ac",
    "ring_double_tap":    "#ff8a65",
    "fist":               "#e57373",
    "spiderman":          "#ba68c8",
}

GESTURE_LABELS_DISPLAY = {
    "rest":               "REST",
    "thumb_index_tap":    "LEFT CLICK",
    "thumb_middle_tap":   "RIGHT CLICK",
    "thumb_ring_tap":     "SCROLL UP",
    "thumb_pinky_tap":    "SCROLL DOWN",
    "index_double_tap":   "DOUBLE CLICK",
    # Unassigned in mouse mode — show gesture name for debugging
    "index_ring_tap":     "—",
    "middle_pinky_tap":   "—",
    "victory":            "—",
    "thumbs_up":          "—",
    "flex":               "—",
    "index_fold":         "—",
    "middle_double_tap":  "—",
    "ring_double_tap":    "—",
    "fist":               "—",
    "spiderman":          "—",
}

# ── Action log colours ────────────────────────────────────────────────────────
# Defined once here — used by both hardware and simulator drawing code.
LOG_COLORS = {
    "LEFT CLICK":   "#ef5350",
    "RIGHT CLICK":  "#ffa726",
    "SCROLL UP":    "#66bb6a",
    "SCROLL DOWN":  "#ab47bc",
    "DOUBLE CLICK": "#26c6da",
}

# ── Demo sequence (simulator mode only) ──────────────────────────────────────
DEMO_SEQUENCE = [
    ("rest",             "still",        20),
    ("rest",             "roll_right",   25),
    ("rest",             "roll_up",      20),
    ("rest",             "roll_left",    20),
    ("rest",             "still",        10),
    ("thumb_index_tap",  "still",         8),
    ("rest",             "still",         8),
    ("index_double_tap", "still",         8),
    ("rest",             "still",         8),
    ("rest",             "roll_down",    20),
    ("rest",             "roll_right",   15),
    ("thumb_middle_tap", "still",         8),
    ("rest",             "still",         8),
    ("rest",             "roll_up",      20),
    ("thumb_ring_tap",   "still",         8),
    ("rest",             "still",         8),
    ("thumb_pinky_tap",  "still",         8),
    ("rest",             "still",        15),
    ("rest",             "roll_up_right", 20),
    ("thumb_index_tap",  "still",         8),
    ("rest",             "still",        10),
]

# ── History buffers ───────────────────────────────────────────────────────────
HISTORY_LEN   = 40
emg_history   = []
cursor_path_x = []
cursor_path_y = []
action_log    = []

# ── Dashboard layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor("#0d0d0d")
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.55, wspace=0.35,
    left=0.06, right=0.97,
    top=0.88,  bottom=0.07
)

ax_emg      = fig.add_subplot(gs[0, :2])
ax_imu      = fig.add_subplot(gs[0, 2])
ax_features = fig.add_subplot(gs[1, 0])
ax_history  = fig.add_subplot(gs[1, 1])
ax_cursor   = fig.add_subplot(gs[1, 2])
ax_log      = fig.add_subplot(gs[2, :])

DARK = "#1a1a2e"
for ax in [ax_emg, ax_imu, ax_features, ax_history, ax_cursor, ax_log]:
    ax.set_facecolor(DARK)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

CH_COLORS = ["#ef5350", "#ffa726", "#66bb6a", "#ab47bc"]

# ── Dashboard drawing ─────────────────────────────────────────────────────────

def update_dashboard(filtered, imu_sample, pred_gesture, label, color,
                     thumb_state=None):
    
    Redraws all 6 dashboard panels for one frame.
    Called identically from both the hardware and simulator loops.

    filtered     : np.ndarray (4, 200) — filtered EMG window
    imu_sample   : dict — IMU gyro readings
    pred_gesture : str  — predicted gesture name
    label        : str  — display label for the gesture
    color        : str  — hex colour for the gesture
    thumb_state  : str or None — shown in title for simulator mode only
    
    ax_emg.cla(); ax_emg.set_facecolor(DARK)
    for ch in range(4):
        ax_emg.plot(filtered[ch], color=CH_COLORS[ch],
                    linewidth=0.7, alpha=0.85, label=f"Ch{ch+1}")
    ax_emg.set_ylim(-1.5, 1.5)
    mode_tag = "Live EMG" if USE_HARDWARE else "Filtered EMG"
    ax_emg.set_title(f"{mode_tag} — 4 Channels   |   {label}",
                     color=color, fontsize=10, fontweight="bold")
    ax_emg.legend(loc="upper right", fontsize=7,
                  facecolor=DARK, labelcolor="white", framealpha=0.5)
    ax_emg.set_xticks([]); ax_emg.set_yticks([])

    ax_imu.cla(); ax_imu.set_facecolor(DARK)
    ax_imu.bar(["Roll X\n(left/right)", "Roll Y\n(fwd/back)"],
               [imu_sample["gyro_x"], imu_sample["gyro_y"]],
               color=["#4fc3f7", "#f06292"], alpha=0.85)
    ax_imu.axhline(0, color="#555", linewidth=0.8)
    ax_imu.set_ylim(-2, 2)
    imu_tag = "IMU (Live)" if USE_HARDWARE else "Thumb IMU"
    ax_imu.set_title(imu_tag, color="white", fontsize=9)
    ax_imu.tick_params(colors="#aaaaaa", labelsize=7)

    features_disp = np.abs(extract_features(filtered))
    ax_features.cla(); ax_features.set_facecolor(DARK)
    ax_features.bar(range(20), features_disp, color=color, alpha=0.75)
    ax_features.set_title("Feature Vector (20)", color="white", fontsize=9)
    ax_features.set_xticks([]); ax_features.tick_params(colors="#aaa")

    ax_history.cla(); ax_history.set_facecolor(DARK)
    for i, g in enumerate(emg_history):
        alpha = 0.3 + 0.7 * (i / max(len(emg_history), 1))
        ax_history.bar(i, 1, color=GESTURE_COLORS.get(g, "#fff"),
                       alpha=alpha, width=0.9)
    patches = [mpatches.Patch(color=v, label=GESTURE_LABELS_DISPLAY[k])
               for k, v in list(GESTURE_COLORS.items())[:6]]
    ax_history.legend(handles=patches, fontsize=6,
                      facecolor=DARK, labelcolor="white",
                      loc="upper left", framealpha=0.5)
    ax_history.set_title(f"Gesture History (last {HISTORY_LEN})",
                         color="white", fontsize=9)
    ax_history.set_xticks([]); ax_history.set_yticks([])
    ax_history.set_xlim(-0.5, HISTORY_LEN - 0.5)

    ax_cursor.cla(); ax_cursor.set_facecolor(DARK)
    if len(cursor_path_x) > 1:
        ax_cursor.plot(cursor_path_x, cursor_path_y,
                       color="#4fc3f7", linewidth=1.0, alpha=0.6)
        ax_cursor.scatter(cursor_path_x[-1], cursor_path_y[-1],
                          color="white", s=30, zorder=5)
    ax_cursor.set_title("Cursor Trail", color="white", fontsize=9)
    ax_cursor.tick_params(colors="#aaa", labelsize=6)
    ax_cursor.set_xlabel("X", color="#aaa", fontsize=7)
    ax_cursor.set_ylabel("Y", color="#aaa", fontsize=7)
    ax_cursor.invert_yaxis()

    ax_log.cla(); ax_log.set_facecolor(DARK)
    ax_log.set_xlim(0, 1); ax_log.set_ylim(0, 1)
    ax_log.set_xticks([]); ax_log.set_yticks([])
    ax_log.set_title("Action Log", color="white", fontsize=9)
    for i, entry in enumerate(reversed(action_log)):
        alpha = 1.0 - i * 0.15
        # AI-remapped entries are tagged "(AI)" — show in orange
        if entry.endswith("(AI)"):
            c = "#ffa726"
        else:
            c = LOG_COLORS.get(entry, "#ffffff")
        ax_log.text(0.02 + i * 0.17, 0.55, entry,
                    color=c, fontsize=11, fontweight="bold",
                    alpha=max(alpha, 0.2), va="center",
                    transform=ax_log.transAxes)

    # ── Agent status strip (bottom 30% of ax_log) ─────────────────────────────
    # .source tells us which tier resolved context last:
    #   "rules"    → OS detection + context_rules.py (instant, offline)
    #   "cache"    → vision API result from earlier in the session
    #   "vision"   → live vision API call just completed
    #   "fallback" → unknown app, no API key, using static defaults
    src = context_agent.source
    if src == "rules":
        agent_label = "OS RULES"
        agent_color = "#66bb6a"    # green — fast and free
    elif src == "cache":
        agent_label = "VISION CACHE"
        agent_color = "#4fc3f7"    # blue — previously seen app
    elif src == "vision":
        agent_label = "VISION API"
        agent_color = "#ffa726"    # orange — live API call
    else:
        agent_label = "FALLBACK"
        agent_color = "#666666"    # grey — static mappings

    ctx_text  = context_agent.context_detected
    conf_text = (f"{context_agent.confidence * 100:.0f}% conf"
                 if src == "vision" else "")

    ax_log.text(0.02, 0.18, agent_label,
                color=agent_color, fontsize=9, fontweight="bold",
                va="center", transform=ax_log.transAxes)
    ax_log.text(0.18, 0.18, ctx_text,
                color="#aaaaaa", fontsize=8,
                va="center", transform=ax_log.transAxes)
    if conf_text:
        ax_log.text(0.80, 0.18, conf_text,
                    color="#aaaaaa", fontsize=8,
                    va="center", ha="right", transform=ax_log.transAxes)

    title_tag = "LIVE HARDWARE" if USE_HARDWARE else f"SIMULATOR  ◈  {thumb_state or ''}"
    fig.suptitle(f"EMG GLOVE  ◈  {title_tag}  ◈  {label}",
                 fontsize=16, fontweight="bold", color=color, y=0.97)

    plt.pause(0.04)


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\n  Mode: {'REAL HARDWARE' if USE_HARDWARE else 'SIMULATOR'}")
print(f"  {'Frame':<6} {'EMG Gesture':<20} {'dx':>6} {'dy':>6}")
print("  " + "─" * 40)

frame = 0

# ── Hardware loop ─────────────────────────────────────────────────────────────
if USE_HARDWARE:
    print("  Hardware mode active — move sensor to control cursor")
    print("  FAILSAFE: move mouse to top-left corner to stop\n")

    import time

    while True:
        try:
            pred_gesture, dx, dy = manager.process_hardware_frame()

            emg_window = manager.hw_reader.get_emg_window()
            filtered   = filter_all_channels(emg_window)
            imu_sample = manager.hw_reader.get_imu_sample()

            color = GESTURE_COLORS.get(pred_gesture, "#ffffff")
            label = GESTURE_LABELS_DISPLAY.get(pred_gesture, pred_gesture)

            emg_history.append(pred_gesture)
            cx, cy = pyautogui.position()
            cursor_path_x.append(cx)
            cursor_path_y.append(cy)
            if len(emg_history)   > HISTORY_LEN: emg_history.pop(0)
            if len(cursor_path_x) > HISTORY_LEN: cursor_path_x.pop(0)
            if len(cursor_path_y) > HISTORY_LEN: cursor_path_y.pop(0)

            if pred_gesture != "rest":
                src = manager.glove_controller._last_action_source
                log_label = label + " (AI)" if src == "agent" else label
                if not action_log or action_log[-1] != log_label:
                    action_log.append(log_label)
                    if len(action_log) > 6: action_log.pop(0)

            if frame % 10 == 0:
                print(f"  {frame:<6} {pred_gesture:<20} {dx:>6.2f} {dy:>6.2f}")

            update_dashboard(filtered, imu_sample, pred_gesture, label, color)

            frame += 1

        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            manager.stop()
            context_agent.stop()
            break

# ── Simulator loop ────────────────────────────────────────────────────────────
else:
    # ── Centre cursor before the demo sequence starts ─────────────────────────
    # The demo sequence moves the cursor in all four directions. If it starts
    # near a screen edge the pyautogui failsafe triggers immediately.
    # Centering gives the demo ~500px of travel in every direction on a 1080p
    # screen, so it completes without hitting any corner.
    _sw, _sh = pyautogui.size()
    pyautogui.moveTo(_sw // 2, _sh // 2, duration=0.3)
    print(f"  Cursor centred at ({_sw // 2}, {_sh // 2}) for simulation\n")

    for emg_gesture, thumb_state, num_frames in DEMO_SEQUENCE:
        for _ in range(num_frames):

            # 1. EMG pipeline
            raw        = generate_emg_window(emg_gesture)
            filtered   = filter_all_channels(raw)
            features   = extract_features(filtered)

            # 2. IMU pipeline
            imu_sample = generate_imu_sample(thumb_state)
            dx, dy     = imu_to_cursor_velocity(imu_sample)

            # 3. Process through mode manager
            pred_gesture = manager.process_frame(features, dx, dy)
            color        = GESTURE_COLORS.get(pred_gesture, "#ffffff")
            label        = GESTURE_LABELS_DISPLAY.get(pred_gesture, pred_gesture)

            # 4. Update buffers
            emg_history.append(pred_gesture)
            cx, cy = pyautogui.position()
            cursor_path_x.append(cx)
            cursor_path_y.append(cy)
            if len(emg_history)   > HISTORY_LEN: emg_history.pop(0)
            if len(cursor_path_x) > HISTORY_LEN: cursor_path_x.pop(0)
            if len(cursor_path_y) > HISTORY_LEN: cursor_path_y.pop(0)

            if pred_gesture != "rest":
                src = manager.glove_controller._last_action_source
                log_label = label + " (AI)" if src == "agent" else label
                if not action_log or action_log[-1] != log_label:
                    action_log.append(log_label)
                    if len(action_log) > 6: action_log.pop(0)

            print(f"  {frame:<6} {pred_gesture:<20} {thumb_state:<16} "
                  f"{dx:>6.2f} {dy:>6.2f}")

            # 5. Draw dashboard
            update_dashboard(filtered, imu_sample, pred_gesture, label, color,
                             thumb_state=thumb_state)

            frame += 1

    print("\n  Simulation complete!")
    plt.show()"""

"""
main.py
-------
EMG Smart Glove entry point — runs the real-time gesture pipeline.

Terminal output only — no matplotlib dashboard.
The desktop GUI (glove_app.py) will be the visual interface.

Two execution paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until Ctrl+C)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — runs the real-time gesture pipeline.

Terminal output only — no matplotlib dashboard.
The desktop GUI (glove_app.py) will be the visual interface.

Two execution paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until Ctrl+C)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — runs the real-time gesture pipeline.

Terminal output only — no matplotlib dashboard.
glove_app.py is the visual interface; it reads glove_status.json written here.

Two execution paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until Ctrl+C)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — pipeline + GUI, single command.

Launches glove_app.py (the status GUI) on the main thread and runs the
gesture pipeline in a background daemon thread. Close the GUI window to
stop everything.

Two pipeline paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until GUI is closed)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — pipeline + GUI, single command.

Launches glove_app.py (the status GUI) on the main thread and runs the
gesture pipeline in a background daemon thread. Close the GUI window to
stop everything.

Two pipeline paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until GUI is closed)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — pipeline + GUI, single command.

Launches glove_app.py (the status GUI) on the main thread and runs the
gesture pipeline in a background daemon thread. Close the GUI window to
stop everything.

Two pipeline paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until GUI is closed)

Run from GLOVE/ root:
    python main.py
"""
"""
main.py
-------
EMG Smart Glove entry point — pipeline + GUI, single command.

Launches glove_app.py (the status GUI) on the main thread and runs the
gesture pipeline in a background daemon thread. Close the GUI window to
stop everything.

Two pipeline paths selected by USE_HARDWARE in mode_manager.py:
  False → simulator loop (scripted DEMO_SEQUENCE, runs once then exits)
  True  → hardware loop  (reads from ESP32, runs until GUI is closed)

Run from GLOVE/ root:
    python main.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import threading
import pyautogui

from core.emg_filter   import filter_all_channels
from core.emg_features import extract_features
from mode_manager      import ModeManager, USE_HARDWARE, DEVICE_MODES

# ── Simulator-only imports ────────────────────────────────────────────────────
if not USE_HARDWARE:
    from core.emg_simulator import generate_emg_window
    from core.imu_simulator import generate_imu_sample, imu_to_cursor_velocity
    import keyboard as _keyboard

# ── Failsafe config ───────────────────────────────────────────────────────────
# In simulator mode the cursor is driven by scripted IMU values, not the user.
# The failsafe corner-trigger fires spuriously when the script pushes the cursor
# near an edge. Disable it for simulator runs only.
# Hardware mode keeps it enabled — it is the user's real emergency stop.
if not USE_HARDWARE:
    pyautogui.FAILSAFE = False

# ── Mode manager + controllers ────────────────────────────────────────────────
manager = ModeManager(mode="computer_mouse")

# ── Context agent — OS detection + gesture remapping ─────────────────────────
from agent.context_agent import ContextAgent
context_agent = ContextAgent()
context_agent.start()

# ── Load saved user rules on startup ─────────────────────────────────────────
# Restores any gesture overrides the user saved via the GUI in a previous
# session. Without this, user_rules resets to {} on every restart and
# overrides only take effect after the GUI sends a reload_rules command.
try:
    with open("glove_config.json", "r") as _f:
        _cfg = json.load(_f)
    context_agent.user_rules = _cfg.get("user_rules", {})
    _n = sum(len(v) for v in context_agent.user_rules.values())
    if _n:
        print(f"  [Agent] Loaded {_n} user rule override{'s' if _n != 1 else ''} from glove_config.json")
except FileNotFoundError:
    pass   # no config yet — first run
except Exception as _e:
    print(f"  [Agent] Could not load glove_config.json: {_e}")

# Wire the agent to the mouse controller so execute_gesture() can consult it
manager.glove_controller.context_agent = context_agent

# ── Status file — bridge to glove_app.py ─────────────────────────────────────
# glove_app.py polls this file every second to update its display.
# Written every frame; uses atomic rename to avoid partial reads.
STATUS_FILE   = "glove_status.json"
COMMANDS_FILE = "glove_commands.json"   # GUI writes, pipeline reads + deletes

def write_status(gesture, action, source):
    """
    Write current pipeline state to glove_status.json.
    Called every frame. Silently swallows all errors — must never
    crash or slow down the gesture pipeline.

    gesture : str — predicted gesture name e.g. "thumb_index_tap"
    action  : str — human-readable label e.g. "LEFT CLICK"
    source  : str — "agent" or "static"
    """
    try:
        payload = {
            "app_name":         context_agent.app_name,
            "source":           context_agent.source,
            "context_detected": context_agent.context_detected,
            "confidence":       context_agent.confidence,
            "device_mode":      manager.device_mode,
            "last_gesture":     gesture,
            "last_action":      action,
            "action_source":    source,
            "hardware":         "HARDWARE" if USE_HARDWARE else "SIMULATOR",
            "timestamp":        time.time(),
        }
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, STATUS_FILE)   # atomic on all platforms
    except Exception:
        pass

def check_commands():
    """
    Reads glove_commands.json and executes any pending command.
    Deletes the file after execution so each command fires exactly once.
    Called every 10 frames inside run_pipeline() — never crashes the pipeline.
    """
    try:
        with open(COMMANDS_FILE, "r") as f:
            cmd = json.load(f)
        command = cmd.get("command")
        value   = cmd.get("value")
        if command == "set_mode" and value in DEVICE_MODES:
            manager.set_device_mode(value)
            print(f"  [Commands] Mode set → '{value}'")
        elif command == "reload_rules":
            try:
                with open("glove_config.json", "r") as f:
                    cfg = json.load(f)
                user_rules = cfg.get("user_rules", {})
            except FileNotFoundError:
                user_rules = {}
            context_agent.user_rules = user_rules
            n = sum(len(v) for v in user_rules.values())
            print(f"  [Commands] Rules reloaded — {n} user override{'s' if n != 1 else ''}")
        elif command == "apply_settings":
            import modes.mouse_mode    as _mm
            import agent.os_context    as _oc
            import agent.context_agent as _ca
            global KEYBOARD_MODE
            s = cmd.get("settings", {})
            if "click_cooldown"   in s: _mm.CLICK_COOLDOWN  = float(s["click_cooldown"])
            if "scroll_amount"    in s: _mm.SCROLL_AMOUNT    = int(s["scroll_amount"])
            if "os_poll_interval" in s: _oc.POLL_INTERVAL    = float(s["os_poll_interval"])
            if "vision_poll"      in s: _ca.POLL_INTERVAL    = float(s["vision_poll"])
            if "cursor_speed"     in s: manager.glove_controller.speed = float(s["cursor_speed"])
            if "keyboard_mode"    in s:
                KEYBOARD_MODE = bool(s["keyboard_mode"])
                print(f"  [Commands] keyboard_mode={KEYBOARD_MODE} — takes effect on next run")
            applied = [k for k in s if k != "keyboard_mode"]
            print(f"  [Commands] Settings applied — {', '.join(applied) if applied else 'none'}")
        os.remove(COMMANDS_FILE)   # consumed — delete so it fires only once
    except FileNotFoundError:
        pass   # no command pending — normal
    except Exception as e:
        print(f"  [Commands] Error: {e}")

# ── Keyboard test mode ───────────────────────────────────────────────────────
# When True and USE_HARDWARE = False, run an interactive keyboard-driven loop
# instead of the scripted DEMO_SEQUENCE. Set False to restore demo playback.
KEYBOARD_MODE = True   # ← set False to use original DEMO_SEQUENCE

# ── Demo sequence (simulator mode only) ──────────────────────────────────────
DEMO_SEQUENCE = [
    # ── Rest + cursor movement ─────────────────────────────────────────────
    ("rest",             "still",         30),   # 2.4s — startup pause
    ("rest",             "roll_right",    40),   # 3.2s — drift right
    ("rest",             "roll_up",       30),   # 2.4s — drift up
    ("rest",             "roll_left",     40),   # 3.2s — drift left
    ("rest",             "roll_down",     30),   # 2.4s — drift down (back near centre)
    ("rest",             "still",         25),   # 2.0s — pause: good time to switch apps
    # ── Click gestures ─────────────────────────────────────────────────────
    ("thumb_index_tap",  "still",         12),   # left click
    ("rest",             "still",         20),
    ("index_double_tap", "still",         12),   # double click
    ("rest",             "still",         20),
    ("thumb_middle_tap", "still",         12),   # right click
    ("rest",             "still",         20),
    # ── Scroll gestures ────────────────────────────────────────────────────
    ("rest",             "roll_right",    25),
    ("rest",             "still",         15),
    ("thumb_ring_tap",   "still",         12),   # scroll up
    ("rest",             "still",         20),
    ("thumb_pinky_tap",  "still",         12),   # scroll down
    ("rest",             "still",         25),   # 2.0s — pause: switch apps here
    # ── Diagonal movement + click ──────────────────────────────────────────
    ("rest",             "roll_up_right", 30),
    ("thumb_index_tap",  "still",         12),
    ("rest",             "still",         30),   # 2.4s — final pause
]

# ── Gesture display labels (terminal output) ──────────────────────────────────
GESTURE_LABELS = {
    "rest":               "REST",
    "thumb_index_tap":    "LEFT CLICK",
    "thumb_middle_tap":   "RIGHT CLICK",
    "thumb_ring_tap":     "SCROLL UP",
    "thumb_pinky_tap":    "SCROLL DOWN",
    "index_double_tap":   "DOUBLE CLICK",
    "index_ring_tap":     "—",
    "middle_pinky_tap":   "—",
    "victory":            "—",
    "thumbs_up":          "—",
    "flex":               "—",
    "index_fold":         "—",
    "middle_double_tap":  "—",
    "ring_double_tap":    "—",
    "fist":               "—",
    "spiderman":          "—",
}

# ── Startup settings — apply saved values before the pipeline loop ────────────

def _apply_startup_settings():
    """
    Reads glove_config.json["settings"] and applies the saved values using the
    same logic as the apply_settings command handler.  Called once at the top of
    run_pipeline() so settings survive restarts without needing a GUI command.
    """
    import modes.mouse_mode    as _mm
    import agent.os_context    as _oc
    import agent.context_agent as _ca
    global KEYBOARD_MODE

    try:
        with open("glove_config.json", "r") as _f:
            _cfg = json.load(_f)
        s = _cfg.get("settings", {})
    except FileNotFoundError:
        return   # no config yet — first run, use module defaults
    except Exception as _e:
        print(f"  [Pipeline] Could not load saved settings: {_e}")
        return

    if not s:
        return

    if "click_cooldown"   in s: _mm.CLICK_COOLDOWN  = float(s["click_cooldown"])
    if "scroll_amount"    in s: _mm.SCROLL_AMOUNT    = int(s["scroll_amount"])
    if "os_poll_interval" in s: _oc.POLL_INTERVAL    = float(s["os_poll_interval"])
    if "vision_poll"      in s: _ca.POLL_INTERVAL    = float(s["vision_poll"])
    if "cursor_speed"     in s: manager.glove_controller.speed = float(s["cursor_speed"])
    if "keyboard_mode"    in s: KEYBOARD_MODE        = bool(s["keyboard_mode"])

    applied = list(s.keys())
    print(f"  [Pipeline] Startup settings applied — {', '.join(applied)}")


# ── Pipeline function — runs in a background thread ───────────────────────────
# Everything that was at module level is now inside this function so it can
# be launched in a daemon thread while the GUI owns the main thread.

def run_pipeline():
    """
    Runs the full gesture pipeline — either hardware or simulator loop.
    Called in a daemon thread from the bottom of this file.
    Daemon = killed automatically when the main thread (GUI) exits.
    """
    _apply_startup_settings()   # restore saved settings before any frame runs

    frame = 0

    print(f"\n  Mode: {'REAL HARDWARE' if USE_HARDWARE else 'SIMULATOR'}")
    print(f"  {'Frame':<6} {'Gesture':<22} {'Thumb / Source':<18} {'dx':>7} {'dy':>7}")
    print("  " + "─" * 64)

    # ── Hardware loop ─────────────────────────────────────────────────────────
    if USE_HARDWARE:
        print("  Hardware active — gesture to control cursor")
        print("  Close the GUI window to stop\n")

        while True:
            try:
                pred_gesture, dx, dy = manager.process_hardware_frame()
                source  = manager.glove_controller._last_action_source
                action  = GESTURE_LABELS.get(pred_gesture, pred_gesture)
                src_tag = " [AI]" if source == "agent" else ""

                write_status(pred_gesture, action, source)

                if frame % 10 == 0:
                    check_commands()   # execute any pending GUI command
                    print(f"  {frame:<6} {pred_gesture:<22} {source:<18} "
                          f"{dx:>7.2f} {dy:>7.2f}{src_tag}")
                frame += 1

            except Exception as e:
                if "KeyboardInterrupt" in str(type(e)):
                    break
                print(f"  [Pipeline] Error: {e}")
                time.sleep(0.1)

    # ── Simulator loop ────────────────────────────────────────────────────────
    else:
        if KEYBOARD_MODE:
            # ── Keyboard-driven interactive loop ──────────────────────────────

            # Key → IMU thumb state (held — polled every frame via is_pressed)
            _KEY_IMU = {
                "w": "roll_up",
                "a": "roll_left",
                "s": "roll_down",
                "d": "roll_right",
            }

            # Key → gesture name (single-press — consumed once per frame)
            _KEY_GESTURES = {
                "e": "thumb_index_tap",
                "r": "thumb_middle_tap",
                "f": "thumb_ring_tap",
                "v": "thumb_pinky_tap",
                "g": "index_double_tap",
                "t": "victory",
                "y": "thumbs_up",
                "h": "fist",
                "b": "flex",
                "n": "index_fold",
                "u": "spiderman",
                "j": "middle_double_tap",
                "m": "ring_double_tap",
                "i": "index_ring_tap",
                "k": "middle_pinky_tap",
            }

            # Pending gesture — set by on_press_key callback, cleared each frame
            _pending_gesture = {"name": None}
            _stop_flag       = {"flag": False}

            # Register single-press gesture callbacks (fires once per keydown)
            for _key, _gesture in _KEY_GESTURES.items():
                _keyboard.on_press_key(
                    _key,
                    lambda _, g=_gesture: _pending_gesture.update({"name": g}),
                )

            # Mode-switch callbacks — 1/2/3 keys
            _keyboard.on_press_key("1", lambda _: manager.set_device_mode("standard"))
            _keyboard.on_press_key("2", lambda _: manager.set_device_mode("communication"))
            _keyboard.on_press_key("3", lambda _: manager.set_device_mode("tv_remote"))

            # Exit callbacks
            _keyboard.on_press_key("esc", lambda _: _stop_flag.update({"flag": True}))
            _keyboard.on_press_key("q",   lambda _: _stop_flag.update({"flag": True}))

            # ── Print key reference table ──────────────────────────────────────
            print("\n  KEYBOARD TEST MODE — Key Reference")
            print("  " + "─" * 54)
            print("  Movement (hold):  W=up  A=left  S=down  D=right")
            print("  " + "─" * 54)
            print("  E  thumb_index_tap     (left click)")
            print("  R  thumb_middle_tap    (right click)")
            print("  F  thumb_ring_tap      (scroll up)")
            print("  V  thumb_pinky_tap     (scroll down)")
            print("  G  index_double_tap    (double click)")
            print("  T  victory             H  fist")
            print("  Y  thumbs_up           B  flex")
            print("  N  index_fold          U  spiderman")
            print("  J  middle_double_tap   M  ring_double_tap")
            print("  I  index_ring_tap      K  middle_pinky_tap")
            print("  " + "─" * 54)
            print("  1=standard  2=communication  3=tv_remote")
            print("  ESC / Q  →  exit")
            print("  " + "─" * 54 + "\n")

            # Centre cursor before starting
            _sw, _sh = pyautogui.size()
            _cx, _cy = _sw // 2, _sh // 2
            pyautogui.moveTo(_cx, _cy, duration=0.3)
            print(f"  Cursor centred at ({_cx}, {_cy})\n")

            while not _stop_flag["flag"]:
                try:
                    # 1. IMU — poll held WASD keys for thumb state
                    if   _keyboard.is_pressed("w"): thumb_state = "roll_up"
                    elif _keyboard.is_pressed("s"): thumb_state = "roll_down"
                    elif _keyboard.is_pressed("a"): thumb_state = "roll_left"
                    elif _keyboard.is_pressed("d"): thumb_state = "roll_right"
                    else:                           thumb_state = "still"

                    imu_sample = generate_imu_sample(thumb_state)
                    dx, dy     = imu_to_cursor_velocity(imu_sample)

                    # 2. EMG — consume pending gesture (fires once per key press)
                    emg_gesture              = _pending_gesture["name"] or "rest"
                    _pending_gesture["name"] = None

                    # 3. EMG pipeline
                    raw      = generate_emg_window(emg_gesture)
                    filtered = filter_all_channels(raw)
                    features = extract_features(filtered)

                    # 4. Classify + route to controller
                    pred_gesture = manager.process_frame(features, dx, dy)
                    source       = manager.glove_controller._last_action_source
                    action       = GESTURE_LABELS.get(pred_gesture, pred_gesture)
                    src_tag      = " [AI]" if source == "agent" else ""

                    # 5. Write status for glove_app.py
                    write_status(pred_gesture, action, source)

                    # 6. Check for GUI commands every 10 frames
                    if frame % 10 == 0:
                        check_commands()

                    # 7. Terminal output — gesture events + periodic heartbeat
                    if pred_gesture != "rest" or frame % 20 == 0:
                        print(f"  {frame:<6} {pred_gesture:<22} {thumb_state:<14} "
                              f"{dx:>7.2f} {dy:>7.2f}{src_tag}")

                    frame += 1
                    time.sleep(0.05)   # 20fps — smooth cursor movement

                except Exception as e:
                    if "KeyboardInterrupt" not in str(type(e)):
                        print(f"  [Keyboard] Error: {e}")
                    break

            # Unhook all keyboard listeners on exit
            _keyboard.unhook_all()
            print("  Keyboard test mode exited.")

        else:
            # ── Scripted DEMO_SEQUENCE loop ────────────────────────────────────
            # Centre cursor so the demo sequence has room to drift in all directions
            _sw, _sh = pyautogui.size()
            _cx, _cy = _sw // 2, _sh // 2
            pyautogui.moveTo(_cx, _cy, duration=0.3)
            print(f"  Cursor centred at ({_cx}, {_cy})\n")

            try:
                for emg_gesture, thumb_state, num_frames in DEMO_SEQUENCE:
                    for _ in range(num_frames):

                        # 1. EMG pipeline
                        raw      = generate_emg_window(emg_gesture)
                        filtered = filter_all_channels(raw)
                        features = extract_features(filtered)

                        # 2. IMU → cursor velocity
                        imu_sample = generate_imu_sample(thumb_state)
                        dx, dy     = imu_to_cursor_velocity(imu_sample)

                        # 3. Classify + route to mouse controller
                        pred_gesture = manager.process_frame(features, dx, dy)
                        source       = manager.glove_controller._last_action_source
                        action       = GESTURE_LABELS.get(pred_gesture, pred_gesture)
                        src_tag      = " [AI]" if source == "agent" else ""

                        # 4. Write status for glove_app.py
                        write_status(pred_gesture, action, source)

                        # 4a. Check for mode-switch commands from the GUI (every 10 frames)
                        if frame % 10 == 0:
                            check_commands()

                        # 5. Terminal output
                        print(f"  {frame:<6} {pred_gesture:<22} {thumb_state:<18} "
                              f"{dx:>7.2f} {dy:>7.2f}{src_tag}")

                        frame += 1
                        time.sleep(0.08)   # ~12fps — slow enough to watch and switch apps

            except Exception as e:
                if "KeyboardInterrupt" not in str(type(e)):
                    print(f"  [Pipeline] Error: {e}")

            finally:
                context_agent.stop()
                print("  Simulation complete.")


# ── Launch ────────────────────────────────────────────────────────────────────
# The pipeline runs in a daemon thread; the GUI owns the main thread.
# Closing the GUI window kills the daemon thread automatically.

pipeline_thread = threading.Thread(
    target=run_pipeline,
    name="GlovePipeline",
    daemon=True,    # dies when the GUI (main thread) exits
)
pipeline_thread.start()

# GUI must run on the main thread — this call blocks until the window is closed
from glove_app import GloveApp
app = GloveApp()
app.mainloop()

# ── Cleanup after GUI closes ──────────────────────────────────────────────────
context_agent.stop()
manager.stop()
print("  Glove app closed.")