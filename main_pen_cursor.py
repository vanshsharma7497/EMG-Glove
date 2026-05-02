"""
main_pen_cursor.py
------------------
Live dashboard for the pen cursor mode pipeline (Slider position 1).

Demonstrates all pen cursor actions:
  Rolling        → cursor movement
  Plant + still 0.5s → scroll mode armed
  Tilt fwd       → scroll down
  Tilt back      → scroll up
  Single tap     → left click
  Double tap     → double click
  Firm press     → right click
  Tap and hold   → drag start / drag end

Six panels:
  Top-left    : Cursor trail
  Top-right   : IMU readings (gyro_x, gyro_y, gyro_z) live bars
  Mid-left    : Pressure signal
  Mid-right   : Scroll state machine (IDLE / CONFIRMING / SCROLL_READY)
  Bot-left    : Action log
  Bot-right   : Scroll position history

Run from GLOVE/ root:
    python main_pen_cursor.py

── Hardware Switch ────────────────────────────────────────────────────────────
Reads USE_PEN_HARDWARE from mode_manager.py:
  USE_PEN_HARDWARE = False  →  runs scripted demo sequence (original behaviour)
  USE_PEN_HARDWARE = True   →  reads live data from pen ESP32 on COM11

FAILSAFE: move the mouse cursor to the top-left corner of the screen
to immediately stop the program if something goes wrong.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pyautogui

# ── pyautogui safety settings ─────────────────────────────────────────────────
# FAILSAFE: move mouse to top-left corner of screen to emergency-stop.
pyautogui.FAILSAFE = True
# Remove default inter-action delay — we handle timing ourselves.
pyautogui.PAUSE = 0

# ── Hardware switch — imported from mode_manager ──────────────────────────────
from mode_manager import USE_PEN_HARDWARE, PEN_HARDWARE_PORT

# ── Pipeline imports ──────────────────────────────────────────────────────────
from core.pen_simulator import (
    generate_pressure_idle,
    generate_pressure_light_tap,
    generate_pressure_firm_press,
    generate_pressure_tap_and_hold,
    generate_imu_sample,
    PEN_DOWN_THRESHOLD,
    SAMPLE_RATE,
)
from modes.writing_mode import (
    WritingController,
    SCROLL_CONFIRM_SAMPLES,
    SCROLL_STILL_THRESHOLD,
)

from core.pen_hardware_reader import PenHardwareReader

# ---------------------------------------------------------------------------
# Demo sequence
# ---------------------------------------------------------------------------

def build_demo():
    stream = []

    def add(pressure_arr, motion, label):
        for p in pressure_arr:
            gx, gy, gz = generate_imu_sample(motion)
            stream.append((p, gx, gy, gz, label))

    def idle_pen_up(duration=0.2):
        add(generate_pressure_idle(duration), "still", "pen up / idle")

    def idle_pen_down(duration, motion="still"):
        p = generate_pressure_idle(duration)
        # Held down: boost pressure above threshold
        import numpy as np
        p = np.clip(p + 0.35, PEN_DOWN_THRESHOLD, 1.0)
        add(p, motion, f"pen down / {motion}")

    # Start with pen up
    idle_pen_up(0.3)

    # ── Cursor movement ────────────────────────────────────────────────────
    add(generate_pressure_idle(0.5), "right",  "cursor: right")
    add(generate_pressure_idle(0.4), "up",     "cursor: up")
    add(generate_pressure_idle(0.3), "right",  "cursor: right")
    idle_pen_up(0.2)

    # ── Left click ─────────────────────────────────────────────────────────
    add(generate_pressure_light_tap(1), "still", "left click")
    idle_pen_up(0.3)

    # ── Double click ───────────────────────────────────────────────────────
    add(generate_pressure_light_tap(2), "still", "double click")
    idle_pen_up(0.3)

    # ── Right click ────────────────────────────────────────────────────────
    add(generate_pressure_firm_press(), "still", "right click")
    idle_pen_up(0.3)

    # ── Drag ───────────────────────────────────────────────────────────────
    add(generate_pressure_tap_and_hold(0.5), "right", "drag right")
    idle_pen_up(0.3)

    # ── Scroll: plant + hold still 0.5s → tilt forward → tilt back ────────
    # Step 1: plant pen and hold perfectly still for confirmation
    idle_pen_down(0.6, "still")   # 0.6s > 0.5s needed → confirms

    # Step 2: tilt forward → scroll down
    idle_pen_down(0.7, "tilt_forward")

    # Step 3: tilt back → scroll up (still in SCROLL_READY, no re-confirm needed)
    idle_pen_down(0.7, "tilt_back")

    # Step 4: lift pen → exits scroll mode
    idle_pen_up(0.3)

    # ── Interrupted confirmation (should NOT scroll) ───────────────────────
    idle_pen_down(0.3, "still")     # only 0.3s → not enough
    idle_pen_down(0.2, "right")     # movement interrupts → back to IDLE
    idle_pen_down(0.2, "tilt_forward")  # tilt now → should NOT scroll (IDLE)
    idle_pen_up(0.3)

    # ── More cursor movement to finish ─────────────────────────────────────
    add(generate_pressure_idle(0.4), "down",  "cursor: down")
    add(generate_pressure_idle(0.3), "right", "cursor: right")
    idle_pen_up(0.3)

    return stream


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DARK  = "#0d0d0d"
PANEL = "#1a1a2e"

STATE_COLORS = {
    "IDLE"         : "#555555",
    "CONFIRMING"   : "#ffa726",
    "SCROLL_READY" : "#66bb6a",
}

ACTION_COLORS = {
    "LEFT CLICK"   : "#ef5350",
    "DOUBLE CLICK" : "#ffa726",
    "RIGHT CLICK"  : "#ab47bc",
    "DRAG START"   : "#4fc3f7",
    "DRAG END"     : "#80cbc4",
    "SCROLL DOWN"  : "#66bb6a",
    "SCROLL UP"    : "#26c6da",
    "Scroll ready ↕": "#66bb6a",
}

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(
    3, 2, figure=fig,
    hspace=0.55, wspace=0.35,
    left=0.06, right=0.97,
    top=0.90,  bottom=0.06,
)

ax_cursor  = fig.add_subplot(gs[0, 0])
ax_imu     = fig.add_subplot(gs[0, 1])
ax_press   = fig.add_subplot(gs[1, 0])
ax_state   = fig.add_subplot(gs[1, 1])
ax_log     = fig.add_subplot(gs[2, 0])
ax_scroll  = fig.add_subplot(gs[2, 1])

for ax in [ax_cursor, ax_imu, ax_press, ax_state, ax_log, ax_scroll]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


def run():
    controller  = WritingController()
    stream      = build_demo()

    HIST = 150
    cursor_x_hist   = []
    cursor_y_hist   = []
    pressure_hist   = []
    gyro_mag_hist   = []
    scroll_hist     = []
    scroll_total    = 0
    action_log_disp = []   # (label, color)

    # Confirmation progress history for the state panel
    confirm_hist    = []   # 0..1 values
    state_hist      = []   # state label strings

    print(f"\n  Pen cursor mode demo — {len(stream)} samples")
    print(f"  Scroll confirm threshold : {SCROLL_STILL_THRESHOLD} rad/s")
    print(f"  Scroll confirm duration  : {SCROLL_CONFIRM_SAMPLES/SAMPLE_RATE:.1f}s  ({SCROLL_CONFIRM_SAMPLES} samples)")
    print(f"\n  {'Frame':<7} {'Event':<25} {'Scroll state':<15} {'Progress'}")
    print("  " + "─" * 60)

    for frame, (pressure, gx, gy, gz, label) in enumerate(stream):

        prev_log_len = len(controller.action_log)
        status = controller.process_frame(
            pressure, gx, gy, slider_position=1, gyro_z=gz
        )

        # Track new log entries
        if len(controller.action_log) > prev_log_len:
            entry = controller.action_log[-1]
            color = ACTION_COLORS.get(entry, "#ffffff")
            action_log_disp.append((entry, color))
            if len(action_log_disp) > 8:
                action_log_disp.pop(0)

        # Scroll total
        if controller.action_log and "SCROLL DOWN" in controller.action_log[-1]:
            scroll_total -= 1
        elif controller.action_log and "SCROLL UP" in controller.action_log[-1]:
            scroll_total += 1

        # Buffers
        cx, cy = pyautogui.position()
        cursor_x_hist.append(cx)
        cursor_y_hist.append(cy)
        pressure_hist.append(pressure)
        gyro_mag_hist.append(status["gyro_mag"])
        scroll_hist.append(scroll_total)
        confirm_hist.append(status["scroll_progress"])
        state_hist.append(status["scroll_state"])

        for buf in [cursor_x_hist, cursor_y_hist, pressure_hist,
                    gyro_mag_hist, scroll_hist, confirm_hist, state_hist]:
            if len(buf) > HIST:
                buf.pop(0)

        if frame % 30 == 0:
            prog = f"{status['scroll_progress']*100:.0f}%"
            print(f"  {frame:<7} {label:<25} {status['scroll_state']:<15} {prog}")

        if frame % 4 != 0:
            continue

        scroll_state = status["scroll_state"]
        state_color  = STATE_COLORS.get(scroll_state, "#ffffff")
        progress     = status["scroll_progress"]

        # ── Panel 1: Cursor trail ──────────────────────────────────────────
        ax_cursor.cla(); ax_cursor.set_facecolor(PANEL)
        if len(cursor_x_hist) > 1:
            ax_cursor.plot(cursor_x_hist, cursor_y_hist,
                           color="#4fc3f7", linewidth=1.0, alpha=0.7)
            ax_cursor.scatter(cursor_x_hist[-1], cursor_y_hist[-1],
                              color="white", s=35, zorder=5)
        trail_title = "Cursor Trail"
        if status.get("dragging"):
            trail_title += "  ◉ DRAGGING"
        ax_cursor.set_title(trail_title,
                            color="#ffa726" if status.get("dragging") else "white",
                            fontsize=9, fontweight="bold")
        ax_cursor.invert_yaxis()
        ax_cursor.tick_params(colors="#aaa", labelsize=6)
        ax_cursor.set_xlabel("X", color="#aaa", fontsize=7)
        ax_cursor.set_ylabel("Y", color="#aaa", fontsize=7)

        # ── Panel 2: IMU bars ──────────────────────────────────────────────
        ax_imu.cla(); ax_imu.set_facecolor(PANEL)
        imu_labels = ["gyro_x\n(roll L/R)", "gyro_y\n(roll U/D)", "gyro_z\n(tilt)"]
        imu_vals   = [gx, gy, gz]
        imu_colors = ["#ef5350", "#66bb6a", "#4fc3f7"]
        ax_imu.bar(imu_labels, imu_vals, color=imu_colors, alpha=0.85)
        ax_imu.axhline(0, color="#555", linewidth=0.8)
        ax_imu.axhline( SCROLL_STILL_THRESHOLD, color="#ffa726",
                        linestyle="--", linewidth=0.8, alpha=0.6)
        ax_imu.axhline(-SCROLL_STILL_THRESHOLD, color="#ffa726",
                        linestyle="--", linewidth=0.8, alpha=0.6,
                        label=f"still threshold ±{SCROLL_STILL_THRESHOLD}")
        ax_imu.set_ylim(-2.0, 2.0)
        ax_imu.set_title("Pen IMU  (rad/s)", color="white", fontsize=9)
        ax_imu.legend(fontsize=6, facecolor=PANEL, labelcolor="white",
                      framealpha=0.4, loc="upper right")
        ax_imu.tick_params(colors="#aaa", labelsize=7)

        # ── Panel 3: Pressure + gyro magnitude ────────────────────────────
        ax_press.cla(); ax_press.set_facecolor(PANEL)
        ax_press.plot(pressure_hist, color="#f06292", linewidth=0.9,
                      label="pressure", alpha=0.9)
        ax_press.plot(gyro_mag_hist, color="#ffa726", linewidth=0.7,
                      label="gyro mag", alpha=0.7)
        ax_press.axhline(PEN_DOWN_THRESHOLD, color="#f06292",
                         linestyle="--", linewidth=0.8, alpha=0.6)
        ax_press.axhline(SCROLL_STILL_THRESHOLD, color="#ffa726",
                         linestyle="--", linewidth=0.8, alpha=0.6)
        ax_press.set_ylim(-0.05, 2.2)
        ax_press.set_title("Pressure  +  Gyro Magnitude", color="white", fontsize=9)
        ax_press.legend(fontsize=6, facecolor=PANEL, labelcolor="white",
                        framealpha=0.4)
        ax_press.set_xticks([])
        ax_press.tick_params(colors="#aaa")

        # ── Panel 4: Scroll state machine ──────────────────────────────────
        ax_state.cla(); ax_state.set_facecolor(PANEL)
        ax_state.set_xlim(0, 1); ax_state.set_ylim(0, 1)
        ax_state.set_xticks([]); ax_state.set_yticks([])
        ax_state.set_title("Scroll State Machine", color="white", fontsize=9)

        # State label
        ax_state.text(0.5, 0.78, scroll_state, color=state_color,
                      fontsize=18, fontweight="bold", ha="center",
                      transform=ax_state.transAxes)

        # Confirmation progress bar
        bar_w = 0.75
        bar_x = (1 - bar_w) / 2
        bar_h = 0.10
        bar_y = 0.42

        # Background track
        ax_state.add_patch(mpatches.FancyBboxPatch(
            (bar_x, bar_y), bar_w, bar_h,
            boxstyle="round,pad=0.01",
            facecolor="#333", edgecolor="#555",
            transform=ax_state.transAxes, clip_on=False
        ))
        # Fill
        if progress > 0:
            ax_state.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), bar_w * progress, bar_h,
                boxstyle="round,pad=0.01",
                facecolor=state_color, edgecolor="none",
                transform=ax_state.transAxes, clip_on=False
            ))
        ax_state.text(0.5, 0.30,
                      f"Confirm: {progress*100:.0f}%  ({int(progress * SCROLL_CONFIRM_SAMPLES)}/{SCROLL_CONFIRM_SAMPLES} samples)",
                      color="#aaa", fontsize=8, ha="center",
                      transform=ax_state.transAxes)

        # Description line
        descriptions = {
            "IDLE"        : "pen up, moving, or fiddling",
            "CONFIRMING"  : "planted & still — building intent…",
            "SCROLL_READY": "tilt to scroll  ↕",
        }
        ax_state.text(0.5, 0.14,
                      descriptions.get(scroll_state, ""),
                      color=state_color, fontsize=8, ha="center", style="italic",
                      transform=ax_state.transAxes)

        # ── Panel 5: Action log ────────────────────────────────────────────
        ax_log.cla(); ax_log.set_facecolor(PANEL)
        ax_log.set_xlim(0, 1); ax_log.set_ylim(0, 1)
        ax_log.set_xticks([]); ax_log.set_yticks([])
        ax_log.set_title("Action Log", color="white", fontsize=9)
        for li, (entry, col) in enumerate(reversed(action_log_disp[-6:])):
            alpha = max(1.0 - li * 0.15, 0.2)
            ax_log.text(0.05, 0.82 - li * 0.13, entry,
                        color=col, fontsize=9, fontweight="bold",
                        alpha=alpha, transform=ax_log.transAxes)

        # ── Panel 6: Scroll history ────────────────────────────────────────
        ax_scroll.cla(); ax_scroll.set_facecolor(PANEL)
        if scroll_hist:
            ax_scroll.fill_between(range(len(scroll_hist)), scroll_hist,
                                   color="#66bb6a", alpha=0.35)
            ax_scroll.plot(scroll_hist, color="#66bb6a", linewidth=1.0)
        ax_scroll.axhline(0, color="#555", linewidth=0.7)
        ax_scroll.set_title("Scroll Position  (↑ positive = scrolled up)",
                            color="white", fontsize=9)
        ax_scroll.set_xticks([]); ax_scroll.tick_params(colors="#aaa")
        ax_scroll.set_xlim(0, HIST)

        # ── Main title ─────────────────────────────────────────────────────
        pen_str   = "● PEN DOWN" if status["pen_down"] else "○ pen up"
        drag_str  = "  ◉ DRAGGING" if status.get("dragging") else ""
        fig.suptitle(
            f"EMG GLOVE  —  PEN CURSOR MODE  |  {pen_str}{drag_str}  |  {label}",
            fontsize=12, fontweight="bold", color=state_color, y=0.97,
        )

        plt.pause(0.01)

    print(f"\n  Demo complete — scroll position: {scroll_total}")
    plt.show()


# ---------------------------------------------------------------------------
# Hardware mode — live pen ESP32 data
# ---------------------------------------------------------------------------

def run_hardware():
    """
    Live hardware loop — reads pen ESP32 over USB serial and renders
    the same 6-panel dashboard as the simulator demo.
    Runs until the user closes the matplotlib window.
    """
    reader     = PenHardwareReader(port=PEN_HARDWARE_PORT)
    controller = WritingController()

    reader.start()
    time.sleep(0.5)

    HIST = 150
    cursor_x_hist   = []
    cursor_y_hist   = []
    pressure_hist   = []
    gyro_mag_hist   = []
    scroll_hist     = []
    scroll_total    = 0
    action_log_disp = []
    confirm_hist    = []
    state_hist      = []

    print(f"\n  Pen mode — LIVE HARDWARE on {PEN_HARDWARE_PORT}")
    print(f"  Scroll confirm threshold : {SCROLL_STILL_THRESHOLD} rad/s")
    print(f"  Scroll confirm duration  : {SCROLL_CONFIRM_SAMPLES/SAMPLE_RATE:.1f}s  ({SCROLL_CONFIRM_SAMPLES} samples)")
    print(f"  Close the plot window to stop.")
    print(f"\n  {'Frame':<7} {'GX':>6} {'GY':>6} {'GZ':>6} {'Pressure':<8} {'Dock':<6} {'Slide':<6} {'State / Text':<20}")
    print("  " + "─" * 75)

    frame = 0
    try:
        while plt.fignum_exists(fig.number):
            imu      = reader.get_imu_sample()
            pressure = reader.get_pressure()
            slider   = reader.get_slider_position()
            gx, gy, gz = imu["gyro_x"], imu["gyro_y"], imu["gyro_z"]

            prev_log_len = len(controller.action_log)
            status = controller.process_frame(
                pressure, gx, gy, slider_position=slider, gyro_z=gz
            )

            # Track new log entries
            if len(controller.action_log) > prev_log_len:
                entry = controller.action_log[-1]
                color = ACTION_COLORS.get(entry, "#ffffff")
                action_log_disp.append((entry, color))
                if len(action_log_disp) > 8:
                    action_log_disp.pop(0)

            # Scroll total
            if controller.action_log and "SCROLL DOWN" in controller.action_log[-1]:
                scroll_total -= 1
            elif controller.action_log and "SCROLL UP" in controller.action_log[-1]:
                scroll_total += 1

            # Buffers
            cx, cy = pyautogui.position()
            cursor_x_hist.append(cx)
            cursor_y_hist.append(cy)
            pressure_hist.append(pressure)
            gyro_mag_hist.append(status.get("gyro_mag", 0.0))
            scroll_hist.append(scroll_total)
            confirm_hist.append(status.get("scroll_progress", 0.0))
            state_hist.append(status.get("scroll_state", "IDLE"))

            for buf in [cursor_x_hist, cursor_y_hist, pressure_hist,
                        gyro_mag_hist, scroll_hist, confirm_hist, state_hist]:
                if len(buf) > HIST:
                    buf.pop(0)

            if frame % 10 == 0:
                dock_str = "YES" if reader.get_pen_docked() else "NO"
                if slider == 1:
                    prog = f"{status.get('scroll_progress', 0)*100:.0f}%"
                    state_text = f"{status.get('scroll_state', 'IDLE')} {prog}"
                else:
                    state_text = f"Writing: '{status.get('output_text', '').strip()}'"
                
                print(f"  {frame:<7} {gx:>6.2f} {gy:>6.2f} {gz:>6.2f} {pressure:<8.3f} {dock_str:<6} {slider:<6} {state_text:<20}")

            if frame % 4 != 0:
                frame += 1
                time.sleep(1.0 / SAMPLE_RATE)
                continue

            scroll_state = status.get("scroll_state", "IDLE")
            state_color  = STATE_COLORS.get(scroll_state, "#ffffff")
            progress     = status.get("scroll_progress", 0.0)
            label        = f"slider={slider}  press={pressure:.2f}"

            # ── Panel 1: Cursor trail ──────────────────────────────────────
            ax_cursor.cla(); ax_cursor.set_facecolor(PANEL)
            if len(cursor_x_hist) > 1:
                ax_cursor.plot(cursor_x_hist, cursor_y_hist,
                               color="#4fc3f7", linewidth=1.0, alpha=0.7)
                ax_cursor.scatter(cursor_x_hist[-1], cursor_y_hist[-1],
                                  color="white", s=35, zorder=5)
            trail_title = "Cursor Trail"
            if status.get("dragging"):
                trail_title += "  ◉ DRAGGING"
            ax_cursor.set_title(trail_title,
                                color="#ffa726" if status.get("dragging") else "white",
                                fontsize=9, fontweight="bold")
            ax_cursor.invert_yaxis()
            ax_cursor.tick_params(colors="#aaa", labelsize=6)
            ax_cursor.set_xlabel("X", color="#aaa", fontsize=7)
            ax_cursor.set_ylabel("Y", color="#aaa", fontsize=7)

            # ── Panel 2: IMU bars ──────────────────────────────────────────
            ax_imu.cla(); ax_imu.set_facecolor(PANEL)
            imu_labels = ["gyro_x\n(roll L/R)", "gyro_y\n(roll U/D)", "gyro_z\n(tilt)"]
            imu_vals   = [gx, gy, gz]
            imu_colors = ["#ef5350", "#66bb6a", "#4fc3f7"]
            ax_imu.bar(imu_labels, imu_vals, color=imu_colors, alpha=0.85)
            ax_imu.axhline(0, color="#555", linewidth=0.8)
            ax_imu.axhline( SCROLL_STILL_THRESHOLD, color="#ffa726",
                            linestyle="--", linewidth=0.8, alpha=0.6)
            ax_imu.axhline(-SCROLL_STILL_THRESHOLD, color="#ffa726",
                            linestyle="--", linewidth=0.8, alpha=0.6,
                            label=f"still threshold ±{SCROLL_STILL_THRESHOLD}")
            ax_imu.set_ylim(-2.0, 2.0)
            ax_imu.set_title("Pen IMU  (rad/s)  — LIVE", color="white", fontsize=9)
            ax_imu.legend(fontsize=6, facecolor=PANEL, labelcolor="white",
                          framealpha=0.4, loc="upper right")
            ax_imu.tick_params(colors="#aaa", labelsize=7)

            # ── Panel 3: Pressure + gyro magnitude ────────────────────────
            ax_press.cla(); ax_press.set_facecolor(PANEL)
            ax_press.plot(pressure_hist, color="#f06292", linewidth=0.9,
                          label="pressure", alpha=0.9)
            ax_press.plot(gyro_mag_hist, color="#ffa726", linewidth=0.7,
                          label="gyro mag", alpha=0.7)
            ax_press.axhline(PEN_DOWN_THRESHOLD, color="#f06292",
                             linestyle="--", linewidth=0.8, alpha=0.6)
            ax_press.axhline(SCROLL_STILL_THRESHOLD, color="#ffa726",
                             linestyle="--", linewidth=0.8, alpha=0.6)
            ax_press.set_ylim(-0.05, 2.2)
            ax_press.set_title("Pressure  +  Gyro Magnitude", color="white", fontsize=9)
            ax_press.legend(fontsize=6, facecolor=PANEL, labelcolor="white",
                            framealpha=0.4)
            ax_press.set_xticks([])
            ax_press.tick_params(colors="#aaa")

            # ── Panel 4: Scroll state machine ──────────────────────────────
            ax_state.cla(); ax_state.set_facecolor(PANEL)
            ax_state.set_xlim(0, 1); ax_state.set_ylim(0, 1)
            ax_state.set_xticks([]); ax_state.set_yticks([])
            ax_state.set_title("Scroll State Machine", color="white", fontsize=9)
            ax_state.text(0.5, 0.78, scroll_state, color=state_color,
                          fontsize=18, fontweight="bold", ha="center",
                          transform=ax_state.transAxes)
            bar_w = 0.75
            bar_x = (1 - bar_w) / 2
            bar_h = 0.10
            bar_y = 0.42
            ax_state.add_patch(mpatches.FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle="round,pad=0.01",
                facecolor="#333", edgecolor="#555",
                transform=ax_state.transAxes, clip_on=False
            ))
            if progress > 0:
                ax_state.add_patch(mpatches.FancyBboxPatch(
                    (bar_x, bar_y), bar_w * progress, bar_h,
                    boxstyle="round,pad=0.01",
                    facecolor=state_color, edgecolor="none",
                    transform=ax_state.transAxes, clip_on=False
                ))
            ax_state.text(0.5, 0.30,
                          f"Confirm: {progress*100:.0f}%  ({int(progress * SCROLL_CONFIRM_SAMPLES)}/{SCROLL_CONFIRM_SAMPLES} samples)",
                          color="#aaa", fontsize=8, ha="center",
                          transform=ax_state.transAxes)
            descriptions = {
                "IDLE"        : "pen up, moving, or fiddling",
                "CONFIRMING"  : "planted & still — building intent…",
                "SCROLL_READY": "tilt to scroll  ↕",
            }
            ax_state.text(0.5, 0.14,
                          descriptions.get(scroll_state, ""),
                          color=state_color, fontsize=8, ha="center", style="italic",
                          transform=ax_state.transAxes)

            # ── Panel 5: Action log ────────────────────────────────────────
            ax_log.cla(); ax_log.set_facecolor(PANEL)
            ax_log.set_xlim(0, 1); ax_log.set_ylim(0, 1)
            ax_log.set_xticks([]); ax_log.set_yticks([])
            ax_log.set_title("Action Log", color="white", fontsize=9)
            for li, (entry, col) in enumerate(reversed(action_log_disp[-6:])):
                alpha = max(1.0 - li * 0.15, 0.2)
                ax_log.text(0.05, 0.82 - li * 0.13, entry,
                            color=col, fontsize=9, fontweight="bold",
                            alpha=alpha, transform=ax_log.transAxes)

            # ── Panel 6: Scroll history ────────────────────────────────────
            ax_scroll.cla(); ax_scroll.set_facecolor(PANEL)
            if scroll_hist:
                ax_scroll.fill_between(range(len(scroll_hist)), scroll_hist,
                                       color="#66bb6a", alpha=0.35)
                ax_scroll.plot(scroll_hist, color="#66bb6a", linewidth=1.0)
            ax_scroll.axhline(0, color="#555", linewidth=0.7)
            ax_scroll.set_title("Scroll Position  (↑ positive = scrolled up)",
                                color="white", fontsize=9)
            ax_scroll.set_xticks([]); ax_scroll.tick_params(colors="#aaa")
            ax_scroll.set_xlim(0, HIST)

            # ── Main title ─────────────────────────────────────────────────
            pen_str   = "● PEN DOWN" if status["pen_down"] else "○ pen up"
            drag_str  = "  ◉ DRAGGING" if status.get("dragging") else ""
            mode_str  = "PEN CURSOR MODE" if slider == 1 else "PEN KEYBOARD MODE"
            text_str  = f" | TEXT: {status.get('output_text', '').strip()}" if slider == 2 else ""
            
            fig.suptitle(
                f"EMG GLOVE  —  {mode_str}  [HARDWARE]  |  {pen_str}{drag_str}  |  {label}{text_str}",
                fontsize=12, fontweight="bold", color=state_color, y=0.97,
            )

            plt.pause(0.01)
            frame += 1
            time.sleep(1.0 / SAMPLE_RATE)

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        reader.stop()
        print(f"  Hardware session ended — scroll position: {scroll_total}")
        plt.show()


def check_hardware_available(port):
    import serial
    try:
        s = serial.Serial(port)
        s.close()
        return True
    except Exception:
        return False

if __name__ == "__main__":
    print(f"\n  [INIT] Checking for pen hardware on {PEN_HARDWARE_PORT}...")
    
    if check_hardware_available(PEN_HARDWARE_PORT):
        print("  [STATUS] HARDWARE FOUND! Starting live hardware mode.\n")
        run_hardware()
    else:
        print("  [STATUS] HARDWARE NOT FOUND. Falling back to simulated data.\n")
        run()