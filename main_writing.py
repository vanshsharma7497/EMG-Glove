"""
main_writing.py
---------------
Live dashboard for the pen writing mode pipeline.

Runs a scripted demo sequence showing the full pipeline end to end:
  pen sensor data → stroke segmentation → letter recognition →
  language model → text output

Six panels:
  Top-left    : Live stroke path (x,y coordinates being drawn)
  Top-right   : Pressure signal with pen-down threshold line
  Mid-left    : Letter buffer + committed text output
  Mid-right   : Word completion suggestions
  Bot-left    : Action log (recent events)
  Bot-right   : Per-letter confidence bar chart

Run from GLOVE/ root:
    python main_writing.py
"""

import sys
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ── Mock pyautogui so the dashboard runs without a display server ─────────────
# Only inject the mock if pyautogui hasn't already been imported — this prevents
# clobbering a real pyautogui instance if this module is imported alongside
# other code that needs the real thing.
if "pyautogui" not in sys.modules:
    _pg = types.ModuleType("pyautogui")
    _pg.FAILSAFE   = True
    _pg.PAUSE      = 0
    _pg.position   = lambda: (500, 400)
    _pg.size       = lambda: (1920, 1080)
    _pg.moveTo     = lambda x, y, duration=0: None
    _pg.click      = lambda button="left": None
    _pg.doubleClick = lambda button="left": None
    _pg.press      = lambda k: None
    _pg.typewrite  = lambda t, interval=0: None
    sys.modules["pyautogui"] = _pg

# ── Now safe to import pipeline modules ───────────────────────────────────────
import core.pen_simulator as pen_sim
from core.pen_simulator      import (
    generate_pressure_writing, generate_pressure_idle,
    generate_pressure_light_tap, SAMPLE_RATE,
    PEN_DOWN_THRESHOLD,
)
from core.letter_templates   import LETTER_TEMPLATES, _resample
from core.pen_stroke_processor import STROKE_SCALE
from modes.writing_mode      import WritingController

SAMPLE_PERIOD = 1.0 / SAMPLE_RATE

# ---------------------------------------------------------------------------
# Demo sequence — words to write, one letter at a time
# Each entry: (word_string, stroke_duration_seconds)
# ---------------------------------------------------------------------------
DEMO_WORDS = [
    ("love",  0.55),
    ("is",    0.55),
    ("real",  0.55),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def template_to_gyro(letter, duration):
    """Derive gyro values from a letter template (same as writing_mode test)."""
    path      = LETTER_TEMPLATES[letter]
    n_samples = int(duration * SAMPLE_RATE)
    resampled = _resample(path, n=n_samples)
    dx = np.diff(resampled[:, 0], prepend=resampled[0, 0])
    dy = np.diff(resampled[:, 1], prepend=resampled[0, 1])
    gyro_x = (dx / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
    gyro_y = (dy / (STROKE_SCALE * SAMPLE_PERIOD)) * 2.0
    return gyro_x, gyro_y


# ---------------------------------------------------------------------------
# Build full sample stream for the demo
# Each entry: (pressure, gyro_x, gyro_y, label)
# label is used only for the terminal print — not fed to the model
# ---------------------------------------------------------------------------

def build_demo_stream():
    """
    Returns a list of (pressure, gyro_x, gyro_y, event_label) tuples.
    Represents the full sensor stream for the demo sequence.
    """
    stream = []

    def add_idle(duration, label="idle"):
        p = generate_pressure_idle(duration)
        for val in p:
            stream.append((val, 0.0, 0.0, label))

    def add_stroke(letter, duration):
        p      = generate_pressure_writing(duration)
        gx, gy = template_to_gyro(letter, duration)
        n      = min(len(p), len(gx))
        for i in range(n):
            stream.append((p[i], gx[i], gy[i], f"stroke:{letter}"))

    def add_double_tap():
        tap = generate_pressure_light_tap(n_taps=2)
        for val in tap:
            stream.append((val, 0.0, 0.0, "double_tap"))

    add_idle(0.3, "start")

    for word, dur in DEMO_WORDS:
        for letter in word:
            add_stroke(letter, dur)
            add_idle(0.18, f"gap:{letter}")
        add_double_tap()
        add_idle(0.6, "between_words")   # pause between words

    add_idle(0.4, "end")

    return stream


# ---------------------------------------------------------------------------
# Dashboard setup
# ---------------------------------------------------------------------------

DARK  = "#0d0d0d"
PANEL = "#1a1a2e"
ACCENT_COLORS = {
    # Original 11
    "l": "#ef5350", "o": "#ffa726", "v": "#66bb6a",
    "e": "#4fc3f7", "i": "#ab47bc", "s": "#26c6da",
    "r": "#f06292", "a": "#80cbc4", "n": "#fff176",
    "c": "#ce93d8", "u": "#a5d6a7",
    # Remaining 15
    "b": "#ff8a65", "d": "#4db6ac", "f": "#dce775",
    "g": "#4dd0e1", "h": "#f48fb1", "j": "#aed581",
    "k": "#ffb74d", "m": "#ba68c8", "p": "#4fc3f7",
    "q": "#ff7043", "t": "#81c784", "w": "#64b5f6",
    "x": "#a1887f", "y": "#90a4ae", "z": "#ffe082",
}
DEFAULT_COLOR = "#ffffff"

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(DARK)
gs  = gridspec.GridSpec(
    3, 2, figure=fig,
    hspace=0.5, wspace=0.35,
    left=0.06, right=0.97,
    top=0.90,  bottom=0.06,
)

ax_stroke   = fig.add_subplot(gs[0, 0])   # live stroke path
ax_pressure = fig.add_subplot(gs[0, 1])   # pressure signal
ax_text     = fig.add_subplot(gs[1, 0])   # letter buffer + output text
ax_complete = fig.add_subplot(gs[1, 1])   # word completions
ax_log      = fig.add_subplot(gs[2, 0])   # action log
ax_conf     = fig.add_subplot(gs[2, 1])   # confidence bars

for ax in [ax_stroke, ax_pressure, ax_text, ax_complete, ax_log, ax_conf]:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_dashboard():
    controller   = WritingController()
    stream       = build_demo_stream()

    # History buffers for display
    pressure_history = []
    stroke_x_live    = []   # current stroke being drawn
    stroke_y_live    = []
    conf_history     = []   # (letter, confidence) pairs
    HISTORY_LEN      = 120  # pressure samples to show

    print(f"\n  Running writing mode demo — {len(stream)} samples\n")
    print(f"  {'Sample':<8} {'Event':<20} {'Buffer':<15} {'Output'}")
    print("  " + "─" * 58)

    frame = 0
    word_just_committed = False

    for pressure, gyro_x, gyro_y, event_label in stream:

        # ── Feed to controller ─────────────────────────────────────────────
        prev_output = controller.language_model.output_text
        prev_word   = controller.current_word

        status = controller.process_frame(
            pressure, gyro_x, gyro_y, slider_position=2
        )

        # ── Track live stroke path ─────────────────────────────────────────
        if status["pen_down"]:
            # Accumulate current stroke coordinates from processor
            if controller.stroke_processor.stroke_x:
                stroke_x_live = list(controller.stroke_processor.stroke_x)
                stroke_y_live = list(controller.stroke_processor.stroke_y)
        else:
            if stroke_x_live:  # pen just lifted — freeze path briefly
                pass           # keep displaying until next stroke starts

        # Clear stroke display when a new stroke begins
        if (not stroke_x_live and status["pen_down"]
                and controller.stroke_processor.state == "RECORDING"):
            stroke_x_live = []
            stroke_y_live = []

        # Reset stroke trail when pen lifts
        if prev_word != controller.current_word and not status["pen_down"]:
            stroke_x_live = []
            stroke_y_live = []

        # ── Pressure history ───────────────────────────────────────────────
        pressure_history.append(pressure)
        if len(pressure_history) > HISTORY_LEN:
            pressure_history.pop(0)

        # ── Confidence tracking ────────────────────────────────────────────
        if len(controller.action_log) > 0:
            last = controller.action_log[-1]
            if last.startswith("Stroke →") and (
                not conf_history or conf_history[-1][2] != last
            ):
                # Parse: "Stroke → 'x' (conf=0.99)"
                try:
                    parts  = last.split("'")
                    letter = parts[1]
                    conf   = float(last.split("conf=")[1].split(")")[0])
                    conf_history.append((letter, conf, last))
                    if len(conf_history) > 8:
                        conf_history.pop(0)
                except Exception:
                    pass

        # ── Terminal print (every 20 frames) ──────────────────────────────
        if frame % 20 == 0:
            print(f"  {frame:<8} {event_label:<20} "
                  f"'{controller.current_word}'{'':>10} "
                  f"'{controller.language_model.output_text.strip()}'")

        # ── Dashboard update (every 4 frames for speed) ───────────────────
        if frame % 4 == 0:

            word   = controller.current_word
            output = controller.language_model.output_text.strip()

            # ── Panel 1: Live stroke path ──────────────────────────────────
            ax_stroke.cla(); ax_stroke.set_facecolor(PANEL)
            if stroke_x_live and stroke_y_live:
                letter_hint = word[-1] if word else "?"
                color = ACCENT_COLORS.get(letter_hint, DEFAULT_COLOR)
                ax_stroke.plot(stroke_x_live, stroke_y_live,
                               color=color, linewidth=2.0)
                ax_stroke.scatter(stroke_x_live[0],  stroke_y_live[0],
                                  color="lime",   s=40, zorder=5)
                ax_stroke.scatter(stroke_x_live[-1], stroke_y_live[-1],
                                  color="yellow", s=40, zorder=5)
            ax_stroke.set_title("Live Stroke Path", color="white", fontsize=9)
            ax_stroke.invert_yaxis()
            ax_stroke.set_xticks([]); ax_stroke.set_yticks([])
            if status["pen_down"]:
                ax_stroke.set_title("Live Stroke Path  ● WRITING",
                                    color="#66bb6a", fontsize=9, fontweight="bold")

            # ── Panel 2: Pressure signal ───────────────────────────────────
            ax_pressure.cla(); ax_pressure.set_facecolor(PANEL)
            ax_pressure.plot(pressure_history, color="#4fc3f7", linewidth=0.8)
            ax_pressure.axhline(PEN_DOWN_THRESHOLD, color="orange",
                                linestyle="--", linewidth=0.9,
                                label=f"Pen-down ({PEN_DOWN_THRESHOLD})")
            ax_pressure.set_ylim(-0.05, 1.05)
            ax_pressure.set_title("Pressure Signal", color="white", fontsize=9)
            ax_pressure.legend(fontsize=6, facecolor=PANEL,
                               labelcolor="white", framealpha=0.5)
            ax_pressure.set_xticks([]); ax_pressure.tick_params(colors="#aaa")

            # ── Panel 3: Letter buffer + output text ───────────────────────
            ax_text.cla(); ax_text.set_facecolor(PANEL)
            ax_text.set_xlim(0, 1); ax_text.set_ylim(0, 1)
            ax_text.set_xticks([]); ax_text.set_yticks([])

            # Committed text
            ax_text.text(0.05, 0.75, "Output:", color="#888",
                         fontsize=8, transform=ax_text.transAxes)
            ax_text.text(0.05, 0.55, f'"{output}"' if output else '—',
                         color="white", fontsize=14, fontweight="bold",
                         transform=ax_text.transAxes)

            # In-progress word — show each letter in its accent colour
            ax_text.text(0.05, 0.30, "Writing:", color="#888",
                         fontsize=8, transform=ax_text.transAxes)
            x_pos = 0.05
            for ch in word:
                col = ACCENT_COLORS.get(ch, DEFAULT_COLOR)
                ax_text.text(x_pos, 0.12, ch, color=col,
                             fontsize=20, fontweight="bold",
                             transform=ax_text.transAxes)
                x_pos += 0.07

            ax_text.set_title("Text Output", color="white", fontsize=9)

            # ── Panel 4: Completions ───────────────────────────────────────
            ax_complete.cla(); ax_complete.set_facecolor(PANEL)
            ax_complete.set_xlim(0, 1); ax_complete.set_ylim(0, 1)
            ax_complete.set_xticks([]); ax_complete.set_yticks([])
            ax_complete.set_title("Completions", color="white", fontsize=9)

            completions = status["completions"]
            if completions:
                for ci, comp in enumerate(completions[:3]):
                    y = 0.70 - ci * 0.28
                    # Highlight matched prefix in colour
                    prefix = word
                    ax_complete.text(0.08, y, comp, color="#4fc3f7",
                                     fontsize=13, fontweight="bold",
                                     transform=ax_complete.transAxes)
            else:
                ax_complete.text(0.08, 0.45, "—", color="#555",
                                 fontsize=14, transform=ax_complete.transAxes)

            # ── Panel 5: Action log ────────────────────────────────────────
            ax_log.cla(); ax_log.set_facecolor(PANEL)
            ax_log.set_xlim(0, 1); ax_log.set_ylim(0, 1)
            ax_log.set_xticks([]); ax_log.set_yticks([])
            ax_log.set_title("Action Log", color="white", fontsize=9)

            log = controller.action_log[-6:] if controller.action_log else []
            for li, entry in enumerate(reversed(log)):
                alpha = 1.0 - li * 0.15
                col   = "#4fc3f7"
                if "COMMIT" in entry:   col = "#66bb6a"
                if "BACKSPACE" in entry: col = "#ef5350"
                if "low confidence" in entry: col = "#ffa726"
                ax_log.text(0.04, 0.82 - li * 0.14, entry,
                            color=col, fontsize=7,
                            alpha=max(alpha, 0.25),
                            transform=ax_log.transAxes)

            # ── Panel 6: Confidence bars ───────────────────────────────────
            ax_conf.cla(); ax_conf.set_facecolor(PANEL)
            ax_conf.set_title("Recognition Confidence", color="white", fontsize=9)

            if conf_history:
                labels = [f"'{h[0]}'" for h in conf_history]
                values = [h[1] for h in conf_history]
                colors = [ACCENT_COLORS.get(h[0], DEFAULT_COLOR)
                          for h in conf_history]
                bars = ax_conf.bar(range(len(values)), values,
                                   color=colors, alpha=0.8)
                ax_conf.set_xticks(range(len(labels)))
                ax_conf.set_xticklabels(labels, color="white", fontsize=8)
                ax_conf.set_ylim(0, 1.05)
                ax_conf.axhline(0.25, color="orange", linestyle="--",
                                linewidth=0.8, label="threshold")
                ax_conf.tick_params(colors="#aaa")
                ax_conf.legend(fontsize=6, facecolor=PANEL,
                               labelcolor="white", framealpha=0.5)
            else:
                ax_conf.text(0.5, 0.5, "Waiting...", color="#555",
                             fontsize=10, ha="center", va="center",
                             transform=ax_conf.transAxes)

            # ── Title ──────────────────────────────────────────────────────
            pen_state = "✎ PEN DOWN" if status["pen_down"] else "○ pen up"
            fig.suptitle(
                f"EMG GLOVE  —  PEN WRITING MODE  |  {pen_state}",
                fontsize=14, fontweight="bold",
                color="#66bb6a" if status["pen_down"] else "#4fc3f7",
                y=0.97,
            )

            plt.pause(0.01)

        frame += 1

        # Reset cooldown between words (simulation timing fix)
        if event_label == "between_words":
            controller.last_tap_time = 0.0

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n  {'─' * 58}")
    print(f"  Demo complete.")
    print(f"  Final output : '{controller.language_model.output_text.strip()}'")
    print(f"  Expected     : 'love is real'")
    success = controller.language_model.output_text.strip() == "love is real"
    print(f"  Result       : {'✓ PASS' if success else '✗ partial (check log)'}")
    print(f"  {'─' * 58}\n")

    plt.show()


if __name__ == "__main__":
    run_dashboard()