"""
core/pen_simulator.py
---------------------
Simulates the two sensors on the pen attachment:
  1. Pressure sensor  — continuous force value 0.0 to 1.0
  2. Pen IMU          — rolling velocity in X and Y axes (rad/s)

Also manages three state variables that the rest of the pipeline reads:
  - slider_position  : int  — 1 = cursor mode, 2 = writing mode
  - pen_docked       : bool — True = pen on glove, EMG active; False = pen active
  - pen_down         : bool — True = tip is pressing on surface

When real hardware arrives, replace the generate_* functions below with
serial port reads from the ESP32. Everything else in the pipeline stays
identical.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds — adjust these when tuning real hardware signal levels
# ---------------------------------------------------------------------------

PEN_DOWN_THRESHOLD   = 0.25   # pressure above this → pen is touching surface
FIRM_PRESS_THRESHOLD = 0.75   # pressure above this → firm press (right-click)
SAMPLE_RATE          = 200    # pressure sensor samples per second
NOISE_LEVEL          = 0.02   # small random noise added to all signals

# ---------------------------------------------------------------------------
# Pen state — the rest of the pipeline reads these directly
# ---------------------------------------------------------------------------

pen_docked       = True    # start docked: glove is in control
slider_position  = 2       # 1 = cursor mode, 2 = writing mode
pen_down         = False   # tip is not currently on surface

# ---------------------------------------------------------------------------
# Pressure pattern generators
# Each function returns a numpy array of pressure values over time.
# Duration is in seconds. Output is sampled at SAMPLE_RATE.
# ---------------------------------------------------------------------------

def _noise(length):
    """Small gaussian noise added to every signal to make it realistic."""
    return np.random.normal(0, NOISE_LEVEL, length)


def generate_pressure_idle(duration=0.5):
    """
    Pen lifted / resting — pressure near zero.
    This is the baseline between all other actions.
    """
    n = int(duration * SAMPLE_RATE)
    signal = np.zeros(n) + 0.03   # tiny baseline, not exactly 0
    signal += _noise(n)
    signal = np.clip(signal, 0, 1)
    return signal


def generate_pressure_writing(duration=1.0):
    """
    Pen held on surface while drawing a stroke — sustained mid-level pressure.
    Slight drift and noise simulates the natural variation of handwriting pressure.
    """
    n = int(duration * SAMPLE_RATE)
    # Sustained pressure around 0.45, with slow drift and noise
    base = 0.45
    drift = np.linspace(0, np.random.uniform(-0.08, 0.08), n)
    signal = base + drift + _noise(n)
    signal = np.clip(signal, 0, 1)
    return signal


def generate_pressure_light_tap(n_taps=1, tap_gap=0.15):
    """
    Light tap(s) — short pressure spike that rises and falls quickly.

    n_taps   : 1 = single tap (left click), 2 = double tap, 3 = triple tap
    tap_gap  : time in seconds between taps (only relevant for n_taps > 1)

    Returns a pressure array covering the full tap sequence with idle padding.
    """
    TAP_DURATION   = 0.08    # each individual tap lasts 80ms
    PADDING        = 0.05    # idle time before first tap and after last tap
    GAP_DURATION   = tap_gap # idle time between taps

    tap_n     = int(TAP_DURATION * SAMPLE_RATE)
    pad_n     = int(PADDING * SAMPLE_RATE)
    gap_n     = int(GAP_DURATION * SAMPLE_RATE)

    # Build one tap: smooth rise and fall using a sine half-wave
    t         = np.linspace(0, np.pi, tap_n)
    one_tap   = 0.5 * np.sin(t)    # peaks at 0.5 — clearly above PEN_DOWN but below FIRM_PRESS

    segments  = [np.zeros(pad_n) + 0.03]   # leading idle

    for i in range(n_taps):
        segments.append(one_tap.copy())
        if i < n_taps - 1:
            segments.append(np.zeros(gap_n) + 0.03)   # gap between taps

    segments.append(np.zeros(pad_n) + 0.03)            # trailing idle

    signal = np.concatenate(segments)
    signal += _noise(len(signal))
    signal = np.clip(signal, 0, 1)
    return signal


def generate_pressure_firm_press(duration=0.4):
    """
    Firm press — pressure rises above FIRM_PRESS_THRESHOLD and stays there.
    Used for right-click in cursor mode, or text selection start in writing mode.
    """
    n         = int(duration * SAMPLE_RATE)
    ramp_n    = int(0.05 * SAMPLE_RATE)    # 50ms ramp up
    sustain_n = n - 2 * ramp_n
    ramp_down = int(0.05 * SAMPLE_RATE)   # 50ms ramp down

    ramp_up   = np.linspace(0.03, 0.82, ramp_n)
    sustain   = np.ones(sustain_n) * 0.82
    ramp_dn   = np.linspace(0.82, 0.03, ramp_down)

    signal    = np.concatenate([ramp_up, sustain, ramp_dn])
    signal   += _noise(len(signal))
    signal    = np.clip(signal, 0, 1)
    return signal


def generate_pressure_tap_and_hold(hold_duration=0.6):
    """
    Tap that doesn't release — used for click-and-drag.
    Pressure spikes as with a tap but then stays elevated (writing-level pressure).
    """
    tap_n  = int(0.08 * SAMPLE_RATE)
    hold_n = int(hold_duration * SAMPLE_RATE)

    t      = np.linspace(0, np.pi / 2, tap_n)       # half sine — rises but doesn't fall
    tap    = 0.5 * np.sin(t)
    hold   = np.ones(hold_n) * 0.48                  # stays elevated

    signal = np.concatenate([tap, hold])
    signal += _noise(len(signal))
    signal = np.clip(signal, 0, 1)
    return signal


# ---------------------------------------------------------------------------
# IMU generators — same principle as imu_simulator.py for the thumb IMU
# ---------------------------------------------------------------------------

MAX_GYRO           = 2.0    # maximum meaningful gyro reading (rad/s)
CURSOR_SENSITIVITY = 18.0  # pixels per frame in cursor mode
DEAD_ZONE          = 0.08  # ignore movements below this threshold (rad/s)

# Tilt-to-scroll constants
# Tilt is the pen body leaning forward/back while the tip stays planted.
# On the IMU this appears as a sustained gyro_z reading.
TILT_DEAD_ZONE     = 0.12  # tilt below this is ignored (prevents drift scrolling)
TILT_SCROLL_SPEED  = 0.15  # scroll units per sample at full tilt
MAX_TILT           = 1.5   # gyro_z reading that counts as full tilt


def generate_imu_sample(motion="still"):
    """
    Generates one IMU sample: (gyro_x, gyro_y, gyro_z) in rad/s.

    gyro_x, gyro_y : pen rolling left/right and up/down → cursor movement
    gyro_z         : pen body tilting forward/back (tip planted) → scroll

    motion options:
      "still"        — pen resting, no movement
      "right"        — rolling right  → cursor right
      "left"         — rolling left   → cursor left
      "up"           — rolling up     → cursor up
      "down"         — rolling down   → cursor down
      "tilt_forward" — pen tilts forward (tip planted) → scroll down
      "tilt_back"    — pen tilts backward              → scroll up
    """
    noise_x = np.random.normal(0, 0.03)
    noise_y = np.random.normal(0, 0.03)
    noise_z = np.random.normal(0, 0.02)

    # (gyro_x, gyro_y, gyro_z)
    motion_map = {
        "still"        : (0.0,   0.0,   0.0 ),
        "right"        : (0.8,   0.05,  0.0 ),
        "left"         : (-0.8,  0.05,  0.0 ),
        "up"           : (0.05,  -0.7,  0.0 ),
        "down"         : (0.05,   0.7,  0.0 ),
        "tilt_forward" : (0.0,    0.0,   1.1 ),  # pen tips forward → scroll down
        "tilt_back"    : (0.0,    0.0,  -1.1 ),  # pen tips back    → scroll up
    }

    if motion in motion_map:
        gx, gy, gz = motion_map[motion]
    else:
        gx, gy, gz = 0.0, 0.0, 0.0

    return (gx + noise_x, gy + noise_y, gz + noise_z)


def imu_to_cursor_velocity(gyro_x, gyro_y):
    """
    Converts raw gyro_x/gyro_y to (dx, dy) cursor velocity in pixels.
    gyro_z is intentionally excluded here — it is the tilt axis used for
    scrolling and is handled separately by imu_to_scroll().
    Processing: dead zone → normalise → acceleration curve (same as thumb IMU).
    """
    def process_axis(val):
        if abs(val) < DEAD_ZONE:
            return 0.0
        normalised  = np.clip(val / MAX_GYRO, -1.0, 1.0)
        sign        = np.sign(normalised)
        accelerated = sign * (normalised ** 2)
        return accelerated * CURSOR_SENSITIVITY

    dx = process_axis(gyro_x)
    dy = process_axis(gyro_y)
    return dx, dy


def imu_to_scroll(gyro_z):
    """
    Converts pen tilt (gyro_z) to a scroll amount.

    The pen tip stays planted on the surface while the barrel tilts:
      gyro_z > 0  → pen tips forward  → scroll down (content moves up)
      gyro_z < 0  → pen tips back     → scroll up   (content moves down)

    Returns a float: positive = scroll down, negative = scroll up.
    The scroll amount accumulates in the controller and fires as integer
    pyautogui.scroll() calls — same sub-unit accumulation as cursor pixels.
    """
    if abs(gyro_z) < TILT_DEAD_ZONE:
        return 0.0
    normalised = np.clip(gyro_z / MAX_TILT, -1.0, 1.0)
    return normalised * TILT_SCROLL_SPEED


# ---------------------------------------------------------------------------
# Dock / undock control
# ---------------------------------------------------------------------------

def dock_pen():
    """Call this when the pen magnet comes near the glove Hall effect sensor."""
    global pen_docked
    pen_docked = True
    print("  [PenSimulator] Pen docked — glove EMG + thumb IMU now active.")  # was missing 2-space indent


def undock_pen():
    """Call this when the pen is removed from the glove."""
    global pen_docked
    pen_docked = False
    print("  [PenSimulator] Pen undocked — pen is now sole input device.")  # was missing 2-space indent


def set_slider(position):
    """Set slider to 1 (cursor mode) or 2 (writing mode)."""
    global slider_position
    assert position in (1, 2), "Slider position must be 1 or 2."
    slider_position = position
    mode_name = "Cursor Mode" if position == 1 else "Writing Mode"
    print(f"  [PenSimulator] Slider set to position {position} — {mode_name}.")  # was missing 2-space indent


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify all generators work
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Running pen_simulator self-test...")
    print(f"  pen_docked      = {pen_docked}")
    print(f"  slider_position = {slider_position}")
    print(f"  pen_down        = {pen_down}")

    # Generate one of each pressure pattern
    patterns = {
        "Idle"         : generate_pressure_idle(),
        "Writing"      : generate_pressure_writing(),
        "Single Tap"   : generate_pressure_light_tap(1),
        "Double Tap"   : generate_pressure_light_tap(2),
        "Triple Tap"   : generate_pressure_light_tap(3),
        "Firm Press"   : generate_pressure_firm_press(),
        "Tap and Hold" : generate_pressure_tap_and_hold(),
    }

    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 10))
    fig.suptitle("Pen Pressure Sensor — All Patterns", fontsize=14, fontweight="bold")

    for ax, (name, signal) in zip(axes, patterns.items()):
        time_axis = np.arange(len(signal)) / SAMPLE_RATE * 1000   # convert to ms
        ax.plot(time_axis, signal, color="steelblue", linewidth=1.2)
        ax.axhline(PEN_DOWN_THRESHOLD,   color="orange", linestyle="--",
                   linewidth=0.9, label=f"Pen-down threshold ({PEN_DOWN_THRESHOLD})")
        ax.axhline(FIRM_PRESS_THRESHOLD, color="red",    linestyle="--",
                   linewidth=0.9, label=f"Firm-press threshold ({FIRM_PRESS_THRESHOLD})")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Force", fontsize=8)
        ax.set_title(name, fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("pen_simulator_test.png", dpi=120)
    plt.show()
    print("\nSelf-test complete. Chart saved as pen_simulator_test.png")

    # Test dock/undock and slider
    print()
    undock_pen()
    set_slider(1)
    set_slider(2)
    dock_pen()

    # Test IMU
    print()
    print("IMU sample (rolling right):", generate_imu_sample("right"))
    print("Cursor velocity from that :", imu_to_cursor_velocity(*generate_imu_sample("right")))