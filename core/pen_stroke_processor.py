"""
core/pen_stroke_processor.py
-----------------------------
Watches the pressure sensor stream and IMU stream in real time.
Detects when the pen touches the surface (stroke start) and lifts off
(stroke end), collecting the IMU-integrated (x, y) path in between.

Each completed stroke is a list of (x, y) coordinates that represents
the shape of one letter component — ready for template matching.

State machine:
    IDLE ──(pen down)──▶ RECORDING ──(pen up)──▶ IDLE + emit stroke

When real hardware arrives, the only change is where pressure and
gyro values come from — serial port reads instead of simulator calls.
Everything else in this file stays identical.
"""

import numpy as np
from .pen_simulator import PEN_DOWN_THRESHOLD, FIRM_PRESS_THRESHOLD

# ---------------------------------------------------------------------------
# Integration constants
# ---------------------------------------------------------------------------

# How many pixels one unit of normalised gyro velocity maps to per sample.
# Tuned so that a normal letter fits in roughly a 100x100 pixel bounding box.
STROKE_SCALE = 8.0

# Minimum number of (x,y) points a stroke must have to be considered valid.
# Filters out noise spikes that briefly cross the pen-down threshold.
MIN_STROKE_POINTS = 5

# Maximum duration of a single stroke in seconds before it is force-closed.
# Prevents the system from accumulating forever if something goes wrong.
MAX_STROKE_DURATION = 3.0

# How long (in seconds) the pressure must stay below PEN_DOWN_THRESHOLD
# before we declare the pen "up". This debounces brief pressure dips
# that happen mid-stroke when writing naturally.
PEN_UP_DEBOUNCE = 0.04   # 40ms — about 8 samples at 200Hz


# ---------------------------------------------------------------------------
# StrokeProcessor class
# ---------------------------------------------------------------------------

class StrokeProcessor:
    """
    Processes one sample at a time (pressure value + gyro x/y).
    Maintains internal state and emits completed strokes.

    Usage:
        processor = StrokeProcessor()

        # Call once per sensor sample:
        stroke = processor.update(pressure, gyro_x, gyro_y)

        # stroke is None while recording or idle.
        # stroke is a dict when a stroke just completed.
    """

    def __init__(self, sample_rate=200):
        self.sample_rate   = sample_rate
        self.sample_period = 1.0 / sample_rate   # seconds per sample

        # Current state
        self.state = "IDLE"   # "IDLE" or "RECORDING"

        # Current stroke being built
        self.stroke_x     = []   # x coordinate at each sample
        self.stroke_y     = []   # y coordinate at each sample
        self.stroke_pressure = [] # pressure at each sample (useful for analysis)

        # Current pen position (resets to 0,0 at start of each stroke)
        self.pen_x = 0.0
        self.pen_y = 0.0

        # Debounce counter: counts consecutive samples below PEN_DOWN_THRESHOLD
        # while in RECORDING state. Only transitions to IDLE when this counter
        # reaches the debounce threshold — avoids false stroke splits mid-letter.
        self.pen_up_counter = 0
        self.pen_up_debounce_samples = int(PEN_UP_DEBOUNCE * sample_rate)

        # Track stroke duration to enforce MAX_STROKE_DURATION
        self.stroke_sample_count = 0

        # Completed strokes — processor appends here, caller reads and clears
        self.completed_strokes = []

        # Running action classifier output (set by classify_action, read externally)
        self.last_action = None


    def update(self, pressure, gyro_x, gyro_y):
        """
        Process one sensor sample.

        pressure : float 0..1 from pressure sensor
        gyro_x   : float rad/s — pen rolling left/right
        gyro_y   : float rad/s — pen rolling up/down

        Returns:
            A completed stroke dict if a stroke just finished, else None.

            Stroke dict keys:
                'points'    : list of (x, y) tuples — the reconstructed path
                'n_points'  : int — number of points in the stroke
                'duration'  : float — stroke duration in seconds
                'pressure'  : list of float — pressure at each point
                'bbox'      : (min_x, min_y, max_x, max_y) — bounding box
                'normalised': list of (x, y) tuples — path normalised to 0..1
        """

        pen_is_down = pressure >= PEN_DOWN_THRESHOLD

        # ── State: IDLE ────────────────────────────────────────────────────
        if self.state == "IDLE":
            if pen_is_down:
                # Pen just touched surface — start a new stroke
                self._start_stroke()
                self._record_point(pressure, gyro_x, gyro_y)
            # else: still idle, nothing to do

            return None   # no completed stroke yet

        # ── State: RECORDING ──────────────────────────────────────────────
        elif self.state == "RECORDING":

            if pen_is_down:
                # Pen still on surface — record this point
                self.pen_up_counter = 0   # reset debounce
                self._record_point(pressure, gyro_x, gyro_y)

                # Force-close if stroke is too long
                self.stroke_sample_count += 1
                if self.stroke_sample_count >= MAX_STROKE_DURATION * self.sample_rate:
                    return self._end_stroke()

            else:
                # Pressure dropped below threshold — start debounce countdown
                self.pen_up_counter += 1

                # Still record position during debounce (pen may be briefly lifting
                # mid-letter — we don't want a gap in the path)
                self._record_point(pressure, gyro_x, gyro_y)

                if self.pen_up_counter >= self.pen_up_debounce_samples:
                    # Pen has been up long enough — end the stroke
                    return self._end_stroke()

            return None


    def _start_stroke(self):
        """Reset all stroke buffers and enter RECORDING state."""
        self.state               = "RECORDING"
        self.stroke_x            = []
        self.stroke_y            = []
        self.stroke_pressure     = []
        self.pen_x               = 0.0
        self.pen_y               = 0.0
        self.pen_up_counter      = 0
        self.stroke_sample_count = 0


    def _record_point(self, pressure, gyro_x, gyro_y):
        """
        Integrate gyro velocity into position and record the point.

        Integration: x_new = x_old + gyro_x * scale * sample_period
        This is the same accumulation pattern as sub-pixel movement
        in mouse_mode.py, but we save every point instead of discarding.
        """
        # Normalise gyro to -1..1 (same as imu_to_cursor_velocity)
        norm_x = np.clip(gyro_x / 2.0, -1.0, 1.0)
        norm_y = np.clip(gyro_y / 2.0, -1.0, 1.0)

        # Integrate: add velocity * time to get displacement
        self.pen_x += norm_x * STROKE_SCALE * self.sample_period
        self.pen_y += norm_y * STROKE_SCALE * self.sample_period

        self.stroke_x.append(self.pen_x)
        self.stroke_y.append(self.pen_y)
        self.stroke_pressure.append(pressure)


    def _end_stroke(self):
        """
        Finalise the current stroke and return it as a dict.
        Returns None if the stroke is too short to be valid (noise spike).
        """
        self.state = "IDLE"

        points = list(zip(self.stroke_x, self.stroke_y))

        # Discard strokes that are too short — likely noise
        if len(points) < MIN_STROKE_POINTS:
            return None

        duration = len(points) / self.sample_rate

        # Bounding box
        xs = self.stroke_x
        ys = self.stroke_y
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Normalise coordinates to 0..1 range so template matching is
        # scale-invariant. A small 'l' and a large 'l' produce the same
        # normalised path and should match the same template.
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        # Guard against a perfectly horizontal/vertical stroke (range = 0)
        x_range = x_range if x_range > 1e-6 else 1.0
        y_range = y_range if y_range > 1e-6 else 1.0

        normalised = [
            ((x - min(xs)) / x_range, (y - min(ys)) / y_range)
            for x, y in points
        ]

        stroke = {
            "points"    : points,
            "n_points"  : len(points),
            "duration"  : duration,
            "pressure"  : list(self.stroke_pressure),
            "bbox"      : bbox,
            "normalised": normalised,
        }

        self.completed_strokes.append(stroke)
        return stroke


    def classify_action(self, pressure_window):
        """
        Looks at a short window of pressure samples and classifies it as
        one of the tap/press actions — used for Slider 1 (cursor mode)
        where taps mean clicks, not letter strokes.

        pressure_window : list or array of recent pressure samples

        Returns one of:
            "idle"          — nothing happening
            "tap"           — single light tap
            "double_tap"    — two taps close together
            "triple_tap"    — three taps
            "firm_press"    — sustained high pressure
            "tap_and_hold"  — tap that hasn't released
        """
        if len(pressure_window) == 0:
            return "idle"

        arr = np.array(pressure_window)

        # Find the maximum pressure in this window
        peak = arr.max()

        # Count how many separate 'above-threshold' segments exist
        # Each segment = one tap event
        above = arr >= PEN_DOWN_THRESHOLD
        # Find transitions: False→True = tap start
        transitions = np.diff(above.astype(int))
        n_tap_starts = int(np.sum(transitions == 1))

        # Check if pressure is still elevated at the end of the window
        currently_down = above[-1]

        # ── Decision logic ────────────────────────────────────────────────
        if peak < PEN_DOWN_THRESHOLD:
            return "idle"

        if peak >= FIRM_PRESS_THRESHOLD:
            return "firm_press"

        if currently_down and n_tap_starts == 1:
            return "tap_and_hold"

        if n_tap_starts >= 3:
            return "triple_tap"

        if n_tap_starts == 2:
            return "double_tap"

        if n_tap_starts == 1:
            return "tap"

        return "idle"


# ---------------------------------------------------------------------------
# Self-test — run this file directly to verify stroke detection works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from core.pen_simulator import (
        generate_pressure_writing,
        generate_pressure_light_tap,
        generate_pressure_idle,
        generate_imu_sample,
        SAMPLE_RATE,
    )

    print("Running pen_stroke_processor self-test...\n")

    processor = StrokeProcessor(sample_rate=SAMPLE_RATE)

    # ── Build a test sequence: idle → stroke → idle → stroke → idle ────────
    # We'll simulate writing two separate strokes (two letter components)

    # Pressure: idle, then writing, then idle, then writing, then idle
    p_idle_1  = generate_pressure_idle(duration=0.2)
    p_stroke1 = generate_pressure_writing(duration=0.6)
    p_idle_2  = generate_pressure_idle(duration=0.3)
    p_stroke2 = generate_pressure_writing(duration=0.4)
    p_idle_3  = generate_pressure_idle(duration=0.2)

    pressure_stream = np.concatenate([
        p_idle_1, p_stroke1, p_idle_2, p_stroke2, p_idle_3
    ])

    # IMU: different motion for each stroke to get different shapes
    # Stroke 1: rolling right then down (like a forward curve)
    # Stroke 2: rolling up then right (like an upstroke)
    total_samples = len(pressure_stream)
    stroke1_start = len(p_idle_1)
    stroke1_end   = stroke1_start + len(p_stroke1)
    stroke2_start = stroke1_end + len(p_idle_2)
    stroke2_end   = stroke2_start + len(p_stroke2)

    gyro_x_stream = np.zeros(total_samples)
    gyro_y_stream = np.zeros(total_samples)

    # Stroke 1: right then curving down
    mid1 = (stroke1_start + stroke1_end) // 2
    gyro_x_stream[stroke1_start:mid1]  =  0.9
    gyro_y_stream[stroke1_start:mid1]  =  0.1
    gyro_x_stream[mid1:stroke1_end]    =  0.3
    gyro_y_stream[mid1:stroke1_end]    =  0.8

    # Stroke 2: upward then right
    mid2 = (stroke2_start + stroke2_end) // 2
    gyro_x_stream[stroke2_start:mid2]  =  0.1
    gyro_y_stream[stroke2_start:mid2]  = -0.9
    gyro_x_stream[mid2:stroke2_end]    =  0.8
    gyro_y_stream[mid2:stroke2_end]    = -0.2

    # Add noise to IMU
    gyro_x_stream += np.random.normal(0, 0.03, total_samples)
    gyro_y_stream += np.random.normal(0, 0.03, total_samples)

    # ── Feed samples one by one into the processor ─────────────────────────
    completed = []
    state_log = []

    for i in range(total_samples):
        result = processor.update(
            pressure_stream[i],
            gyro_x_stream[i],
            gyro_y_stream[i],
        )
        state_log.append(processor.state)
        if result is not None:
            completed.append(result)
            print(f"  Stroke {len(completed)} completed:")
            print(f"    Points    : {result['n_points']}")
            print(f"    Duration  : {result['duration']:.3f}s")
            print(f"    Bbox      : {tuple(round(v,2) for v in result['bbox'])}")
            print()

    print(f"Total strokes detected: {len(completed)}")

    # ── Test action classifier on tap patterns ─────────────────────────────
    print("\nAction classifier test:")
    from core.pen_simulator import generate_pressure_light_tap, generate_pressure_firm_press

    test_cases = {
        "single tap"   : generate_pressure_light_tap(1),
        "double tap"   : generate_pressure_light_tap(2),
        "triple tap"   : generate_pressure_light_tap(3),
        "firm press"   : generate_pressure_firm_press(),
        "idle"         : generate_pressure_idle(),
    }
    for name, signal in test_cases.items():
        action = processor.classify_action(signal)
        match  = "✓" if name.replace(" ", "_") in action or action in name else "~"
        print(f"  {match}  Input: {name:14s}  →  Classified as: {action}")

    # ── Plot ───────────────────────────────────────────────────────────────
    time_axis = np.arange(total_samples) / SAMPLE_RATE * 1000  # ms

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Stroke Processor Self-Test", fontsize=13, fontweight="bold")

    # Panel 1: pressure stream with state overlay
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time_axis, pressure_stream, color="steelblue", linewidth=1.0, label="Pressure")
    ax1.axhline(PEN_DOWN_THRESHOLD,   color="orange", linestyle="--", linewidth=0.9,
                label=f"Pen-down ({PEN_DOWN_THRESHOLD})")
    ax1.axhline(FIRM_PRESS_THRESHOLD, color="red",    linestyle="--", linewidth=0.9,
                label=f"Firm-press ({FIRM_PRESS_THRESHOLD})")
    # Shade RECORDING regions
    recording = np.array([1 if s == "RECORDING" else 0 for s in state_log])
    ax1.fill_between(time_axis, 0, 1, where=recording.astype(bool),
                     alpha=0.15, color="green", label="Recording")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Pressure")
    ax1.set_title("Pressure Stream + Detected Stroke Windows")
    ax1.legend(fontsize=7)

    # Panel 2: raw IMU
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_axis, gyro_x_stream, color="#ef5350", linewidth=0.8, label="Gyro X")
    ax2.plot(time_axis, gyro_y_stream, color="#66bb6a", linewidth=0.8, label="Gyro Y")
    ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("rad/s")
    ax2.set_title("IMU Gyroscope Stream")
    ax2.legend(fontsize=7)

    # Panels 3 & 4: reconstructed stroke paths
    colors = ["#ef5350", "#4fc3f7"]
    for i, stroke in enumerate(completed):
        ax = fig.add_subplot(2, 2, 3 + i)
        xs = [p[0] for p in stroke["points"]]
        ys = [p[1] for p in stroke["points"]]
        ax.plot(xs, ys, color=colors[i], linewidth=1.5)
        ax.scatter(xs[0],  ys[0],  color="white",  s=40, zorder=5, label="Start")
        ax.scatter(xs[-1], ys[-1], color="yellow", s=40, zorder=5, label="End")
        ax.set_title(f"Stroke {i+1} — Reconstructed Path ({stroke['n_points']} pts)")
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
        ax.set_facecolor("#f5f5f5")

    plt.tight_layout()
    plt.savefig("pen_stroke_processor_test.png", dpi=120)
    plt.show()
    print("\nSelf-test complete. Chart saved as pen_stroke_processor_test.png")