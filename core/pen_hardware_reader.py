"""
core/pen_hardware_reader.py
----------------------------
Hardware reader for the EMG Smart Glove pen accessory.

Opens COM6 and reads the serial packet from the pen ESP32:
    GX,GY,GZ,CH1,CH2,CH3,CH4\n

Feeds data to the pen IMU pipeline, pressure pipeline, dock/slider state,
and exposes it to ModeManager as a drop-in replacement for
core/pen_simulator.py globals.

Replaces:
    core/pen_simulator.py  → pen_docked, slider_position, pen_down,
                              get_pressure(), get_pen_docked(),
                              get_slider_position(), get_imu_sample()

Usage:
    reader = PenHardwareReader(port="COM6")
    reader.start()

    # In your main loop:
    imu   = reader.get_imu_sample()
    press = reader.get_pressure()
    dock  = reader.get_pen_docked()
    slide = reader.get_slider_position()

    dx, dy = imu_to_cursor_velocity(imu["gyro_x"], imu["gyro_y"])

    reader.stop()

Packet format (from pen ESP32 sketch):
    GX,GY,GZ,CH1,CH2,CH3,CH4
    Example: 0.0412,0.0178,-0.0034,0,1,0,1823

    GX, GY, GZ  : gyro in rad/s (floats)
    CH1          : GPIO 35, Hall sensor OUT — 0 = pen docked (magnet present),
                   1 = pen undocked
    CH2          : GPIO 33, Slider position 1 — 0 = cursor mode active
    CH3          : GPIO 32, Slider position 2 — 0 = writing mode active
    CH4          : GPIO 34, FSR raw ADC 0–4095 (integer)
"""

import serial
import threading
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

PEN_DOWN_THRESHOLD   = 0.25   # pressure above this → pen is touching surface
FIRM_PRESS_THRESHOLD = 0.75   # pressure above this → firm press (right-click)
SAMPLE_RATE          = 50     # Hz — matches pen ESP32 sketch

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 115200

ADC_MAX = 4095.0

# ── IMU constants — identical to pen_simulator.py ────────────────────────────

MAX_GYRO           = 2.0     # maximum meaningful gyro reading (rad/s)
CURSOR_SENSITIVITY = 18.0    # pixels per frame in cursor mode
DEAD_ZONE          = 0.08    # ignore movements below this threshold (rad/s)

# Tilt-to-scroll constants — identical to pen_simulator.py
TILT_DEAD_ZONE     = 0.12    # tilt below this is ignored (prevents drift scrolling)
TILT_SCROLL_SPEED  = 0.15    # scroll units per sample at full tilt
MAX_TILT           = 1.5     # gyro_z reading that counts as full tilt

# ── Gyro bias correction — default 0.0 until calibrated on real hardware ─────

GYRO_BIAS_X = 0.0
GYRO_BIAS_Y = 0.0
GYRO_BIAS_Z = 0.0

# ── FSR calibration — from your bench test ───────────────────────────────────
# Update these after running reader.calibrate()
DEFAULT_FSR_CALIBRATION = {
    "rest": 0,
    "max": 4095,
}

# ── Module-level state — updated every frame by the reader thread ─────────────
# These mirror the pen_simulator.py globals so imports don't break.

pen_docked       = True    # start docked: glove is in control
slider_position  = 2       # 1 = cursor mode, 2 = writing mode
pen_down         = False   # tip is not currently on surface


class PenHardwareReader:
    """
    Hardware reader for the pen accessory ESP32.

    One serial port, one background thread, one packet per sample.
    Reads gyro, dock state, slider position, and FSR pressure from
    the pen ESP32 and exposes them to the pipeline as a drop-in
    replacement for core/pen_simulator.py.
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 fsr_calibration=None, apply_bias_correction=True):
        self.port                  = port
        self.baud                  = baud
        self.fsr_calibration       = fsr_calibration or dict(DEFAULT_FSR_CALIBRATION)
        self.apply_bias_correction = apply_bias_correction

        self._serial  = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()

        # ── IMU state ─────────────────────────────────────────────────────────
        self._latest_imu = {
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
        }

        # ── Pen sensor state ─────────────────────────────────────────────────
        self._pressure       = 0.0
        self._pen_docked     = True
        self._slider_position = 2
        self._fsr_raw        = 0

        # ── Stats ─────────────────────────────────────────────────────────────
        self.samples_received = 0
        self.parse_errors     = 0
        self.connected        = False


    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """Open serial port, wait for READY handshake, start background reader thread."""
        print(f"  [PenHW] Connecting to {self.port} at {self.baud} baud...")

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0,
            )
            # Some ESP32 dev boards require DTR/RTS to be disabled 
            # to prevent them from hanging in the bootloader state via PySerial
            self._serial.setDTR(False)
            self._serial.setRTS(False)
        except serial.SerialException as e:
            raise RuntimeError(
                f"Cannot open {self.port}. "
                f"Check USB connection and Arduino Serial Monitor is closed.\n  {e}"
            )

        # Wait for READY handshake
        print("  [PenHW] Waiting for pen ESP32 READY...")
        for _ in range(20):
            try:
                line = self._serial.readline().decode("utf-8").strip()
                if line == "READY":
                    print("  [PenHW] Pen ESP32 ready")
                    break
            except Exception:
                pass

        self.connected = True
        self._running  = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="PenHW-Reader"
        )
        self._thread.start()
        print(f"  [PenHW] Reader thread started at {SAMPLE_RATE}Hz")


    def stop(self):
        """Stop reader thread and close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        print(f"  [PenHW] Stopped. {self.samples_received} samples, "
              f"{self.parse_errors} parse errors.")


    def get_imu_sample(self):
        """
        Returns the latest pen IMU reading as a dict.
        Same key format as hardware_reader.py — drop-in replacement.

        Returns:
            {
                "gyro_x": float,   # rad/s
                "gyro_y": float,   # rad/s
                "gyro_z": float,   # rad/s
                "accel_x": float,  # placeholder 0.0
                "accel_y": float,  # placeholder 0.0
                "accel_z": float,  # placeholder 0.0
            }
        """
        with self._lock:
            return dict(self._latest_imu)


    def get_pressure(self):
        """
        Returns the latest FSR pressure normalised to 0.0–1.0.
        Uses FSR calibration data (rest → max range).
        """
        with self._lock:
            return self._pressure


    def get_pen_docked(self):
        """
        Returns True when the pen is docked (Hall sensor CH1 == 0,
        magnet present).
        """
        with self._lock:
            return self._pen_docked


    def get_slider_position(self):
        """
        Returns 1 (cursor mode) or 2 (writing mode) based on CH2/CH3.
        """
        with self._lock:
            return self._slider_position


    def get_fsr_raw(self):
        """Returns the most recent raw FSR ADC value (0–4095)."""
        with self._lock:
            return self._fsr_raw


    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate(self, duration_seconds=5):
        """
        Measures rest baseline and max activation for the FSR pressure sensor.
        Run this once per session before collecting data.

        Usage:
            reader.calibrate()
        """
        import time

        print("\n  ── FSR Calibration ──────────────────────────────")

        # Phase 1: rest
        print(f"\n  LIFT the pen — no pressure on the tip.")
        print(f"  Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording rest baseline...")

        rest_samples = []
        start = time.time()
        while time.time() - start < duration_seconds:
            rest_samples.append(self.get_fsr_raw())
            time.sleep(0.02)

        rest_mean = np.mean(rest_samples) if rest_samples else 0
        print(f"  Rest: {rest_mean:.1f}")

        # Phase 2: max activation
        print(f"\n  PRESS the pen tip as hard as you can.")
        print(f"  Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording max activation...")

        max_samples = []
        start = time.time()
        while time.time() - start < duration_seconds:
            max_samples.append(self.get_fsr_raw())
            time.sleep(0.02)

        max_val = np.percentile(max_samples, 95) if max_samples else 4095
        print(f"  Max: {max_val:.1f}")

        # Update calibration
        self.fsr_calibration["rest"] = rest_mean
        self.fsr_calibration["max"]  = max_val

        print(f"\n  Calibration complete:")
        print(f"    FSR: rest={self.fsr_calibration['rest']:.0f}  "
              f"max={self.fsr_calibration['max']:.0f}")


    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _normalise_fsr(self, raw_value):
        """Normalise raw FSR ADC value to 0.0–1.0 using calibration."""
        rest = self.fsr_calibration["rest"]
        maxv = self.fsr_calibration["max"]
        if maxv <= rest:
            return 0.0
        return float(np.clip((raw_value - rest) / (maxv - rest), 0.0, 1.0))


    def _reader_loop(self):
        """
        Background thread.
        Reads combined packet: GX,GY,GZ,CH1,CH2,CH3,CH4
        Updates IMU dict, pressure, dock state, and slider on every sample.
        Also updates module-level globals (pen_docked, slider_position, pen_down).
        """
        global pen_docked, slider_position, pen_down

        while self._running:
            try:
                raw_bytes = self._serial.readline()
                if not raw_bytes:
                    print("  [PenHW ERROR] Read timeout! ESP32 sent NOTHING in the last 1.0s. Is it stuck or in bootloader?")
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not line or line == "READY":
                    continue

                if self.samples_received + self.parse_errors < 5:
                    print(f"  [PenHW DEBUG] Raw line: '{line}'")

                parts = line.split(",")
                if len(parts) not in [6, 7]:
                    if self.parse_errors < 5:
                        print(f"  [PenHW ERROR] Expected 6 or 7 parts, got {len(parts)}. Line: '{line}'")
                    self.parse_errors += 1
                    continue

                # Parse IMU
                gx = float(parts[0])
                gy = float(parts[1])
                gz = float(parts[2])

                # If values are large, they are likely raw MPU6050 integers instead of rad/s
                if abs(gx) > 20 or abs(gy) > 20 or abs(gz) > 20:
                    gx = (gx / 131.0) * (3.14159265 / 180.0)
                    gy = (gy / 131.0) * (3.14159265 / 180.0)
                    gz = (gz / 131.0) * (3.14159265 / 180.0)

                # Apply bias correction
                if self.apply_bias_correction:
                    gx -= GYRO_BIAS_X
                    gy -= GYRO_BIAS_Y
                    gz -= GYRO_BIAS_Z

                # Parse digital channels
                ch1 = int(parts[3])   # Hall sensor: 0 = docked
                ch2 = int(parts[4])   # Slider pos 1: 0 = cursor mode
                ch3 = int(parts[5])   # Slider pos 2: 0 = writing mode
                
                # FSR might be missing if testing sketch only sends 6 values
                ch4 = int(parts[6]) if len(parts) == 7 else 0

                # Derive pen state
                docked = False        # ← Forcefully undocked for now as requested
                # docked = (ch1 == 0)

                slider = 1            # ← Forcefully set to Cursor Mode (1) for now
                # if ch2 == 0:
                #     slider = 1   # cursor mode
                # elif ch3 == 0:
                #     slider = 2   # writing mode
                # else:
                #     slider = 2   # default to writing mode

                pressure = self._normalise_fsr(ch4)
                is_pen_down = (pressure >= PEN_DOWN_THRESHOLD)

                # Update state under lock
                with self._lock:
                    self._latest_imu = {
                        "gyro_x" : gx,
                        "gyro_y" : gy,
                        "gyro_z" : gz,
                        "accel_x": 0.0,
                        "accel_y": 0.0,
                        "accel_z": 0.0,
                    }
                    self._pressure        = pressure
                    self._pen_docked      = docked
                    self._slider_position = slider
                    self._fsr_raw         = ch4

                # Update module-level globals (read by pipeline via import)
                pen_docked      = docked
                slider_position = slider
                pen_down        = is_pen_down

                self.samples_received += 1

            except ValueError:
                self.parse_errors += 1
            except serial.SerialException:
                print("  [PenHW] Serial connection lost")
                self._running = False
                self.connected = False
                break
            except Exception as e:
                print(f"  [PenHW] Reader error: {e}")


# ── Cursor velocity pipeline — identical to pen_simulator.py ─────────────────

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


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 62)
    print("  Pen Hardware Reader Self-Test")
    print("  Tests IMU, pressure, dock state, and slider from pen ESP32")
    print("=" * 62)

    hw = PenHardwareReader(port=DEFAULT_PORT)

    try:
        hw.start()
        time.sleep(0.5)

        print("\n  Reading 60 samples (move pen + press tip)...\n")
        print(f"  {'#':<5} {'GX':>8} {'GY':>8} {'GZ':>8} "
              f"{'Press':>7} {'Dock':>6} {'Slide':>6} {'FSR':>6}")
        print("  " + "─" * 62)

        for i in range(60):
            time.sleep(0.1)

            imu   = hw.get_imu_sample()
            press = hw.get_pressure()
            dock  = hw.get_pen_docked()
            slide = hw.get_slider_position()
            fsr   = hw.get_fsr_raw()

            print(f"  {i:<5} "
                  f"{imu['gyro_x']:>8.3f} "
                  f"{imu['gyro_y']:>8.3f} "
                  f"{imu['gyro_z']:>8.3f} "
                  f"{press:>7.3f} "
                  f"{'YES' if dock else 'NO':>6} "
                  f"{slide:>6} "
                  f"{fsr:>6}")

        print(f"\n  Samples received : {hw.samples_received}")
        print(f"  Parse errors     : {hw.parse_errors}")
        print(f"\n  Module globals:")
        print(f"    pen_docked      = {pen_docked}")
        print(f"    slider_position = {slider_position}")
        print(f"    pen_down        = {pen_down}")

    finally:
        hw.stop()
        print("\n  Self-test complete.")
