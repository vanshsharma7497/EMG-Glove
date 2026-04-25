"""
core/hardware_reader.py
------------------------
Unified hardware reader for the EMG Smart Glove.

Opens COM5 once and reads the combined serial packet from the ESP32:
    GX,GY,GZ,CH1,CH2,CH3,CH4\n

Feeds data to both the IMU pipeline and the EMG pipeline simultaneously.
This is the single entry point for all real hardware data.

Replaces:
    core/imu_simulator.py   → get_imu_sample()
    core/emg_simulator.py   → generate_emg_window()

Usage:
    reader = HardwareReader(port="COM5")
    reader.start()

    # In your main loop:
    imu_sample = reader.get_imu_sample()
    emg_window = reader.get_emg_window()

    dx, dy = imu_to_cursor_velocity(imu_sample)
    filtered = filter_all_channels(emg_window)
    features = extract_features(filtered)

    reader.stop()

Packet format (from combined Arduino sketch):
    GX,GY,GZ,CH1,CH2,CH3,CH4
    Example: 0.0412,0.0178,-0.0034,98,102,1456,105

    GX, GY, GZ  : gyro in rad/s (floats)
    CH1-CH4     : raw ADC values 0-4095 (ints)
"""

import serial
import threading
import numpy as np
from collections import deque

# ── Constants ─────────────────────────────────────────────────────────────────

N_CHANNELS   = 4
ADC_MAX      = 4095.0
WINDOW_SIZE  = 200      # samples per EMG window — matches emg_simulator.py
SAMPLE_RATE  = 50       # Hz — matches Arduino sketch

DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 115200

# ── IMU constants — identical to imu_simulator.py ────────────────────────────
DEAD_ZONE          = 0.08
MAX_GYRO           = 1.5
CURSOR_SENSITIVITY = 20.0
ACCELERATION_CURVE = 2.0

# ── IMU bias correction — from your validated rest readings ───────────────────
GYRO_BIAS_X = 0.047
GYRO_BIAS_Y = 0.019
GYRO_BIAS_Z = 0.002

# ── EMG calibration — from your bench test ───────────────────────────────────
# Update these after running reader.calibrate()
DEFAULT_CALIBRATION = {
    "ch1": {"rest": 100, "max": 2500},
    "ch2": {"rest": 100, "max": 2500},
    "ch3": {"rest": 100, "max": 2500},
    "ch4": {"rest": 100, "max": 2500},
}


class HardwareReader:
    """
    Single unified reader for all ESP32 sensor data.

    One serial port, one background thread, one packet per sample.
    Both IMU and EMG data are always fresh and always in sync.

    This solves the problem of two separate readers (imu_hardware.py and
    emg_hardware.py) both trying to open the same COM port.
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 calibration=None, apply_bias_correction=True):
        self.port                 = port
        self.baud                 = baud
        self.calibration          = calibration or {
            k: dict(v) for k, v in DEFAULT_CALIBRATION.items()
        }
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
            "accel_z": 1.0,
        }

        # ── EMG state ─────────────────────────────────────────────────────────
        # Rolling window buffer — shape (4, WINDOW_SIZE)
        self._emg_buffers = [
            deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
            for _ in range(N_CHANNELS)
        ]
        self._latest_emg_raw = [0] * N_CHANNELS

        # ── Stats ─────────────────────────────────────────────────────────────
        self.samples_received = 0
        self.parse_errors     = 0
        self.connected        = False


    # =========================================================================
    # Public API
    # =========================================================================

    def start(self):
        """Open serial port and start background reader thread."""
        print(f"  [HW] Connecting to {self.port} at {self.baud} baud...")

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0,
            )
        except serial.SerialException as e:
            raise RuntimeError(
                f"Cannot open {self.port}. "
                f"Check USB connection and Arduino Serial Monitor is closed.\n  {e}"
            )

        # Wait for READY handshake
        print("  [HW] Waiting for ESP32 READY...")
        for _ in range(20):
            try:
                line = self._serial.readline().decode("utf-8").strip()
                if line == "READY":
                    print("  [HW] ESP32 ready")
                    break
            except Exception:
                pass

        self.connected = True
        self._running  = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="HW-Reader"
        )
        self._thread.start()
        print(f"  [HW] Reader thread started at {SAMPLE_RATE}Hz")


    def stop(self):
        """Stop reader thread and close serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        print(f"  [HW] Stopped. {self.samples_received} samples, "
              f"{self.parse_errors} parse errors.")


    def get_imu_sample(self):
        """
        Returns the latest IMU reading as a dict.
        Same format as imu_simulator.generate_imu_sample() — drop-in replacement.

        Returns:
            {
                "gyro_x": float,   # rad/s
                "gyro_y": float,   # rad/s
                "gyro_z": float,   # rad/s
                "accel_x": float,
                "accel_y": float,
                "accel_z": float,
            }
        """
        with self._lock:
            return dict(self._latest_imu)


    def get_emg_window(self):
        """
        Returns the most recent WINDOW_SIZE EMG samples as numpy array.
        Shape: (4, 200) — same as emg_simulator.generate_emg_window().
        Values normalised to 0.0-1.0 using calibration data.
        """
        with self._lock:
            return np.array(
                [list(buf) for buf in self._emg_buffers],
                dtype=np.float32
            )


    def get_latest_emg_raw(self):
        """Returns the most recent raw ADC value for each channel."""
        with self._lock:
            return list(self._latest_emg_raw)


    def get_latest_emg_normalised(self):
        """Returns the most recent normalised EMG value per channel."""
        raw = self.get_latest_emg_raw()
        return [self._normalise_emg(raw[ch], ch) for ch in range(N_CHANNELS)]


    # =========================================================================
    # Calibration
    # =========================================================================

    def calibrate(self, duration_seconds=5):
        """
        Measures rest baseline and max activation for each connected channel.
        Run this once per session before collecting gesture data.

        Usage:
            reader.calibrate()
        """
        import time

        print("\n  ── EMG Calibration ──────────────────────────────")

        # Phase 1: rest
        print(f"\n  RELAX your hand completely.")
        print(f"  Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording rest baseline...")

        rest_samples = [[] for _ in range(N_CHANNELS)]
        start = time.time()
        while time.time() - start < duration_seconds:
            raw = self.get_latest_emg_raw()
            for ch in range(N_CHANNELS):
                rest_samples[ch].append(raw[ch])
            time.sleep(0.02)

        rest_means = [np.mean(s) if s else 100 for s in rest_samples]
        print(f"  Rest: {[round(r, 1) for r in rest_means]}")

        # Phase 2: max activation
        print(f"\n  SQUEEZE as hard as you can.")
        print(f"  Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording max activation...")

        max_samples = [[] for _ in range(N_CHANNELS)]
        start = time.time()
        while time.time() - start < duration_seconds:
            raw = self.get_latest_emg_raw()
            for ch in range(N_CHANNELS):
                max_samples[ch].append(raw[ch])
            time.sleep(0.02)

        max_vals = [np.percentile(s, 95) if s else 2500
                    for s in max_samples]
        print(f"  Max: {[round(m, 1) for m in max_vals]}")

        # Update calibration
        for ch in range(N_CHANNELS):
            key = f"ch{ch + 1}"
            self.calibration[key]["rest"] = rest_means[ch]
            self.calibration[key]["max"]  = max_vals[ch]

        print(f"\n  Calibration complete:")
        for ch in range(N_CHANNELS):
            key = f"ch{ch + 1}"
            print(f"    Ch{ch+1}: rest={self.calibration[key]['rest']:.0f}  "
                  f"max={self.calibration[key]['max']:.0f}")


    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _normalise_emg(self, raw_value, channel_idx):
        """Normalise raw ADC value to 0.0-1.0 using calibration."""
        key  = f"ch{channel_idx + 1}"
        rest = self.calibration[key]["rest"]
        maxv = self.calibration[key]["max"]
        if maxv <= rest:
            return 0.0
        return float(np.clip((raw_value - rest) / (maxv - rest), 0.0, 1.0))


    def _reader_loop(self):
        """
        Background thread.
        Reads combined packet: GX,GY,GZ,CH1,CH2,CH3,CH4
        Updates IMU dict and EMG buffers on every sample.
        """
        while self._running:
            try:
                raw_bytes = self._serial.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not line or line == "READY":
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    self.parse_errors += 1
                    continue

                # Parse IMU
                gx = float(parts[0])
                gy = float(parts[1])
                gz = float(parts[2])

                # Apply bias correction
                if self.apply_bias_correction:
                    gx -= GYRO_BIAS_X
                    gy -= GYRO_BIAS_Y
                    gz -= GYRO_BIAS_Z

                # Parse EMG
                ch_raw = [int(parts[3]), int(parts[4]),
                          int(parts[5]), int(parts[6])]

                # Normalise EMG
                ch_norm = [self._normalise_emg(ch_raw[ch], ch)
                           for ch in range(N_CHANNELS)]

                # Update state under lock
                with self._lock:
                    self._latest_imu = {
                        "gyro_x" : gx,
                        "gyro_y" : gy,
                        "gyro_z" : gz,
                        "accel_x": 0.0,
                        "accel_y": 0.0,
                        "accel_z": 1.0,
                    }
                    for ch in range(N_CHANNELS):
                        self._emg_buffers[ch].append(ch_norm[ch])
                        self._latest_emg_raw[ch] = ch_raw[ch]

                self.samples_received += 1

            except ValueError:
                self.parse_errors += 1
            except serial.SerialException:
                print("  [HW] Serial connection lost")
                self._running = False
                self.connected = False
                break
            except Exception as e:
                print(f"  [HW] Reader error: {e}")


# ── Cursor velocity pipeline — identical to imu_simulator.py ─────────────────

def apply_dead_zone(value, threshold=DEAD_ZONE):
    if abs(value) < threshold:
        return 0.0
    return np.sign(value) * (abs(value) - threshold)


def apply_acceleration_curve(value, exponent=ACCELERATION_CURVE):
    return np.sign(value) * (abs(value) ** exponent)


def imu_to_cursor_velocity(imu_sample):
    """
    Converts IMU sample dict to cursor (dx, dy) velocity in pixels/frame.
    Identical to imu_simulator.imu_to_cursor_velocity().
    """
    raw_x = apply_dead_zone(imu_sample["gyro_x"])
    raw_y = apply_dead_zone(imu_sample["gyro_y"])

    norm_x = np.clip(raw_x / MAX_GYRO, -1.0, 1.0)
    norm_y = np.clip(raw_y / MAX_GYRO, -1.0, 1.0)

    curved_x = apply_acceleration_curve(norm_x)
    curved_y = apply_acceleration_curve(norm_y)

    dx =  curved_x * CURSOR_SENSITIVITY
    dy = -curved_y * CURSOR_SENSITIVITY

    return dx, dy


# ── Self-test ─────────────────────────────────────────────────────────────────
# Removed: module-level `reader = HardwareReader()` — created an unused instance
# on every import (including during ModeManager init), wasting memory.
if __name__ == "__main__":
    import time

    print("=" * 58)
    print("  Hardware Reader Self-Test")
    print("  Tests both IMU and EMG from single serial stream")
    print("=" * 58)

    hw = HardwareReader(port=DEFAULT_PORT)

    try:
        hw.start()
        time.sleep(0.5)

        print("\n  Reading 60 samples (move board + flex hand)...\n")
        print(f"  {'#':<5} {'GX':>8} {'GY':>8} {'GZ':>8} "
              f"{'CH3 Raw':>9} {'CH3 Norm':>9} {'Signal':<15}")
        print("  " + "─" * 68)

        for i in range(60):
            time.sleep(0.1)

            imu  = hw.get_imu_sample()
            raw  = hw.get_latest_emg_raw()
            norm = hw.get_latest_emg_normalised()
            dx, dy = imu_to_cursor_velocity(imu)

            bar_len = int(norm[2] * 15)
            bar     = "█" * bar_len + "░" * (15 - bar_len)

            print(f"  {i:<5} "
                  f"{imu['gyro_x']:>8.3f} "
                  f"{imu['gyro_y']:>8.3f} "
                  f"{imu['gyro_z']:>8.3f} "
                  f"{raw[2]:>9} "
                  f"{norm[2]:>9.3f} "
                  f"{bar}")

        print(f"\n  Samples received : {hw.samples_received}")
        print(f"  Parse errors     : {hw.parse_errors}")
        print(f"\n  EMG window shape : {hw.get_emg_window().shape}")
        print(f"  IMU sample keys  : {list(hw.get_imu_sample().keys())}")

    finally:
        hw.stop()
        print("\n  Self-test complete.")