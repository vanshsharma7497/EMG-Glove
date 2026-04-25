"""
core/imu_hardware.py
---------------------
Real hardware replacement for core/imu_simulator.py.

Reads live gyroscope data from the MPU-6050 via ESP32 serial port.
The ESP32 runs the packet sketch which outputs CSV lines at 100Hz:
    GX,GY,GZ\n   (values in rad/s)

Drop-in replacement:
    # In any file that currently does:
    from core.imu_simulator import generate_imu_sample, imu_to_cursor_velocity

    # Replace with:
    from core.imu_hardware import get_imu_sample, imu_to_cursor_velocity

The imu_to_cursor_velocity() function is identical to imu_simulator.py —
same dead zone, normalisation, and acceleration curve. No changes needed
anywhere else in the pipeline.

Usage:
    imu = IMUHardware(port="COM5", baud=115200)
    imu.start()

    # In your loop:
    sample = imu.get_sample()
    dx, dy = imu_to_cursor_velocity(sample)

    imu.stop()
"""

import serial
import threading
import numpy as np
from collections import deque

# ── Constants — must match imu_simulator.py exactly ──────────────────────────
# These values are used by imu_to_cursor_velocity() which is shared between
# simulator and hardware. Keeping them identical means the pipeline behaviour
# is identical whether running on simulated or real data.

DEAD_ZONE          = 0.08   # rad/s — movements below this are ignored
MAX_GYRO           = 1.5    # rad/s — maximum gyro value we care about
CURSOR_SENSITIVITY = 20.0   # pixels per frame at full deflection
ACCELERATION_CURVE = 2.0    # exponent — >1 gives trackpad-like feel

# ── Default serial settings ───────────────────────────────────────────────────
DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 115200

# ── Gyro bias correction ──────────────────────────────────────────────────────
# From your validated rest readings:
#   GX rests at ~0.047 rad/s, GY at ~0.019 rad/s, GZ at ~0.002 rad/s
# Subtracting these offsets centres the sensor at true zero.
# If you recalibrate, update these three values.
GYRO_BIAS_X = 0.047
GYRO_BIAS_Y = 0.019
GYRO_BIAS_Z = 0.002


class IMUHardware:
    """
    Reads live gyro data from ESP32 serial port in a background thread.
    The main loop calls get_sample() which always returns the latest reading.

    Thread safety: the background thread writes to _latest_sample under a lock.
    get_sample() reads under the same lock. No data races.
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 apply_bias_correction=True):
        """
        port                  : COM port the ESP32 is on (e.g. "COM5")
        baud                  : must match Serial.begin() in Arduino sketch
        apply_bias_correction : subtract rest-state bias from all readings
        """
        self.port   = port
        self.baud   = baud
        self.apply_bias_correction = apply_bias_correction

        self._serial  = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()

        # Latest sample — updated by background thread, read by main loop
        self._latest_sample = {
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
        }

        # Rolling buffer of recent samples for diagnostics
        self._history = deque(maxlen=100)

        # Stats
        self.samples_received = 0
        self.parse_errors     = 0
        self.connected        = False


    def start(self):
        """
        Open serial port and start background reader thread.
        Blocks until the ESP32 sends "READY" or times out after 5 seconds.
        """
        print(f"  [IMU] Connecting to {self.port} at {self.baud} baud...")

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=2.0,   # 2s read timeout per line
            )
        except serial.SerialException as e:
            raise RuntimeError(
                f"Cannot open {self.port}. "
                f"Check USB connection and port number.\n  {e}"
            )

        # Wait for READY handshake from ESP32
        print("  [IMU] Waiting for ESP32 READY...")
        ready = False
        for _ in range(20):   # wait up to ~4 seconds (20 × 200ms timeout)
            try:
                line = self._serial.readline().decode("utf-8").strip()
                if line == "READY":
                    ready = True
                    break
                # Ignore boot garbage lines
            except Exception:
                pass

        if not ready:
            # Not fatal — ESP32 may have already sent READY before we opened port.
            # Continue anyway; the reader thread will start picking up data.
            print("  [IMU] READY not received — continuing anyway")

        self.connected = True
        self._running  = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,   # dies automatically when main program exits
            name="IMU-Reader"
        )
        self._thread.start()
        print(f"  [IMU] Reader thread started — receiving at 100Hz")


    def stop(self):
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        print(f"  [IMU] Stopped. {self.samples_received} samples received, "
              f"{self.parse_errors} parse errors.")


    def get_sample(self):
        """
        Returns the latest gyro reading as a dict:
            {
                "gyro_x": float,   # rad/s — left/right roll
                "gyro_y": float,   # rad/s — up/down roll
                "gyro_z": float,   # rad/s — barrel twist (not used for cursor)
                "accel_x": 0.0,    # placeholder — matches imu_simulator.py dict
                "accel_y": 0.0,
                "accel_z": 1.0,
            }

        The accel fields are placeholders so this dict is a drop-in
        replacement for the dict returned by imu_simulator.generate_imu_sample().
        """
        with self._lock:
            sample = dict(self._latest_sample)

        # Add placeholder accel fields to match imu_simulator.py dict format
        sample["accel_x"] = 0.0
        sample["accel_y"] = 0.0
        sample["accel_z"] = 1.0

        return sample


    def get_recent_samples(self, n=10):
        """Returns the last n samples as a list — useful for diagnostics."""
        with self._lock:
            return list(self._history)[-n:]


    # ── Background reader thread ──────────────────────────────────────────────

    def _reader_loop(self):
        """
        Runs in a background thread. Reads lines from serial port,
        parses CSV packets, applies bias correction, and stores latest sample.

        Packet format from ESP32: "GX,GY,GZ\n"
        Example:  "0.0426,0.0172,-0.0011\n"
        """
        while self._running:
            try:
                raw = self._serial.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()

                # Skip non-data lines (READY, ERROR, boot messages)
                if not line or not line[0].lstrip("-").replace(".", "").isdigit():
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    self.parse_errors += 1
                    continue

                gx = float(parts[0])
                gy = float(parts[1])
                gz = float(parts[2])

                # Apply bias correction — centres sensor at true zero
                if self.apply_bias_correction:
                    gx -= GYRO_BIAS_X
                    gy -= GYRO_BIAS_Y
                    gz -= GYRO_BIAS_Z

                sample = {
                    "gyro_x": gx,
                    "gyro_y": gy,
                    "gyro_z": gz,
                }

                with self._lock:
                    self._latest_sample = sample
                    self._history.append(sample)

                self.samples_received += 1

            except ValueError:
                self.parse_errors += 1
            except serial.SerialException:
                # Port closed or disconnected
                print("  [IMU] Serial connection lost")
                self._running = False
                self.connected = False
                break
            except Exception as e:
                print(f"  [IMU] Reader error: {e}")


# ── Cursor velocity pipeline ──────────────────────────────────────────────────
# Identical to imu_simulator.py — same dead zone, normalisation, acceleration.
# Kept here so hardware and simulator are fully interchangeable.

def apply_dead_zone(value, threshold=DEAD_ZONE):
    """Ignore movements below threshold — prevents cursor drift when still."""
    if abs(value) < threshold:
        return 0.0
    return np.sign(value) * (abs(value) - threshold)


def apply_acceleration_curve(value, exponent=ACCELERATION_CURVE):
    """Non-linear curve — slow = precise, fast = accelerates. Like a trackpad."""
    return np.sign(value) * (abs(value) ** exponent)


def imu_to_cursor_velocity(imu_sample):
    """
    Full pipeline: raw IMU sample dict → cursor (dx, dy) in pixels/frame.
    Identical behaviour to imu_simulator.imu_to_cursor_velocity().

    Returns:
        dx : horizontal velocity (positive = right)
        dy : vertical velocity (positive = down)
    """
    raw_x = apply_dead_zone(imu_sample["gyro_x"])
    raw_y = apply_dead_zone(imu_sample["gyro_y"])

    norm_x = np.clip(raw_x / MAX_GYRO, -1.0, 1.0)
    norm_y = np.clip(raw_y / MAX_GYRO, -1.0, 1.0)

    curved_x = apply_acceleration_curve(norm_x)
    curved_y = apply_acceleration_curve(norm_y)

    dx =  curved_x * CURSOR_SENSITIVITY
    dy = -curved_y * CURSOR_SENSITIVITY   # invert Y: roll up = cursor up

    return dx, dy


# ── Convenience: module-level instance ───────────────────────────────────────
# Allows simple usage:
#   from core.imu_hardware import imu, imu_to_cursor_velocity
#   imu.start()
#   sample = imu.get_sample()

imu = IMUHardware()


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("=" * 50)
    print("  IMU Hardware Self-Test")
    print("=" * 50)

    hw = IMUHardware(port=DEFAULT_PORT, apply_bias_correction=True)

    try:
        hw.start()
        print("\n  Reading 50 samples (5 seconds)...\n")
        print(f"  {'Sample':<8} {'GX (rad/s)':>12} {'GY (rad/s)':>12} "
              f"{'GZ (rad/s)':>12} {'dx (px)':>10} {'dy (px)':>10}")
        print("  " + "─" * 68)

        for i in range(50):
            time.sleep(0.1)
            sample = hw.get_sample()
            dx, dy = imu_to_cursor_velocity(sample)
            print(f"  {i:<8} "
                  f"{sample['gyro_x']:>12.4f} "
                  f"{sample['gyro_y']:>12.4f} "
                  f"{sample['gyro_z']:>12.4f} "
                  f"{dx:>10.2f} "
                  f"{dy:>10.2f}")

        print(f"\n  Total samples received : {hw.samples_received}")
        print(f"  Parse errors           : {hw.parse_errors}")
        print(f"  Expected ~500 samples  : "
              f"{'✓ OK' if hw.samples_received > 400 else '✗ LOW - check baud rate'}")

    finally:
        hw.stop()
        print("\n  Self-test complete.")