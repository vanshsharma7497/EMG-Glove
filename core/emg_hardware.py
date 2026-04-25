"""
core/emg_hardware.py
---------------------
Real hardware replacement for core/emg_simulator.py.

Reads live EMG data from 1-4 MyoWare 2.0 sensors via ESP32 ADC pins.
The ESP32 reads analog values from each sensor and sends them over serial.

Channel mapping (matches emg_simulator.py and documentation):
    Ch1 — GPIO 32 — Thenar eminence (thumb flexor)
    Ch2 — GPIO 33 — First dorsal interosseous (index finger)
    Ch3 — GPIO 34 — Forearm flexor (middle/ring finger flexors)
    Ch4 — GPIO 35 — Forearm extensor (ring/pinky extensors)

For the Phase 2 bench test, only Ch3 is connected. The other channels
return 0.0 until sensors are added. The pipeline handles this correctly
because a channel outputting 0.0 looks like a resting muscle.

Drop-in replacement:
    # Currently in emg_dataset.py and mode_manager.py:
    from core.emg_simulator import generate_emg_window

    # For hardware, the pipeline reads one sample at a time, not windows.
    # See EMGHardware.get_window() which assembles a full window buffer.

Arduino sketch required:
    The ESP32 must be running the combined packet sketch (emg_imu_packet.ino)
    which outputs one line per sample at 1000Hz:
        EMG,ch1,ch2,ch3,ch4,gx,gy,gz\\n
    
    Until the combined sketch is written, use the single-channel test sketch
    and only Ch3 will have real data.

Normalisation:
    Raw ADC values (0-4095) are normalised to 0.0-1.0.
    Calibration values (rest baseline and max activation) are measured
    once per session and stored in self.calibration.
"""

import serial
import threading
import numpy as np
from collections import deque

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of EMG channels
N_CHANNELS = 4

# ESP32 ADC resolution — 12-bit = 0 to 4095
ADC_MAX = 4095.0

# Sample rate — MyoWare reads at 1kHz to match emg_simulator.py
EMG_SAMPLE_RATE = 1000  # Hz

# Window size — 200 samples = 200ms at 1kHz (matches emg_simulator.py)
WINDOW_SIZE = 200

# Default serial settings
DEFAULT_PORT = "COM5"
DEFAULT_BAUD = 115200

# ── Calibration values from your bench test ───────────────────────────────────
# These are measured from your actual sensor readings:
#   Rest baseline:  ~100 RAW  (~0.08V)
#   Max activation: ~2500 RAW (estimated from tight fist trend)
#
# Update these after running calibrate() for best accuracy.

DEFAULT_CALIBRATION = {
    "ch1": {"rest": 100, "max": 2500},
    "ch2": {"rest": 100, "max": 2500},
    "ch3": {"rest": 100, "max": 2500},  # measured from your bench test
    "ch4": {"rest": 100, "max": 2500},
}


class EMGHardware:
    """
    Reads live EMG data from MyoWare sensors via ESP32 serial port.

    Maintains a rolling buffer of WINDOW_SIZE samples per channel.
    get_window() returns a (4, 200) numpy array matching the format
    that emg_filter.py and emg_features.py expect.

    Usage:
        emg = EMGHardware(port="COM5")
        emg.start()

        # In your loop:
        window = emg.get_window()      # shape (4, 200)
        filtered = filter_all_channels(window)
        features = extract_features(filtered)
        gesture = clf.predict([features_scaled])[0]

        emg.stop()
    """

    def __init__(self, port=DEFAULT_PORT, baud=DEFAULT_BAUD,
                 calibration=None):
        """
        port        : COM port the ESP32 is on
        baud        : must match Serial.begin() in Arduino sketch
        calibration : dict of per-channel rest/max values.
                      If None, uses DEFAULT_CALIBRATION.
        """
        self.port        = port
        self.baud        = baud
        self.calibration = calibration or DEFAULT_CALIBRATION

        self._serial  = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()

        # Rolling buffer — one deque per channel, each holds WINDOW_SIZE samples
        # Initialised with zeros (= resting muscle signal)
        self._buffers = [
            deque([0.0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
            for _ in range(N_CHANNELS)
        ]

        # Latest raw ADC values per channel — for diagnostics
        self._latest_raw = [0] * N_CHANNELS

        # Stats
        self.samples_received = 0
        self.parse_errors     = 0
        self.connected        = False

        # Calibration state
        self._calibrating    = False
        self._cal_buffer     = [[] for _ in range(N_CHANNELS)]


    def start(self):
        """
        Open serial port and start background reader thread.
        """
        print(f"  [EMG] Connecting to {self.port} at {self.baud} baud...")

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1.0,
            )
        except serial.SerialException as e:
            raise RuntimeError(
                f"Cannot open {self.port}. "
                f"Check USB connection and port number.\n  {e}"
            )

        self.connected = True
        self._running  = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="EMG-Reader"
        )
        self._thread.start()
        print(f"  [EMG] Reader thread started")


    def stop(self):
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
        print(f"  [EMG] Stopped. {self.samples_received} samples, "
              f"{self.parse_errors} parse errors.")


    def get_window(self):
        """
        Returns the most recent WINDOW_SIZE samples for all 4 channels
        as a numpy array of shape (4, 200).

        Values are normalised to 0.0-1.0 range using calibration data.
        This is the same shape as generate_emg_window() in emg_simulator.py.

        The returned window is a snapshot — safe to use while the reader
        thread continues updating the buffer.
        """
        with self._lock:
            window = np.array([list(buf) for buf in self._buffers],
                              dtype=np.float32)
        return window


    def get_latest_raw(self):
        """Returns the most recent raw ADC value for each channel."""
        with self._lock:
            return list(self._latest_raw)


    def get_latest_normalised(self):
        """Returns the most recent normalised value (0.0-1.0) per channel."""
        raw = self.get_latest_raw()
        return [self._normalise(raw[ch], ch) for ch in range(N_CHANNELS)]


    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, duration_seconds=5):
        """
        Interactive calibration session.
        Records rest baseline and maximum activation for each channel.

        Usage:
            emg.calibrate()
            # Follow the prompts — relax then contract

        Updates self.calibration with measured values.
        """
        print("\n  ── EMG Calibration ──────────────────────────────────")
        print(f"  Will record {duration_seconds}s rest then {duration_seconds}s max")

        import time

        # Phase 1: rest baseline
        print("\n  RELAX your hand completely. Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording rest baseline...")

        rest_samples = [[] for _ in range(N_CHANNELS)]
        start = time.time()
        while time.time() - start < duration_seconds:
            raw = self.get_latest_raw()
            for ch in range(N_CHANNELS):
                rest_samples[ch].append(raw[ch])
            time.sleep(0.01)

        rest_means = [np.mean(s) if s else 100 for s in rest_samples]
        print(f"  Rest baseline: {[round(r, 1) for r in rest_means]}")

        # Phase 2: maximum activation
        print(f"\n  SQUEEZE as hard as you can. Recording in 3 seconds...")
        time.sleep(3)
        print("  Recording maximum activation...")

        max_samples = [[] for _ in range(N_CHANNELS)]
        start = time.time()
        while time.time() - start < duration_seconds:
            raw = self.get_latest_raw()
            for ch in range(N_CHANNELS):
                max_samples[ch].append(raw[ch])
            time.sleep(0.01)

        max_means = [np.percentile(s, 95) if s else 2500
                     for s in max_samples]
        print(f"  Max activation: {[round(m, 1) for m in max_means]}")

        # Update calibration
        for ch in range(N_CHANNELS):
            key = f"ch{ch + 1}"
            self.calibration[key]["rest"] = rest_means[ch]
            self.calibration[key]["max"]  = max_means[ch]

        print(f"\n  Calibration complete. Updated values:")
        for ch in range(N_CHANNELS):
            key = f"ch{ch + 1}"
            print(f"    Ch{ch+1}: rest={self.calibration[key]['rest']:.0f}  "
                  f"max={self.calibration[key]['max']:.0f}")
        print()


    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalise(self, raw_value, channel_idx):
        """
        Normalises a raw ADC value to 0.0-1.0 using calibration data.

        Values below rest baseline map to 0.0 (resting muscle).
        Values at or above max activation map to 1.0.
        Linear interpolation between rest and max.
        """
        key  = f"ch{channel_idx + 1}"
        rest = self.calibration[key]["rest"]
        maxv = self.calibration[key]["max"]

        if maxv <= rest:
            return 0.0

        normalised = (raw_value - rest) / (maxv - rest)
        return float(np.clip(normalised, 0.0, 1.0))


    def _reader_loop(self):
        """
        Background thread. Reads serial lines and updates channel buffers.

        Currently handles two packet formats:

        Format 1 — Single channel test sketch (what you have now):
            "RAW: 1234  VOLTAGE: 0.9934\\n"
            Only updates Ch3 (index 2).

        Format 2 — Combined packet sketch (future):
            "EMG,ch1,ch2,ch3,ch4\\n"
            Updates all 4 channels simultaneously.

        The reader auto-detects which format is incoming.
        """
        while self._running:
            try:
                raw_bytes = self._serial.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                # ── Format 2: combined EMG packet ─────────────────────────
                if line.startswith("EMG,"):
                    parts = line.split(",")
                    if len(parts) == 5:
                        values = [int(p) for p in parts[1:]]
                        normalised = [self._normalise(values[ch], ch)
                                      for ch in range(N_CHANNELS)]
                        with self._lock:
                            for ch in range(N_CHANNELS):
                                self._buffers[ch].append(normalised[ch])
                                self._latest_raw[ch] = values[ch]
                        self.samples_received += 1

                # ── Format 1: single channel test sketch ───────────────────
                elif line.startswith("RAW:"):
                    # Parse "RAW: 1234  VOLTAGE: 0.9934"
                    parts = line.split()
                    if len(parts) >= 2:
                        raw_val = int(parts[1])
                        normalised = self._normalise(raw_val, 2)  # Ch3
                        with self._lock:
                            self._buffers[2].append(normalised)
                            self._latest_raw[2] = raw_val
                        self.samples_received += 1

            except ValueError:
                self.parse_errors += 1
            except serial.SerialException:
                print("  [EMG] Serial connection lost")
                self._running = False
                self.connected = False
                break
            except Exception as e:
                print(f"  [EMG] Reader error: {e}")


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  EMG Hardware Self-Test")
    print("=" * 55)
    print("  Make sure the single-channel EMG sketch is running")
    print("  on the ESP32 (the one that prints RAW: values)")
    print()

    emg = EMGHardware(port=DEFAULT_PORT)

    try:
        emg.start()
        time.sleep(0.5)  # let buffer fill

        print("  Reading live EMG signal...")
        print("  Relax your hand, then make a fist, then relax again\n")
        print(f"  {'Sample':<8} {'Ch3 Raw':>10} {'Ch3 Norm':>10} "
              f"{'Signal Level':<20}")
        print("  " + "─" * 55)

        for i in range(100):
            time.sleep(0.1)
            raw    = emg.get_latest_raw()
            norm   = emg.get_latest_normalised()

            # Simple ASCII bar to visualise signal level
            bar_len = int(norm[2] * 20)
            bar     = "█" * bar_len + "░" * (20 - bar_len)

            print(f"  {i:<8} {raw[2]:>10} {norm[2]:>10.3f} {bar}")

        print(f"\n  Samples received : {emg.samples_received}")
        print(f"  Parse errors     : {emg.parse_errors}")

        # Show a sample window
        print(f"\n  Sample window shape: {emg.get_window().shape}")
        print(f"  Ch3 window (last 10 values):")
        print(f"  {emg.get_window()[2, -10:]}")

    finally:
        emg.stop()
        print("\n  Self-test complete.")