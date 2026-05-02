"""
mode_manager.py
---------------
Central router for all sensor data. Determines which controller handles
each frame based on the current device mode and pen dock state.

── Hardware Switch ───────────────────────────────────────────────────────────
Set USE_HARDWARE = True to use real sensors instead of simulators.
This is the only line you ever need to change when switching between
simulation and real hardware.

  USE_HARDWARE = False  →  runs on simulated data (original behaviour)
  USE_HARDWARE = True   →  reads from ESP32 via COM5

When USE_HARDWARE is True, the ModeManager:
  - Starts the HardwareReader (opens COM5, reads combined packet)
  - Calls reader.get_emg_window() instead of generate_emg_window()
  - Calls reader.get_imu_sample() instead of generate_imu_sample()
  - Everything above the sensor layer is completely unchanged

── Two top-level concepts ────────────────────────────────────────────────────

  device_mode  : the overall behavioural mode the glove is in.

    "standard"       → original behaviour: mouse (glove) + writing (pen)
    "communication"  → Communication Mode: CommunicationController handles
                       both pen frames (writing) and glove gestures (phrases)
    "tv_remote"      → TV Remote Mode: TVRemoteController handles all glove
                       gestures as TV commands (select, volume, skip, etc.)

  glove_mode   : which standard glove controller to use when pen is docked
                 and device_mode is "standard".

    "computer_mouse" → MouseController (currently the only implemented one)

── Routing table ─────────────────────────────────────────────────────────────

  device_mode = "standard", pen docked:
      EMG + IMU → process_emg_frame() → MouseController

  device_mode = "standard", pen undocked:
      Pen sensors → process_pen_frame() → WritingController

  device_mode = "communication", pen undocked:
      Pen sensors → process_pen_frame() → CommunicationController

  device_mode = "communication", pen docked:
      EMG classified → gesture name → CommunicationController.process_glove_gesture()

── Switching modes ────────────────────────────────────────────────────────────
  manager.set_device_mode("communication")   # enter Communication Mode
  manager.set_device_mode("standard")        # return to standard behaviour
"""

import joblib
import numpy as np

# ── HARDWARE SWITCH ───────────────────────────────────────────────────────────
# Change this one line to switch between simulation and real hardware.
#
#   False = simulator mode (original behaviour, no hardware needed)
#   True  = hardware mode  (reads from ESP32 on COM5)
#
USE_HARDWARE = False   # ← change to True when hardware is connected

# ── PEN HARDWARE SWITCH ──────────────────────────────────────────────────────
# Change this one line to switch between pen simulation and real pen hardware.
#
#   False = pen simulator mode (original behaviour, no pen hardware needed)
#   True  = pen hardware mode  (reads from pen ESP32 on COM6)
#
USE_PEN_HARDWARE = True   # ← change to True when pen hardware is connected

# ── Hardware serial ports ─────────────────────────────────────────────────────
# Only used when the corresponding USE_*_HARDWARE = True
HARDWARE_PORT     = "COM11"
PEN_HARDWARE_PORT = "COM11"

# ── Conditional imports ───────────────────────────────────────────────────────
if USE_HARDWARE:
    from core.hardware_reader import HardwareReader, imu_to_cursor_velocity
    from core.emg_filter      import filter_all_channels
    from core.emg_features    import extract_features
else:
    from core.emg_simulator   import generate_emg_window, GESTURE_PROFILES
    from core.emg_filter      import filter_all_channels
    from core.emg_features    import extract_features
    from core.imu_simulator   import generate_imu_sample, imu_to_cursor_velocity
    from core.pen_simulator   import pen_docked, slider_position

if USE_PEN_HARDWARE:
    from core.pen_hardware_reader import PenHardwareReader

from core.emg_dataset    import LABEL_NAMES

from modes.mouse_mode          import MouseController
from modes.writing_mode        import WritingController
from modes.communication_mode  import CommunicationController
from modes.tv_remote_mode      import TVRemoteController
from tv_simulator              import TVCommandSimulator

# ── Available glove modes (pen docked, device_mode = "standard") ─────────────
GLOVE_MODES = {
    "computer_mouse"    : MouseController,
    # "computer_keyboard" : KeyboardController,   # future
    # "television"        : RemoteController,     # future
    # "wheelchair"        : WheelchairController, # future
    # "musician"          : MusicianController,   # future
}

# ── Available device modes ────────────────────────────────────────────────────
DEVICE_MODES = {
    "standard"       : "Standard mode — mouse (glove) + writing (pen)",
    "communication"  : "Communication Mode — non-verbal user interface",
    "tv_remote"      : "TV Remote Mode — gesture-controlled TV interface",
}

# ── Glove gestures that Communication Mode handles ────────────────────────────
COMM_MODE_GLOVE_GESTURES = {
    "thumb_index_tap",
    "thumb_middle_tap",
    "thumb_ring_tap",
    "thumb_pinky_tap",
    "index_double_tap",
    "index_ring_tap",     # NEW — needs classifier retraining
    # Removed: "middle_index_tap", "pinky_ring_tap" — not in GESTURE_PROFILES vocabulary
}

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATH  = "models/emg_model.pkl"
SCALER_PATH = "models/emg_scaler.pkl"


class ModeManager:
    """
    Central router for all sensor data.

    Usage — simulator mode (USE_HARDWARE = False):
        manager = ModeManager(glove_mode="computer_mouse")
        manager.run_simulation_loop()   # runs the demo sequence

    Usage — hardware mode (USE_HARDWARE = True):
        manager = ModeManager(glove_mode="computer_mouse")
        manager.run_hardware_loop()     # reads from ESP32

    Switching device modes at runtime:
        manager.set_device_mode("communication")
        manager.set_device_mode("standard")
    """

    def __init__(self, glove_mode="computer_mouse", device_mode="standard",
                 mode=None):
        # Support legacy 'mode' parameter from original main.py
        if mode is not None:
            glove_mode = mode

        print(f"\n  ModeManager initializing")
        print(f"  Hardware mode : {'REAL HARDWARE' if USE_HARDWARE else 'SIMULATOR'}")
        print(f"  Device mode   : '{device_mode}'")
        print(f"  Glove mode    : '{glove_mode}'")

        if glove_mode not in GLOVE_MODES:
            raise ValueError(
                f"Unknown glove mode '{glove_mode}'. "
                f"Available: {list(GLOVE_MODES.keys())}"
            )
        if device_mode not in DEVICE_MODES:
            raise ValueError(
                f"Unknown device mode '{device_mode}'. "
                f"Available: {list(DEVICE_MODES.keys())}"
            )

        # ── EMG classifier ────────────────────────────────────────────────────
        self.clf    = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        print(f"  EMG model loaded from {MODEL_PATH}")

        # ── Hardware reader (only when USE_HARDWARE = True) ───────────────────
        self.hw_reader = None
        if USE_HARDWARE:
            self.hw_reader = HardwareReader(port=HARDWARE_PORT)
            self.hw_reader.start()
            print(f"  Hardware reader started on {HARDWARE_PORT}")

        # ── Pen hardware reader (only when USE_PEN_HARDWARE = True) ───────────
        self.pen_hw_reader = None
        if USE_PEN_HARDWARE:
            self.pen_hw_reader = PenHardwareReader(port=PEN_HARDWARE_PORT)
            self.pen_hw_reader.start()
            print(f"  Pen hardware reader started on {PEN_HARDWARE_PORT}")

        # ── Controllers ───────────────────────────────────────────────────────
        self.glove_controller = GLOVE_MODES[glove_mode]()
        self.pen_controller   = WritingController()
        self.glove_mode       = glove_mode
        self.comm_controller  = CommunicationController()
        self.tv_simulator     = TVCommandSimulator()
        self.tv_controller    = TVRemoteController(self.tv_simulator)

        self.device_mode   = device_mode
        self._last_context = None

        print(f"  All controllers ready")
        print(f"  Active: {DEVICE_MODES[device_mode]}\n")


    # =========================================================================
    # Hardware loop — called by main.py when USE_HARDWARE = True
    # =========================================================================

    def process_hardware_frame(self):
        """
        Called once per frame in the hardware main loop.

        Reads one sample from the hardware reader, runs the full pipeline,
        and returns the predicted gesture + cursor velocity.

        Returns:
            pred_gesture : str   — predicted gesture label
            dx           : float — cursor x velocity
            dy           : float — cursor y velocity
        """
        if not USE_HARDWARE or self.hw_reader is None:
            raise RuntimeError(
                "process_hardware_frame() called but USE_HARDWARE = False. "
                "Set USE_HARDWARE = True in mode_manager.py first."
            )

        # Get latest sensor data
        imu_sample = self.hw_reader.get_imu_sample()
        emg_window = self.hw_reader.get_emg_window()

        # IMU → cursor velocity
        dx, dy = imu_to_cursor_velocity(imu_sample)

        # EMG → features → classifier
        filtered        = filter_all_channels(emg_window)
        features        = extract_features(filtered)
        features_scaled = self.scaler.transform([features])
        pred_gesture    = LABEL_NAMES[int(self.clf.predict(features_scaled)[0])]

        # Route to active controller
        self.process_emg_frame(features, dx, dy)

        return pred_gesture, dx, dy


    # =========================================================================
    # Mode switching
    # =========================================================================

    def set_device_mode(self, mode):
        """Switch between device modes at runtime."""
        if mode not in DEVICE_MODES:
            raise ValueError(
                f"Unknown device mode '{mode}'. "
                f"Available: {list(DEVICE_MODES.keys())}"
            )
        if mode == self.device_mode:
            return
        print(f"  [ModeManager] Device mode: '{self.device_mode}' → '{mode}'")
        self.device_mode   = mode
        self._last_context = None


    def stop(self):
        """Clean shutdown — stops hardware readers if running."""
        if self.hw_reader is not None:
            self.hw_reader.stop()
        if self.pen_hw_reader is not None:
            self.pen_hw_reader.stop()


    # =========================================================================
    # Glove context — pen docked
    # =========================================================================

    def process_emg_frame(self, features, dx=0.0, dy=0.0):
        """
        Called every frame when pen is docked.
        Classifies EMG features → gesture label, then routes to controller.
        Returns the predicted gesture label string.
        """
        self._log_context_switch("glove")

        features_scaled = self.scaler.transform([features])
        pred_gesture    = LABEL_NAMES[int(self.clf.predict(features_scaled)[0])]

        if self.device_mode == "communication":
            if pred_gesture in COMM_MODE_GLOVE_GESTURES:
                self.comm_controller.process_glove_gesture(pred_gesture)
        elif self.device_mode == "tv_remote":
            self.tv_controller.process_frame(pred_gesture, dx, dy)
        else:
            self.glove_controller.process_frame(pred_gesture, dx, dy)

        return pred_gesture


    # =========================================================================
    # Pen context — pen undocked
    # =========================================================================

    def process_pen_frame(self, pressure=None, gyro_x=None, gyro_y=None,
                           slider=2, gyro_z=0.0):
        """
        Called every frame when pen is undocked.
        When USE_PEN_HARDWARE is True and arguments are None, reads live
        values from self.pen_hw_reader instead of using passed-in values.
        """
        self._log_context_switch("pen")

        # ── Read live pen hardware if available ───────────────────────────
        if USE_PEN_HARDWARE and self.pen_hw_reader is not None:
            imu     = self.pen_hw_reader.get_imu_sample()
            pressure = self.pen_hw_reader.get_pressure()
            gyro_x   = imu["gyro_x"]
            gyro_y   = imu["gyro_y"]
            gyro_z   = imu["gyro_z"]
            slider   = self.pen_hw_reader.get_slider_position()

        if self.device_mode == "communication":
            return self.comm_controller.process_pen_frame(
                pressure, gyro_x, gyro_y, gyro_z=gyro_z
            )
        else:
            return self.pen_controller.process_frame(
                pressure, gyro_x, gyro_y, slider_position=slider, gyro_z=gyro_z
                # gyro_z was missing here — required for tilt-scroll in cursor mode (slider=1)
            )


    # =========================================================================
    # Unified auto-router
    # =========================================================================

    def process_frame_auto(self, features, emg_dx, emg_dy,
                           pressure, pen_gyro_x, pen_gyro_y,
                           slider=2, pen_gyro_z=0.0):
        """
        Unified entry point — checks pen_docked each frame and routes
        automatically to the correct controller.
        """
        # Read pen_docked from the correct source
        if USE_PEN_HARDWARE and self.pen_hw_reader is not None:
            import core.pen_hardware_reader as phr
            docked = phr.pen_docked
        else:
            import core.pen_simulator as ps
            docked = ps.pen_docked

        if docked:
            gesture = self.process_emg_frame(features, emg_dx, emg_dy)
            return {
                "device_mode" : self.device_mode,
                "context"     : "glove",
                "gesture"     : gesture,
                "pen_data"    : None,
            }
        else:
            pen_status = self.process_pen_frame(
                pressure, pen_gyro_x, pen_gyro_y,
                slider=slider, gyro_z=pen_gyro_z,
            )
            return {
                "device_mode" : self.device_mode,
                "context"     : "pen",
                "gesture"     : None,
                "pen_data"    : pen_status,
            }

    # =========================================================================
    # Simulator frame — called by original main.py (USE_HARDWARE = False)
    # =========================================================================

    def process_frame(self, features, dx, dy):
        """
        Original entry point used by main.py simulator loop.
        Unchanged from v4.0 — routes features + IMU to active controller.
        Returns predicted gesture label.
        """
        return self.process_emg_frame(features, dx, dy)


    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _log_context_switch(self, context):
        if context != self._last_context:
            label = "Glove (EMG)" if context == "glove" else "Pen"
            print(f"  [ModeManager] Context → {label}  |  Mode: {self.device_mode}")
            self._last_context = context


    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def active_context(self):
        if USE_PEN_HARDWARE and self.pen_hw_reader is not None:
            import core.pen_hardware_reader as phr
            return "glove" if phr.pen_docked else "pen"
        else:
            import core.pen_simulator as ps
            return "glove" if ps.pen_docked else "pen"

    @property
    def writing_text(self):
        return self.pen_controller.display_text

    @property
    def current_word(self):
        return self.pen_controller.current_word

    @property
    def completions(self):
        return self.pen_controller.completions

    @property
    def comm_preview(self):
        return self.comm_controller.preview_text

    @property
    def comm_confirmed(self):
        return self.comm_controller.confirmed_message

    @property
    def comm_history(self):
        return self.comm_controller.conversation_history

    @property
    def comm_volume(self):
        return self.comm_controller.volume