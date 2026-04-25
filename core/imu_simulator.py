"""
core/imu_simulator.py
---------------------
Simulates the thumb-mounted MPU-6050 IMU (gyroscope) for the glove.

Provides discrete thumb motion states (still, roll_right, roll_left, etc.) and
converts raw gyro readings to cursor (dx, dy) pixel velocity via dead-zone
filtering, normalisation, and a non-linear acceleration curve.

Architecture:
  Called by: main.py simulator loop, mode_manager.py (when USE_HARDWARE=False)
  Replaced by: core/hardware_reader.py imu_to_cursor_velocity() in hardware mode
  Exports:   generate_imu_sample(), imu_to_cursor_velocity(), THUMB_STATES,
             IMU_SAMPLE_RATE, DEAD_ZONE, MAX_GYRO, CURSOR_SENSITIVITY
"""
import numpy as np

# ── IMU Constants ─────────────────────────────────────────────────────────────
IMU_SAMPLE_RATE = 100        # 100Hz is realistic for MPU-6050
DEAD_ZONE = 0.08             # thumb movements below this threshold are ignored
                             # prevents cursor drift when thumb is still
MAX_GYRO = 1.5               # maximum gyroscope reading (rad/s) we care about
CURSOR_SENSITIVITY = 20.0     # how fast cursor moves at full thumb roll
ACCELERATION_CURVE = 2.0     # exponent for acceleration curve
                             # >1 means slow rolls are very precise,
                             # fast rolls accelerate quickly — just like a trackpad

# ── Thumb Motion States ───────────────────────────────────────────────────────
# These describe what the thumb is doing at any given moment.
# In real hardware this comes from the MPU-6050 gyroscope readings.

THUMB_STATES = {
    "still":        {"gyro_x": 0.00, "gyro_y": 0.00},   # thumb not moving
    "roll_right":   {"gyro_x": 1.20, "gyro_y": 0.00},   # cursor → right
    "roll_left":    {"gyro_x":-1.20, "gyro_y": 0.00},   # cursor → left
    "roll_up":      {"gyro_x": 0.00, "gyro_y": 1.20},   # cursor → up
    "roll_down":    {"gyro_x": 0.00, "gyro_y":-1.20},   # cursor → down
    "roll_up_right":{"gyro_x": 0.85, "gyro_y": 0.85},   # diagonal
    "roll_up_left": {"gyro_x":-0.85, "gyro_y": 0.85},   # diagonal
}


def generate_imu_sample(thumb_state="still", noise_level=0.03):
    """
    Simulates one IMU reading from the thumb-mounted MPU-6050.
    
    Returns a dict with:
        gyro_x  : rotation around X axis (thumb rolling left/right)
        gyro_y  : rotation around Y axis (thumb rolling up/down)
        gyro_z  : rotation around Z axis (thumb twisting — not used for cursor)
        accel_x : linear acceleration X (used later for tap detection refinement)
        accel_y : linear acceleration Y
        accel_z : linear acceleration Z (gravity component)
    """
    state = THUMB_STATES.get(thumb_state, THUMB_STATES["still"])
    
    # Core gyroscope readings + realistic sensor noise
    gyro_x = state["gyro_x"] + np.random.normal(0, noise_level)
    gyro_y = state["gyro_y"] + np.random.normal(0, noise_level)
    gyro_z = np.random.normal(0, noise_level * 0.5)   # minimal Z movement
    
    # Accelerometer — thumb resting on palm means ~1g on Z axis (gravity)
    accel_x = np.random.normal(0, noise_level)
    accel_y = np.random.normal(0, noise_level)
    accel_z = 1.0 + np.random.normal(0, noise_level)  # gravity
    
    return {
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
    }


def apply_dead_zone(value, threshold=DEAD_ZONE):
    """
    If the gyroscope reading is below the threshold, treat it as zero.
    This prevents the cursor from drifting when the thumb is nearly still —
    the same behaviour as the dead zone in the centre of a joystick.
    """
    if abs(value) < threshold:
        return 0.0
    # Subtract the dead zone so motion starts from 0, not from the threshold
    return np.sign(value) * (abs(value) - threshold)


def apply_acceleration_curve(value, exponent=ACCELERATION_CURVE):
    """
    Applies a non-linear curve to the velocity.
    Small movements become even smaller (precision zone).
    Large movements become proportionally faster (speed zone).
    This is identical to how macOS/Windows trackpad acceleration works.
    """
    return np.sign(value) * (abs(value) ** exponent)


def imu_to_cursor_velocity(imu_sample):
    """
    Full pipeline: raw IMU reading → cursor (dx, dy) velocity in pixels/frame.
    
    Returns:
        dx : horizontal cursor velocity (positive = right)
        dy : vertical cursor velocity (positive = down)
    """
    # Step 1 — apply dead zone
    raw_x = apply_dead_zone(imu_sample["gyro_x"])
    raw_y = apply_dead_zone(imu_sample["gyro_y"])

    # Step 2 — normalize to 0..1 range based on MAX_GYRO
    norm_x = np.clip(raw_x / MAX_GYRO, -1.0, 1.0)
    norm_y = np.clip(raw_y / MAX_GYRO, -1.0, 1.0)

    # Step 3 — apply acceleration curve
    curved_x = apply_acceleration_curve(norm_x)
    curved_y = apply_acceleration_curve(norm_y)

    # Step 4 — scale to pixel velocity
    # Note: gyro_y positive = thumb rolls "up" = cursor moves up = negative dy
    dx = curved_x * CURSOR_SENSITIVITY
    dy = -curved_y * CURSOR_SENSITIVITY   # invert Y so up feels like up

    return dx, dy


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    states_to_test = ["still", "roll_right", "roll_left", "roll_up", "roll_down"]
    
    print(f"\n{'Thumb State':<16} {'gyro_x':>8} {'gyro_y':>8} {'dx':>8} {'dy':>8}")
    print("─" * 55)

    fig, axes = plt.subplots(1, len(states_to_test), figsize=(14, 3))
    fig.suptitle("Cursor Velocity per Thumb State", fontsize=12)

    for i, state in enumerate(states_to_test):
        # Generate 50 samples to show the noise distribution
        dxs, dys = [], []
        for _ in range(50):
            sample = generate_imu_sample(state)
            dx, dy = imu_to_cursor_velocity(sample)
            dxs.append(dx)
            dys.append(dy)

        print(f"{state:<16} {np.mean([generate_imu_sample(state)['gyro_x'] for _ in range(10)]):>8.3f} "
              f"{np.mean([generate_imu_sample(state)['gyro_y'] for _ in range(10)]):>8.3f} "
              f"{np.mean(dxs):>8.3f} {np.mean(dys):>8.3f}")

        # Plot dx/dy distribution as a scatter
        axes[i].scatter(dxs, dys, alpha=0.5, s=10, color="steelblue")
        axes[i].axhline(0, color="white", linewidth=0.5)
        axes[i].axvline(0, color="white", linewidth=0.5)
        axes[i].set_xlim(-10, 10)
        axes[i].set_ylim(-10, 10)
        axes[i].set_title(state.replace("_", "\n"), fontsize=8)
        axes[i].set_facecolor("#1a1a2e")
        axes[i].tick_params(colors="gray", labelsize=7)
        axes[i].set_xlabel("dx", fontsize=7)
        axes[i].set_ylabel("dy", fontsize=7)

    plt.tight_layout()
    plt.show()