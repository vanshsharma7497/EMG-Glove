"""
core/emg_filter.py
------------------
Three-stage EMG signal filter pipeline applied to every incoming window.

Stages (applied in order to each channel):
  1. Highpass  20Hz  — removes slow motion artifacts
  2. Notch     50Hz  — removes power-line interference
  3. Lowpass  450Hz  — removes high-frequency electronic noise

Architecture:
  Called by: emg_dataset.py, mode_manager.py (via ModeManager), main.py
  Imports:   SAMPLE_RATE from core.emg_simulator
  Exports:   filter_all_channels(raw_window) → filtered_window, same shape (4, 200)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from core.emg_simulator import generate_emg_window, SAMPLE_RATE

# ── Filter Definitions ────────────────────────────────────────────────────────
# scipy's butter() designs a Butterworth filter — a common choice because it has
# a maximally flat response in the passband (doesn't distort the signal it keeps).
# 
# Arguments:
#   N      = filter order (higher = steeper cutoff, but more computation)
#   Wn     = cutoff frequency, normalized as fraction of Nyquist frequency
#   btype  = 'high', 'low', or 'bandstop'
#   output = 'sos' (second-order sections) — numerically more stable than 'ba'

NYQUIST = SAMPLE_RATE / 2   # = 500 Hz — the maximum frequency we can represent

def make_highpass_filter(cutoff_hz=20, order=4):
    """Removes low-frequency motion artifacts below 20Hz."""
    return scipy_signal.butter(order, cutoff_hz / NYQUIST, btype='high', output='sos')

def make_notch_filter(notch_hz=50, quality_factor=30):
    """
    Removes power line interference at exactly 50Hz (use 60 if you're in the US).
    Quality factor controls how narrow the notch is — higher = more precise removal.
    """
    b, a = scipy_signal.iirnotch(notch_hz / NYQUIST, quality_factor)
    # Convert to sos for consistency
    return scipy_signal.tf2sos(b, a)

def make_lowpass_filter(cutoff_hz=450, order=4):
    """Removes high-frequency electronic noise above 450Hz."""
    return scipy_signal.butter(order, cutoff_hz / NYQUIST, btype='low', output='sos')


# ── Pre-computed filters ──────────────────────────────────────────────────────
# Built once at import time and reused every call to apply_filter_pipeline().
# Previously these were recreated on every call (300 objects/sec at 25fps × 4ch).
_HP    = make_highpass_filter()
_NOTCH = make_notch_filter()
_LP    = make_lowpass_filter()


def apply_filter_pipeline(raw_signal):
    """
    Runs a single channel signal through all three filters in sequence.
    raw_signal: 1D numpy array of one channel's samples
    Returns: filtered 1D numpy array, same length
    """
    # sosfilt applies the filter — we chain all three using pre-built filter objects
    filtered = scipy_signal.sosfilt(_HP,    raw_signal)
    filtered = scipy_signal.sosfilt(_NOTCH, filtered)
    filtered = scipy_signal.sosfilt(_LP,    filtered)

    return filtered


def filter_all_channels(raw_window):
    """
    Applies the full filter pipeline to all 4 channels.
    raw_window: shape (4, 200)
    Returns: filtered array, same shape (4, 200)
    """
    return np.array([apply_filter_pipeline(ch) for ch in raw_window])


# ── Visualization ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gesture = "flex"   # was: "fist" — not in GESTURE_PROFILES; fixed to valid gesture
    raw = generate_emg_window(gesture)
    
    # Add some fake 50Hz noise to channel 0 so we can see the notch filter work
    time = np.linspace(0, 0.2, raw.shape[1])
    raw[0] += 0.3 * np.sin(2 * np.pi * 50 * time)   # inject power line noise
    
    filtered = filter_all_channels(raw)

    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    fig.suptitle(f"Gesture: '{gesture}' — Raw vs Filtered (all 4 channels)", fontsize=13)

    for ch in range(4):
        # Raw signal
        axes[ch][0].plot(raw[ch], color="tomato", linewidth=0.7)
        axes[ch][0].set_title(f"Ch{ch+1} — Raw", fontsize=9)
        axes[ch][0].set_ylim(-1.5, 1.5)
        axes[ch][0].set_xticks([])

        # Filtered signal
        axes[ch][1].plot(filtered[ch], color="steelblue", linewidth=0.7)
        axes[ch][1].set_title(f"Ch{ch+1} — Filtered", fontsize=9)
        axes[ch][1].set_ylim(-1.5, 1.5)
        axes[ch][1].set_xticks([])

    plt.tight_layout()
    plt.show()