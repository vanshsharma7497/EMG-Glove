"""
core/emg_features.py
--------------------
Extracts a 20-element feature vector from a filtered 4-channel EMG window.

Five time-domain features are computed per channel (MAV, RMS, ZCR, WL, VAR),
producing 5 × 4 = 20 features. This vector is what the SVM classifier trains on.

Architecture:
  Called by: emg_dataset.py (training), mode_manager.py (inference every frame)
  Exports:   extract_features(filtered_window) → np.ndarray shape (20,)
             get_feature_labels() → list of 20 label strings
"""
import numpy as np
import matplotlib.pyplot as plt
from core.emg_simulator import generate_emg_window, GESTURE_PROFILES
from core.emg_filter import filter_all_channels

# ── Feature Functions ─────────────────────────────────────────────────────────
# Each function takes a 1D array (one channel's filtered signal)
# and returns a single number summarizing some property of that signal.

def mean_absolute_value(channel):
    """
    MAV — the average signal strength.
    The most important EMG feature. Higher = more muscle activation.
    """
    return np.mean(np.abs(channel))


def root_mean_square(channel):
    """
    RMS — similar to MAV but more sensitive to large spikes.
    Closely related to the power of the signal.
    """
    return np.sqrt(np.mean(channel ** 2))


def zero_crossing_rate(channel):
    """
    ZCR — how many times the signal crosses zero per window.
    Higher frequency signals cross zero more often.
    Different gestures have different dominant frequencies, so this
    helps distinguish gestures that have similar amplitude.
    """
    crossings = np.diff(np.sign(channel))   # sign change = zero crossing
    return np.sum(crossings != 0)


def waveform_length(channel):
    """
    WL — total cumulative length of the signal waveform.
    Captures complexity + frequency + amplitude all in one number.
    A high-amplitude, high-frequency signal will have a very long waveform length.
    """
    return np.sum(np.abs(np.diff(channel)))


def variance(channel):
    """
    VAR — how spread out the signal values are around the mean.
    Resting muscle = low variance. Active muscle = high variance.
    """
    return np.var(channel)


# ── Extract All Features from One Window ─────────────────────────────────────

FEATURE_NAMES = ["MAV", "RMS", "ZCR", "WL", "VAR"]
FEATURE_FUNCTIONS = [mean_absolute_value, root_mean_square,
                     zero_crossing_rate, waveform_length, variance]

def extract_features(filtered_window):
    """
    Takes a filtered window of shape (4 channels, 200 samples).
    Returns a flat 1D feature vector of length 20 (5 features x 4 channels).
    
    This flat vector is what the ML model will train on.
    """
    feature_vector = []

    for ch in range(filtered_window.shape[0]):
        for fn in FEATURE_FUNCTIONS:
            feature_vector.append(fn(filtered_window[ch]))

    return np.array(feature_vector)   # shape: (20,)


def get_feature_labels():
    """Returns readable labels for all 20 features — useful for plotting."""
    return [f"Ch{ch+1}_{name}" for ch in range(4) for name in FEATURE_NAMES]


# ── Visualization ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gestures = list(GESTURE_PROFILES.keys())
    all_vectors = []

    for gesture in gestures:
        raw = generate_emg_window(gesture)
        filtered = filter_all_channels(raw)
        vector = extract_features(filtered)
        all_vectors.append(vector)
        print(f"{gesture:12s} → feature vector: {np.round(vector, 3)}")

    # Plot feature vectors as a heatmap so we can see how gestures differ
    all_vectors = np.array(all_vectors)   # shape: (5 gestures, 20 features)
    labels = get_feature_labels()

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(all_vectors, aspect="auto", cmap="YlOrRd")

    ax.set_yticks(range(len(gestures)))
    ax.set_yticklabels(gestures)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Feature Vectors per Gesture — brighter = higher value", fontsize=12)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()