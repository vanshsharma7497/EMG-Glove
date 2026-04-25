"""
core/emg_dataset.py
-------------------
Generates and persists the labeled EMG feature dataset used to train the classifier.

Runs the full pipeline (simulate → filter → extract features) for each gesture and
each sample. Default: 200 samples × 16 gestures = 3200 total feature vectors.

Architecture:
  Called by: emg_classifier.py (training) and indirectly via LABEL_NAMES import
  Imports:   emg_simulator, emg_filter, emg_features
  Exports:   LABEL_NAMES (list of 16 gesture strings), generate_dataset(),
             save_dataset(), load_dataset()
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.emg_simulator import generate_emg_window, GESTURE_PROFILES
from core.emg_filter import filter_all_channels
from core.emg_features import extract_features
# ── Gesture label mapping ─────────────────────────────────────────────────────
# ML models work with numbers, not strings. So we map each gesture name to an
# integer — this is called encoding the labels.

GESTURE_LABELS = {name: idx for idx, name in enumerate(GESTURE_PROFILES.keys())}
LABEL_NAMES = list(GESTURE_PROFILES.keys())

# GESTURE_LABELS will look like:
# {'rest': 0, 'thumb_index_tap': 1, 'thumb_middle_tap': 2, ... 'index_fold': 11}


def generate_dataset(samples_per_gesture=200, noise_level=0.02):
    """
    Generates a full labeled dataset by simulating many windows per gesture.

    samples_per_gesture: how many windows to generate per gesture.
                         200 gives us 2400 total samples (200 x 12 gestures).
                         More = better model, slower training.

    Returns:
        X : feature matrix, shape (total_samples, 20)
        y : label array,   shape (total_samples,)
    """
    X = []   # will hold feature vectors
    y = []   # will hold integer labels

    for gesture_name, label in GESTURE_LABELS.items():
        print(f"  Generating {samples_per_gesture} samples for: {gesture_name}...")

        for _ in range(samples_per_gesture):
            # Full pipeline: simulate → filter → extract features
            raw = generate_emg_window(gesture_name, noise_level=noise_level)
            filtered = filter_all_channels(raw)
            features = extract_features(filtered)

            X.append(features)
            y.append(label)

    X = np.array(X)   
    y = np.array(y)   

    return X, y


def save_dataset(X, y, path="emg_dataset.npz"):
    """Save dataset to disk so we don't have to regenerate every time."""
    np.savez(path, X=X, y=y)
    print(f"\n  Dataset saved to {path}")


def load_dataset(path="emg_dataset.npz"):
    """Load a previously saved dataset."""
    data = np.load(path)
    return data["X"], data["y"]


# ── Generate, inspect, and save ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating dataset...")
    X, y = generate_dataset(samples_per_gesture=200)

    print(f"\nDataset ready:")
    print(f"  X shape : {X.shape}  ← (total samples, features)")
    print(f"  y shape : {y.shape}  ← (total samples,)")
    print(f"  Gestures: {GESTURE_LABELS}")

    # Check class balance — each gesture should have equal samples
    print(f"\nSamples per gesture:")
    for name, idx in GESTURE_LABELS.items():
        count = np.sum(y == idx)
        print(f"  {name:12s} (label {idx}): {count} samples")

    # Peek at one feature vector
    print(f"\nExample feature vector (sample 0, gesture=rest):")
    print(f"  {X[0]}")

    save_dataset(X, y)