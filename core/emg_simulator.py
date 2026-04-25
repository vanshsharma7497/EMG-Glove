"""
core/emg_simulator.py
---------------------
Generates synthetic EMG (electromyography) signals for all 16 gesture classes.

Each gesture is defined by a GESTURE_PROFILES entry that controls how strongly
each of the 4 sensor channels activates and at what dominant frequency.
generate_emg_window() returns one 200ms window (200 samples at 1kHz) of
amplitude-modulated Gaussian noise that mimics real muscle activity.

Architecture:
  Called by: emg_dataset.py (batch generation), emg_filter.py (test), main.py (sim loop)
  Exports:   GESTURE_PROFILES, SAMPLE_RATE, WINDOW_SIZE, generate_emg_window()
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 1000          # 1000 samples per second (1 kHz) — standard for EMG
WINDOW_DURATION = 0.2       # each signal window is 200ms long
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)   # = 200 samples per window

# ── Gesture Profiles ─────────────────────────────────────────────────────────
# Each gesture has a unique "signature" across 4 sensor channels.
# Think of each channel as one EMG sensor placed at a different muscle.
# 'amplitude' = how strongly that muscle activates (0.0 to 1.0 scale)
# 'frequency_boost' = shifts the dominant frequency of the signal slightly

GESTURE_PROFILES = {
    # ── Channel mapping ───────────────────────────────────────────────────────
    #   Ch1 = thumb flexor muscle
    #   Ch2 = index finger flexor muscle
    #   Ch3 = middle finger flexor muscle
    #   Ch4 = ring + pinky flexor muscle group
    #
    # Design rules:
    #   1. Each gesture has a unique amplitude PATTERN (which channels are high/low).
    #   2. If two gestures have a similar pattern, their frequency_boost values
    #      are made clearly different so the ZCR feature separates them.
    #   3. amplitude 0.05 = near-silent (muscle not engaged)
    #      amplitude 0.90 = strong contraction
    #   4. frequency_boost adds to the 80Hz EMG base frequency.
    #      boost 0 = 80Hz signal, boost 30 = 110Hz signal.

    # ── 1. Rest ───────────────────────────────────────────────────────────────
    # All muscles silent. Very low amplitude, no frequency boost.
    # This is the baseline the classifier compares everything against.
    "rest": {
        "amplitude":       [0.05, 0.05, 0.05, 0.05],
        "frequency_boost": [0,    0,    0,    0   ]
    },

    # ── 2. Thumb + Index tap ──────────────────────────────────────────────────
    # Thumb and index pinch together. Ch1 + Ch2 both high.
    # Ch3 and Ch4 nearly silent — middle/ring/pinky are relaxed.
    # High frequency boost on Ch2 to distinguish from thumbs_up (Ch1 only).
    "thumb_index_tap": {
        "amplitude":       [0.85, 0.90, 0.08, 0.05],
        "frequency_boost": [22,   28,   2,    1   ]
    },

    # ── 3. Thumb + Middle tap ─────────────────────────────────────────────────
    # Thumb and middle finger contract. Ch1 + Ch3 high, Ch2 + Ch4 low.
    # Frequency on Ch3 is clearly higher than Ch4 gestures to stay distinct.
    "thumb_middle_tap": {
        "amplitude":       [0.82, 0.08, 0.88, 0.08],
        "frequency_boost": [20,   2,    26,   2   ]
    },

    # ── 4. Thumb + Ring tap ───────────────────────────────────────────────────
    # Thumb and ring finger contract. Ch1 + Ch4 high, Ch2 + Ch3 low.
    # Ch4 boost = 22 here. Compare with thumb_pinky_tap below where Ch4 boost = 32.
    # That frequency gap is what separates these two despite similar patterns.
    "thumb_ring_tap": {
        "amplitude":       [0.80, 0.07, 0.10, 0.83],
        "frequency_boost": [19,   2,    3,    22  ]
    },

    # ── 5. Thumb + Pinky tap ──────────────────────────────────────────────────
    # Also Ch1 + Ch4 high — same pattern as thumb_ring_tap.
    # KEY DIFFERENCE: Ch4 amplitude is slightly higher (0.90 vs 0.83)
    # AND Ch4 frequency_boost is 32 vs 22 — a 10-point gap the ZCR feature
    # will reliably detect. Ch1 amplitude is also slightly lower.
    "thumb_pinky_tap": {
        "amplitude":       [0.74, 0.05, 0.07, 0.90],
        "frequency_boost": [16,   1,    2,    32  ]
    },

    # ── 6. Index + Ring tap ───────────────────────────────────────────────────
    # Index and ring fire together — NO thumb involvement.
    # Ch2 + Ch4 high, Ch1 + Ch3 low.
    # This is the only gesture where Ch1 (thumb) is low AND Ch2 + Ch4 are high.
    # Very distinctive pattern — no confusion risk.
    "index_ring_tap": {
        "amplitude":       [0.07, 0.85, 0.08, 0.83],
        "frequency_boost": [2,    24,   2,    20  ]
    },

    # ── 7. Middle + Pinky tap ─────────────────────────────────────────────────
    # Middle and pinky fire together — NO thumb, NO index.
    # Ch3 + Ch4 high, Ch1 + Ch2 low.
    # Only gesture where Ch1 and Ch2 are both low while Ch3 and Ch4 are both high.
    "middle_pinky_tap": {
        "amplitude":       [0.06, 0.07, 0.86, 0.84],
        "frequency_boost": [1,    2,    23,   27  ]
    },

    # ── 8. Index double tap ───────────────────────────────────────────────────
    # Index fires alone, very strongly, at high frequency.
    # The high frequency_boost (35) on Ch2 makes this very distinct from
    # thumb_index_tap (where Ch1 is also high and Ch2 boost is only 28).
    # Double-tap timing is handled by the mode controller, not the classifier —
    # this profile represents the muscle signature of a single rapid index fire.
    "index_double_tap": {
        "amplitude":       [0.10, 0.95, 0.08, 0.05],
        "frequency_boost": [3,    35,   2,    1   ]
    },

    # ── 9. Victory ────────────────────────────────────────────────────────────
    # Index AND middle extended upward, ring and pinky curled down.
    # Ch2 + Ch3 high together. Ch1 (thumb) low — thumb is tucked.
    # Ch4 low — ring/pinky are curled inward (flexors engaged but not strongly).
    # Only gesture with Ch2 + Ch3 both high and Ch1 low.
    "victory": {
        "amplitude":       [0.08, 0.82, 0.84, 0.10],
        "frequency_boost": [2,    21,   24,   3   ]
    },

    # ── 10. Thumbs up ─────────────────────────────────────────────────────────
    # Thumb extended upward, all 4 fingers curled into fist.
    # Ch1 (thumb extensor) dominant. Ch2/Ch3/Ch4 low — fingers are curled
    # but passively, so their flexors aren't strongly activated.
    # Only gesture where Ch1 alone is clearly high.
    "thumbs_up": {
        "amplitude":       [0.88, 0.08, 0.07, 0.06],
        "frequency_boost": [30,   2,    2,    1   ]
    },

    # ── 11. Flex (fingers spread wide) ───────────────────────────────────────
    # All fingers spread as far as possible — maximum extension.
    # All 4 channels activate moderately and evenly.
    # Differs from rest: amplitude is 12x higher (~0.60 vs 0.05).
    # Differs from any tap: no single channel dominates.
    # Lower frequency boost across all channels — extension is slower than a tap.
    "flex": {
        "amplitude":       [0.62, 0.60, 0.61, 0.60],
        "frequency_boost": [10,   11,   10,   11  ]
    },

    # ── 12. Index fold ────────────────────────────────────────────────────────
    # Index finger curled DOWN, all other fingers extended.
    # Ch2 (index flexor) is actively contracting → but wait — if index is
    # CURLED, its flexor IS active. The KEY insight: the OTHER fingers are
    # extended and their extensors are active too (Ch1, Ch3, Ch4 moderate-high).
    # Ch2 is NOT low — it's the curling finger so its flexor fires.
    # What makes this unique: Ch1 + Ch3 + Ch4 all high, Ch2 ALSO high but
    # at a LOWER frequency than the others, and overall pattern differs
    # from flex because thumb is more dominant and Ch2 frequency is suppressed.
    #
    # Actually — correcting the physical model:
    # In EMG, we measure FLEXOR activation. Index curled = index flexor active.
    # Other fingers extended = their flexors are RELAXED (extensors do the work,
    # but we don't have extensor sensors in this 4-channel setup).
    # So index_fold = Ch2 HIGH (curling), Ch1/Ch3/Ch4 LOW-MODERATE (relaxed).
    # This is distinct from index_double_tap because Ch2 amplitude is lower
    # and frequency_boost is much lower (sustained hold, not a fast tap).
    "index_fold": {
        "amplitude":       [0.15, 0.78, 0.14, 0.12],
        "frequency_boost": [4,    14,   3,    3   ]
    },

    # ── 13. Middle double tap ─────────────────────────────────────────────────
    # Middle finger fires alone, very strongly, at high frequency.
    # Mirror of index_double_tap but on Ch3 instead of Ch2.
    # Ch3 amplitude (0.95) and frequency_boost (35) are both maximised to
    # create a sharp, fast burst — the same logic as index_double_tap.
    # Ch2 being low (0.08) distinguishes this from victory (Ch2 + Ch3 both high).
    # Double-tap timing is handled by the mode controller, not the classifier.
    "middle_double_tap": {
        "amplitude":       [0.08, 0.08, 0.95, 0.06],
        "frequency_boost": [2,    2,    35,   1   ]
    },

    # ── 14. Ring double tap ───────────────────────────────────────────────────
    # Ring finger fires alone, very strongly, at high frequency.
    # Ch4 is the ring+pinky shared channel — firing it alone at high frequency
    # and with Ch1 (thumb) LOW is what separates this from thumb_ring_tap
    # (Ch1 + Ch4 high) and thumb_pinky_tap (Ch1 + Ch4 high, higher boost).
    # The high frequency_boost (35) on Ch4 also separates it from the sustained
    # Ch4 activations in thumb_ring_tap (boost 22) and thumb_pinky_tap (boost 32).
    # Double-tap timing is handled by the mode controller, not the classifier.
    "ring_double_tap": {
        "amplitude":       [0.06, 0.07, 0.08, 0.95],
        "frequency_boost": [1,    2,    2,    35  ]
    },

    # ── 15. Fist ──────────────────────────────────────────────────────────────
    # All 4 fingers curl inward simultaneously — maximum flexion.
    # All channels high, similar amplitudes, moderate-high frequency.
    # KEY difference from flex (gesture 11):
    #   flex   = fingers spreading OUTWARD (extension) — lower amplitude (~0.61),
    #            lower frequency boost (~10–11)
    #   fist   = fingers curling INWARD (flexion)      — higher amplitude (~0.88),
    #            higher frequency boost (~18–20)
    # The amplitude gap (~0.27 per channel) and frequency gap (~8–9 points)
    # together give the classifier two independent features to separate them.
    "fist": {
        "amplitude":       [0.88, 0.87, 0.89, 0.88],
        "frequency_boost": [18,   20,   19,   18  ]
    },

    # ── 16. Spiderman (ring + middle bent, index + pinky extended) ────────────
    # Middle and ring fingers curl in → Ch3 + Ch4 high.
    # Index and pinky are extended → Ch2 low, Ch1 (thumb) low.
    # Closest neighbour: middle_pinky_tap (Ch3 + Ch4 high, Ch1 + Ch2 low).
    # Separation strategy: lower frequency boost than middle_pinky_tap.
    #   middle_pinky_tap boost: Ch3=23, Ch4=27
    #   spiderman        boost: Ch3=13, Ch4=15  ← clearly lower band
    # Also: spiderman is a SUSTAINED HOLD, not a tap — amplitude is slightly
    # lower (0.78–0.80 vs 0.84–0.86) reflecting a held contraction vs a snap.
    # Note: Ch4 covers ring+pinky together. In spiderman, the pinky is
    # EXTENDED, but the ring flexor fires — Ch4 will still activate because
    # the ring muscle group dominates that sensor's placement.
    "spiderman": {
        "amplitude":       [0.08, 0.10, 0.80, 0.78],
        "frequency_boost": [2,    3,    13,   15  ]
    },
}

def generate_emg_window(gesture_name, noise_level=0.02):
    """
    Generates one 200ms window of simulated EMG data for a given gesture.
    Returns a 2D array of shape (4 channels, 200 samples).
    
    How the simulation works:
    - Real EMG is bandlimited Gaussian noise whose amplitude & frequency
      content changes based on which muscles are active.
    - We model this by generating noise and shaping it with a sine wave
      at a gesture-specific frequency to mimic that frequency shift.
    """
    
    profile = GESTURE_PROFILES[gesture_name]
    num_channels = 4
    time = np.linspace(0, WINDOW_DURATION, WINDOW_SIZE)
    
    all_channels = []
    
    for ch in range(num_channels):
        amp = profile["amplitude"][ch]
        freq = 80 + profile["frequency_boost"][ch]   # base EMG freq ~80Hz + boost
        
        # Core EMG signal: amplitude-scaled noise shaped by a sine carrier
        # The sine wave mimics the dominant frequency component of muscle activation
        signal = amp * np.random.randn(WINDOW_SIZE) * np.sin(2 * np.pi * freq * time)
        
        # Add small background noise (always present, even at rest)
        signal += noise_level * np.random.randn(WINDOW_SIZE)
        
        all_channels.append(signal)
    
    return np.array(all_channels)   # shape: (4, 200)


# ── Quick Visual Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    gestures_to_plot = list(GESTURE_PROFILES.keys())   # plots all 16 gestures
    
    fig, axes = plt.subplots(len(gestures_to_plot), 4, figsize=(14, 10))
    fig.suptitle("Simulated EMG Signals — 4 Channels per Gesture", fontsize=14)
    
    for row, gesture in enumerate(gestures_to_plot):
        data = generate_emg_window(gesture)
        for ch in range(4):
            ax = axes[row][ch]
            ax.plot(data[ch], linewidth=0.7, color="steelblue")
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f"{gesture} | ch{ch+1}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()