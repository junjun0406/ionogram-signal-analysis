"""
Ionogram O/X Mode Separation

Public version for GitHub.
- Loads .sav ionosonde data
- Performs O/X mode separation
- Histogram-based noise reduction
- Visualization

Usage:
    python mode_separation.py --file path/to/ionogram.sav
"""

import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ==============================
# Data loading
# ==============================

def load_data(file_path: str):
    """Load ionogram .sav file."""
    data = sio.readsav(file_path)

    num_rgt = len(data['pulse_i'][0])
    rgt_array = np.arange(num_rgt)

    freq_array = np.arange(
        data['sct'].frequency[0][0][1] / 1000.,
        data['sct'].frequency[0][0][2] / 1000. + 0.02,
        0.02
    )

    return data, rgt_array, freq_array


# ==============================
# Mode separation
# ==============================

def compute_mode_separation(data, phase_rotation=1j):
    """
    O/X mode separation.

    phase_rotation : complex
        Complex phase rotation coefficient.
        Example:
            1j   -> 90 degree phase shift
           -1j   -> -90 degree
            1    -> no phase shift
    """

    # complex signals
    antA = data['pulse_i'][1] + 1j*data['pulse_q'][1]
    antB = data['pulse_i'][0] + 1j*data['pulse_q'][0]

    # --- mode separation ---
    O = antA + phase_rotation * antB
    X = phase_rotation * antA + antB

    O_norm = np.abs(O)**2
    X_norm = np.abs(X)**2

    return O_norm, X_norm


# ==============================
# Noise reduction
# ==============================

def subtract_noise(arr, num_freq):
    """Histogram-based noise reduction."""
    arr = arr.copy()

    for i in range(num_freq):
        tmp = arr[:, i]

        hist, bins = np.histogram(tmp[100:800], bins=100)
        peaks, _ = find_peaks(hist)

        if len(peaks) == 0:
            continue

        noise_level = bins[peaks[0]]
        arr[:, i] = tmp - noise_level

    return arr


# ==============================
# Plot
# ==============================

def plot_modes(O_norm, X_norm, freq_array, rgt_array):

    fig, axes = plt.subplots(1, 2, figsize=(6, 12))

    cmin = 0.8 * np.median(O_norm[50:800, 100:1400])
    cmax = 2.0 * np.median(O_norm[50:800, 100:1400])

    axes[0].pcolormesh(freq_array, rgt_array, O_norm, vmin=cmin, vmax=cmax)
    axes[0].set_xlim([1, 15])
    axes[0].set_ylim([100, 800])
    axes[0].set_title("O mode")

    axes[1].pcolormesh(freq_array, rgt_array, X_norm, vmin=cmin, vmax=cmax)
    axes[1].set_xlim([1, 15])
    axes[1].set_ylim([100, 800])
    axes[1].set_title("X mode")

    plt.show()


# ==============================
# Main
# ==============================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    data, rgt_array, freq_array = load_data(args.file)

    O_norm, X_norm = compute_mode_separation(data)

    # remove direct/cal region
    O_norm[0:100, :] = 1
    O_norm[800:1000, :] = 1
    X_norm[0:100, :] = 1
    X_norm[800:1000, :] = 1

    O_norm = 10*np.log10(O_norm)
    X_norm = 10*np.log10(X_norm)

    num_freq = len(freq_array)

    O_norm = subtract_noise(O_norm, num_freq)
    X_norm = subtract_noise(X_norm, num_freq)

    plot_modes(O_norm, X_norm, freq_array, rgt_array)


if __name__ == "__main__":
    main()