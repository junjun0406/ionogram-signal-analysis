"""
Peak detection example using O/X mode separation pipeline.

Usage:
    python mode_peaks.py --file path/to/ionogram.sav --freq <MHz>
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from mode_separation import load_data, compute_mode_separation, subtract_noise


# ==============================
# Peak detection
# ==============================

def detect_peaks(profile, start_rg=200, end_rg=500):
    """Detect peaks in a 1D range-gate power profile within [start_rg, end_rg)."""
    valid = profile[start_rg:end_rg]

    peaks, props = find_peaks(
        valid,
        height=48,
        prominence=15,
        distance=10,
        width=10
    )

    return peaks + start_rg


# ==============================
# Main
# ==============================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--freq", type=float, default=8.0)
    args = parser.parse_args()

    data, rgt_array, freq_array = load_data(args.file)

    O_norm, X_norm = compute_mode_separation(data)

    # direct/cal remove
    O_norm[0:100, :] = 1
    O_norm[800:1000, :] = 1
    X_norm[0:100, :] = 1
    X_norm[800:1000, :] = 1

    O_norm = 10*np.log10(O_norm)
    X_norm = 10*np.log10(X_norm)

    num_freq = len(freq_array)

    O_norm = subtract_noise(O_norm, num_freq)
    X_norm = subtract_noise(X_norm, num_freq)

    # ==============================
    # Peak detection part
    # ==============================

    i = int(np.argmin(np.abs(freq_array - args.freq)))

    profile_X = X_norm[:, i]
    profile_O = O_norm[:, i]

    peaks_X = detect_peaks(profile_X)
    peaks_O = detect_peaks(profile_O)

    print("X peaks:", peaks_X)
    print("O peaks:", peaks_O)

    # ==============================
    # Plot
    # ==============================

    fig, axes = plt.subplots(1, 2, figsize=(6, 12))

    cmin = 0.8 * np.median(O_norm[50:800, 100:1400])
    cmax = 2.0 * np.median(O_norm[50:800, 100:1400])

    axes[0].pcolormesh(freq_array, rgt_array, O_norm, vmin=cmin, vmax=cmax, shading="auto")
    axes[1].pcolormesh(freq_array, rgt_array, X_norm, vmin=cmin, vmax=cmax, shading="auto")

    xmin = float(freq_array[0])
    xmax = float(freq_array[-1])

    for rgs in peaks_O:
        axes[0].hlines(rgs, xmin=xmin, xmax=xmax, colors="red", linestyles="--", linewidth=1)

    for rgs in peaks_X:
        axes[1].hlines(rgs, xmin=xmin, xmax=xmax, colors="red", linestyles="--", linewidth=1)

    axes[0].axvline(freq_array[i], color="red", linestyle="--", linewidth=1.5)
    axes[1].axvline(freq_array[i], color="red", linestyle="--", linewidth=1.5)

    axes[0].set_xlim([1, 15])
    axes[0].set_ylim([100, 800])
    axes[0].set_title("O mode")

    axes[1].set_xlim([1, 15])
    axes[1].set_ylim([100, 800])
    axes[1].set_title("X mode")

    plt.show()


if __name__ == "__main__":
    main()