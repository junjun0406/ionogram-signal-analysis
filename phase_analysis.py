"""
Ionogram Phase Difference Analysis (public version)

- Loads ionosonde .sav data
- Computes inter-antenna phase difference map
- Creates phase-difference histograms within a specified ROI
- Applies ROI mask + threshold mask + vertical stripe mask

Usage:
  python ionogram_phase_analysis.py --file path/to/ionogram.sav

Notes:
- This is a simplified, public-friendly CLI script.
- No file_information() (timestamp extraction) is used for public version.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# Reuse loader from mode_separation
from mode_separation import load_data


# ==============================
# Masks
# ==============================

# phase_mask: outside ROI OR low-confidence OR vertical-stripe columns

def make_ROI_mask(
    rgt_range: Tuple[int, int],
    freq_range: Tuple[float, float],
    freq_array: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Mask everything outside the specified ROI.
    Mask convention: True = masked, False = valid.
    """
    r_bottom, r_top = rgt_range
    f_left = np.searchsorted(freq_array, freq_range[0], side="left")
    f_right = np.searchsorted(freq_array, freq_range[1], side="right")

    mask = np.ones(shape, dtype=bool)
    mask[r_bottom:r_top + 1, f_left:f_right] = False
    return mask


def make_noise_mask(
    antenna0: np.ndarray,
    antenna1: np.ndarray,
    thr: float,
) -> np.ndarray:
    """
    Mask low-confidence pixels based on magnitude product.
    """
    conf = np.abs(antenna0) * np.abs(antenna1)
    return conf < thr


def make_vertical_stripe_mask(
    antenna0: np.ndarray,
    antenna1: np.ndarray,
    rgt_range: Tuple[int, int],
    freq_range: Tuple[float, float],
    freq_array: np.ndarray,
    base_mask: np.ndarray,
    min_count_ratio: float,
) -> np.ndarray:
    """
    Detect vertical stripe noise columns inside ROI after threshold masking.
    Mark columns with too many valid pixels as "stripe" (mask entire column).
    """
    r_bottom, r_top = rgt_range
    f_left = np.searchsorted(freq_array, freq_range[0], side="left")
    f_right = np.searchsorted(freq_array, freq_range[1], side="right")

    roi = (slice(r_bottom, r_top + 1), slice(f_left, f_right))

    valid_roi = (
        (~base_mask[roi]) &
        (np.abs(antenna0)[roi] > 0) &
        (np.abs(antenna1)[roi] > 0)
    )

    valid_count = valid_roi.sum(axis=0)
    min_count = int(np.ceil((r_top - r_bottom + 1) * min_count_ratio))
    bad_cols = np.where(valid_count >= min_count)[0] + f_left

    mask = np.zeros_like(antenna0, dtype=bool)
    mask[:, bad_cols] = True

    print(f"[vertical stripe noise] count = {len(bad_cols)} / {max(1, (f_right - f_left))}")
    return mask


# ==============================
# Plotting
# ==============================

def plot_phase_difference(
    antenna0: np.ndarray,
    antenna1: np.ndarray,
    rgt_array: np.ndarray,
    freq_array: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[int, int],
    mask: np.ndarray,
    ax,
    title: str,
) -> None:
    """
    Plot inter-antenna phase difference (deg) on an ionogram grid.
    """
    x_left, x_right = xlim
    y_bottom, y_top = ylim

    valid = ~mask

    dphi_deg = np.full(antenna0.shape, np.nan, dtype=float)
    dphi_rad = np.angle(antenna1[valid] * np.conj(antenna0[valid]))
    dphi_deg[valid] = np.degrees(dphi_rad)

    phase_cmap = LinearSegmentedColormap.from_list(
        "phase_cycle_black_orange_red_blue_black",
        [
            (0.00, "black"),   # -180
            (0.25, "blue"),    # -90
            (0.50, "red"),     # 0
            (0.75, "orange"),  # 90
            (1.00, "black"),   # 180
        ],
    )

    pcm = ax.pcolormesh(
        freq_array,
        rgt_array,
        dphi_deg,
        shading="auto",
        cmap=phase_cmap,
        vmin=-180,
        vmax=180,
    )

    ax.set_xlim([x_left, x_right])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim([y_bottom, y_top])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Range gate")
    ax.set_title(title)

    fig = ax.figure
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Phase difference (deg)")
    cbar.set_ticks([-180, -90, 0, 90, 180])


def draw_histogram_ROI(
    rgt_range: Tuple[int, int],
    freq_range: Tuple[float, float],
    ax,
) -> None:
    """
    Draw ROI box used for histogram on the phase-difference plot.
    """
    ax.vlines(freq_range[0], rgt_range[0], rgt_range[1], colors="green", linestyles="--", linewidth=2)
    ax.vlines(freq_range[1], rgt_range[0], rgt_range[1], colors="green", linestyles="--", linewidth=2)
    ax.hlines(rgt_range[0], freq_range[0], freq_range[1], colors="green", linestyles="--", linewidth=2)
    ax.hlines(rgt_range[1], freq_range[0], freq_range[1], colors="green", linestyles="--", linewidth=2)


def make_histogram(
    antenna0: np.ndarray,
    antenna1: np.ndarray,
    mask: np.ndarray,
    axes,
    title: str,
) -> None:
    """
    Build raw and smoothed histograms of phase difference within mask-valid region.
    """
    dphi_rad = np.angle(antenna1 * np.conj(antenna0))
    dphi_deg = np.degrees(dphi_rad)

    valid = ~mask
    phi = dphi_deg[valid]

    bins = 360
    hist, edges = np.histogram(phi, bins=bins, range=(-180, 180))
    centers = (edges[:-1] + edges[1:]) / 2

    # --- Raw histogram ---
    axes[0].plot(centers, hist)
    axes[0].set_title(f"Histogram (raw) - {title}")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Phase diff (deg)")

    peaks, _ = find_peaks(hist, prominence=0.3 * hist.max(), distance=5)
    for k, p in enumerate(peaks):
        x = centers[p]
        y = hist[p]
        axes[0].axvline(x, color="red", linestyle="--", linewidth=1, label="Peak" if k == 0 else None)
        axes[0].text(x, y * 1.1, f"{x:.1f}°", ha="center", va="bottom", fontsize=12)

    # --- Smoothed histogram ---
    hist_s = gaussian_filter1d(hist.astype(float), sigma=2.0)
    w = np.clip(hist_s, 0, None)

    axes[1].plot(centers, w)
    axes[1].set_title("Histogram (smoothed)")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Phase diff (deg)")

    peaks_s, _ = find_peaks(w, prominence=0.05 * np.max(w), distance=10)

    width_res = peak_widths(w, peaks_s, rel_height=0.5)
    left_ips, right_ips = width_res[2], width_res[3]

    for k, (p, li, ri) in enumerate(zip(peaks_s, left_ips, right_ips)):
        i0 = max(0, int(np.floor(li)))
        i1 = min(len(w) - 1, int(np.ceil(ri)))

        x = centers[i0:i1 + 1]
        ww = w[i0:i1 + 1]
        if ww.sum() <= 0:
            continue

        mu = (x * ww).sum() / ww.sum()

        axes[1].axvline(mu, color="red", linestyle="--", linewidth=1, label="Mean" if k == 0 else None)
        axes[1].axvline(centers[i0], color="black", linestyle=":", linewidth=1, label="Half-max edge" if k == 0 else None)
        axes[1].axvline(centers[i1], color="black", linestyle=":", linewidth=1)

        axes[1].text(mu, w[p] * 1.1, f"{mu:.1f}°", ha="center", va="bottom", fontsize=12)
        axes[1].text(
            centers[i0],
            w[p] * 0.9,
            f"{centers[i0]:.1f}°",
            ha="right",
            va="top",
            fontsize=11,
            color="black",
        )

        axes[1].text(
            centers[i1],
            w[p] * 0.9,
            f"{centers[i1]:.1f}°",
            ha="left",
            va="top",
            fontsize=11,
            color="black",
        )

    axes[0].legend()
    axes[1].legend()

    ymax_common = max(hist.max() + 20, w.max() + 20)
    for ax in axes:
        ax.set_xlim(-180, 180)
        ax.set_ylim(0, ymax_common)


# ==============================
# CLI / Main
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ionogram phase difference analysis (public)")
    p.add_argument("--file", required=True, help="Path to .sav file")

    p.add_argument("--thr_conf", type=float, default=700, help="Confidence threshold for noise mask")
    p.add_argument("--stripe_ratio", type=float, default=0.3, help="Min count ratio for stripe detection")

    # ROI parameters: rgt_bottom rgt_top / freq_left freq_right
    p.add_argument("--phase_rgt", type=int, nargs=2, default=[100, 500], metavar=("RGT0", "RGT1"))
    p.add_argument("--phase_freq", type=float, nargs=2, default=[1.0, 15.0], metavar=("F0", "F1"))

    p.add_argument("--hist_rgt", type=int, nargs=2, default=[100, 500], metavar=("RGT0", "RGT1"))
    p.add_argument("--hist_freq", type=float, nargs=2, default=[1.0, 15.0], metavar=("F0", "F1"))

    p.add_argument("--xlim", type=float, nargs=2, default=[1.0, 15.0], metavar=("X0", "X1"))
    p.add_argument("--ylim", type=int, nargs=2, default=[100, 500], metavar=("Y0", "Y1"))

    return p.parse_args()


def main() -> None:
    args = parse_args()

    data, rgt_array, freq_array = load_data(args.file)

    antenna0 = data["pulse_i"][0] + 1j * data["pulse_q"][0]
    antenna1 = data["pulse_i"][1] + 1j * data["pulse_q"][1]

    antenna0 = antenna0.copy()
    antenna1 = antenna1.copy()
    antenna0[0:100, :] = 0
    antenna0[800:1000, :] = 0
    antenna1[0:100, :] = 0
    antenna1[800:1000, :] = 0

    phase_rgt = (args.phase_rgt[0], args.phase_rgt[1])
    phase_freq = (args.phase_freq[0], args.phase_freq[1])
    hist_rgt = (args.hist_rgt[0], args.hist_rgt[1])
    hist_freq = (args.hist_freq[0], args.hist_freq[1])

    xlim = (args.xlim[0], args.xlim[1])
    ylim = (args.ylim[0], args.ylim[1])

    # Masks for phase plot
    mask_roi_phase = make_ROI_mask(phase_rgt, phase_freq, freq_array, antenna0.shape)
    mask_thr = make_noise_mask(antenna0, antenna1, args.thr_conf)
    mask_stripe = make_vertical_stripe_mask(
        antenna0, antenna1,
        phase_rgt, phase_freq, freq_array,
        base_mask=mask_thr,
        min_count_ratio=args.stripe_ratio,
    )
    phase_mask = mask_roi_phase | mask_thr | mask_stripe

    # --- Phase map plot ---
    title = f"Phase difference (antenna1 - antenna0)"
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_phase_difference(
        antenna0, antenna1,
        rgt_array, freq_array,
        xlim, ylim,
        phase_mask,
        ax,
        title=title,
    )

    # Histogram ROI overlay
    draw_histogram_ROI(hist_rgt, hist_freq, ax)
    plt.show()

    # --- Histogram plot ---
    mask_roi_hist = make_ROI_mask(hist_rgt, hist_freq, freq_array, antenna0.shape)
    histogram_mask = mask_thr | mask_stripe | mask_roi_hist

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    make_histogram(
        antenna0, antenna1,
        histogram_mask,
        axes,
        title=title,
    )
    plt.show()


if __name__ == "__main__":
    main()