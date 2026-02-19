import os
from typing import Tuple
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


# -----------------------------


# ファイル名の日時を抽出
def file_information(filename: str) -> Tuple[str, str, str, str, str]:

    # ファイル名だけ取得
    basename = os.path.basename(filename)
    # -> 'VP2TO_20250101130000.sav'

    # 拡張子を除去
    name_no_ext = os.path.splitext(basename)[0]
    # -> 'VP2TO_20250101130000'

    # '_' で分割して日時部分を取得
    s = name_no_ext.split('_')[1]
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    hour = s[8:10]
    minute = s[10:12]

    return year, month, day, hour, minute


# 指定した範囲外を全てマスク
# （マスクはマスクする部分を1(True)とする）
def make_ROI_mask(rgt_range: tuple[int, int], # 範囲指定
                  freq_range: tuple[float, float], # 範囲指定
                  freq_array: np.ndarray,
                  shape: tuple[int, int]
                  ) -> np.ndarray:

    r_bottom, r_top = rgt_range
    f_left = np.searchsorted(freq_array, freq_range[0], side="left")
    f_right = np.searchsorted(freq_array, freq_range[1], side="right")

    mask = np.ones(shape, dtype=bool)
        # 全部マスク
    mask[r_bottom:r_top + 1, f_left:f_right] = False
        # 指定した範囲だけマスク除去

    return mask


# 閾値を超えていないところをマスク
def make_noise_mask(antenna0: np.ndarray,
                    antenna1: np.ndarray,
                    thr: int,
                    ) -> np.ndarray:

    mag0 = np.abs(antenna0)
    mag1 = np.abs(antenna1)
    conf = mag0 * mag1

    mask = (conf < thr)
        # conf が小さいところはマスク

    return mask


# 指定した範囲内の縦線ノイズをマスク
# （この縦線ノイズ除去は、閾値ノイズ除去後にしか使えないことに注意）
def make_vertical_stripe_mask(antenna0: np.ndarray,
                              antenna1: np.ndarray,
                              rgt_range: tuple[int, int], # 範囲指定
                              freq_range: tuple[float, float], # 範囲指定
                              freq_array: np.ndarray,
                              base_mask: np.ndarray, # 閾値ノイズ除去
                              min_count_ratio: float,
                              ) -> np.ndarray:

    r_bottom, r_top = rgt_range
    f_left = np.searchsorted(freq_array, freq_range[0], side="left")
    f_right = np.searchsorted(freq_array, freq_range[1], side="right")

    roi = (slice(r_bottom, r_top + 1),slice(f_left, f_right))

    valid_roi = (
            (~base_mask[roi]) &
            (np.abs(antenna0)[roi] > 0) &
            (np.abs(antenna1)[roi] > 0)
    )
        # データが０ではない画素を有効画素とする

    valid_count = valid_roi.sum(axis=0)
        # 列ごとに有効画素数を数える
    min_count = int(np.ceil((r_top - r_bottom + 1) * min_count_ratio))
    bad_cols = np.where(valid_count >= min_count)[0] + f_left
        # 縦線と判定された列（ROIの中の相対インデックス）

    mask = np.zeros_like(antenna0, dtype=bool)
    mask[:, bad_cols] = True

    print(f"[vertical stripe noise] count = {len(bad_cols)} / {f_right - f_left}")

    return mask


# アンテナ間の位相差をプロット
def plot_phase_difference(antenna0: np.ndarray,
                          antenna1: np.ndarray,
                          rgt_array: np.ndarray,
                          freq_array: np.ndarray,
                          xlim: tuple[float, float], # プロットの範囲指定
                          ylim: tuple[int, int], # プロットの範囲指定
                          mask: np.ndarray, # ノイズマスク
                          ax,
                          dt: tuple[str, str, str, str, str],
                          ch_ant0: int, ch_ant1: int # プロットのタイトル
                         ) -> None:

    x_left, x_right = xlim
    y_bottom, y_top = ylim

    valid = ~mask

    dphi_deg_360 = np.full(antenna0.shape, np.nan, dtype=float)

    dphi_rad = np.angle(antenna1[valid] * np.conj(antenna0[valid]))
    dphi_deg_360[valid] = np.degrees(dphi_rad)

    phase_cmap = LinearSegmentedColormap.from_list(
        "phase_cycle_black_orange_red_blue_black",
        [
            (0.00, "black"),  # 0°
            (0.25, "blue"),  # 90°
            (0.50, "red"),  # 180°
            (0.75, "orange"),  # 270°
            (1.00, "black")  # 360°
        ]
    )

    pcm = ax.pcolormesh(
        freq_array,
        rgt_array,
        dphi_deg_360,
        shading="auto",
        cmap=phase_cmap,
        vmin=-180,
        vmax=180
    )

    ax.set_xlim([x_left, x_right])
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_ylim([y_bottom, y_top])
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("range gate")
    (y, mo, d, h, mi) = dt
    ax.set_title(f"{y}/{mo}/{d}, {h}:{mi} (ch{ch_ant1} − ch{ch_ant0})")

    fig = ax.figure
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Phase difference [deg]")
    cbar.set_ticks([-180, -90, 0, 90, 180])

    return


# ヒストグラムを作成する範囲を表示
def draw_histogram_ROI(rgt_range: tuple[int, int],
                       freq_range: tuple[float, float],
                       ax
                       ) -> None:

    ax.vlines(
        x=freq_range[0],
        ymin=rgt_range[0],
        ymax=rgt_range[1],
        colors='green',
        linestyles='--',
        linewidth=2
    )

    ax.vlines(
        x=freq_range[1],
        ymin=rgt_range[0],
        ymax=rgt_range[1],
        colors='green',
        linestyles='--',
        linewidth=2
    )

    ax.hlines(
        y=rgt_range[0],
        xmin=freq_range[0],
        xmax=freq_range[1],
        colors='green',
        linestyles='--',
        linewidth=2
    )

    ax.hlines(
        y=rgt_range[1],
        xmin=freq_range[0],
        xmax=freq_range[1],
        colors='green',
        linestyles='--',
        linewidth=2
    )

    return


# 位相差のプロットから範囲を指定してヒストグラムを作成
def make_histogram(antenna0: np.ndarray,
                   antenna1: np.ndarray,
                   mask: np.ndarray,
                   axes,
                   dt: tuple[str, str, str, str, str],
                   ch_ant0: int, ch_ant1: int #プロットのタイトル
                  ) -> None:

    # ①ヒストグラムの作成

    dphi_rad = np.angle(antenna1 * np.conj(antenna0))
    dphi_deg_360 = np.degrees(dphi_rad)

    valid = ~mask

    phi = dphi_deg_360[valid]
        # 一次元化

    # ヒストグラム作成
    bins = 360  # 1度刻み。粗くしたければ 180(2度), 72(5度) など
    hist, edges = np.histogram(phi, bins=bins, range=(-180, 180))

    # ビン中心（度）
    centers = (edges[:-1] + edges[1:]) / 2

    axes[0].plot(centers, hist)
    (y, mo, d, h, mi) = dt
    axes[0].set_title(f"histogram {y}/{mo}/{d}, {h}:{mi} (ch{ch_ant1} − ch{ch_ant0})")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Phase Diff [degree]")

    # ピーク検出＋表示
    peaks, props = find_peaks(
        hist,
        prominence=0.3 * hist.max(),  # 目立つ山だけ拾う（必要なら調整）
        distance=5  # 近すぎるピークをまとめる（度に対応）
    )

    for k, p in enumerate(peaks):
        x = centers[p]
        y = hist[p]
        axes[0].axvline(x, color="red", linestyle="--", linewidth=1,
                        label="Peak" if k == 0 else None)
        axes[0].text(x, y*1.1, f"{x:.1f}°", rotation=0, va="bottom",
                     ha="center", fontsize=14)


    # ②平滑化ヒストグラムの作成

    hist_s = gaussian_filter1d(hist.astype(float), sigma=2.0)
    w = np.clip(hist_s, 0, None)

    axes[1].plot(centers, w)
    axes[1].set_title("Smoothed histogram")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Phase Diff [degree]")

    # ピーク検出＋表示
    peaks, props = find_peaks(
        w,
        prominence=0.05 * np.max(w),
        distance=10
    )

    # 各ピークの「幅」を計算（結果はビン番号の小数）
    width_res = peak_widths(w, peaks, rel_height=0.5)
    left_ips, right_ips = width_res[2], width_res[3]

    for k, (p, li, ri) in enumerate(zip(peaks, left_ips, right_ips)):
        # 範囲を整数インデックスに
        i0 = max(0, int(np.floor(li)))
        i1 = min(len(w) - 1, int(np.ceil(ri)))

        x = centers[i0:i1+1]
        ww = w[i0:i1+1]
        if ww.sum() <= 0:
            continue

        # 局所重み付き平均
        mu = (x * ww).sum() / ww.sum()

        # x座標（角度）として線を引く
        axes[1].axvline(mu, color="red", linestyle="--", linewidth=1,
                        label="Mean" if k == 0 else None)

        axes[1].axvline(centers[i0], color="black", linestyle=":", linewidth=1,
                        label="Half width" if k == 0 else None)

        axes[1].axvline(centers[i1], color="black", linestyle=":", linewidth=1)

        # ラベル表示（yはピーク高さ w[p] を基準に）
        axes[1].text(mu, w[p] * 1.1, f"{mu:.1f}°",
                     ha="center", va="bottom", fontsize=14)
        axes[1].text(centers[i0], w[p] * 0.8, f"{centers[i0]:.1f}°",
                     ha="right", va="top", fontsize=14)
        axes[1].text(centers[i1], w[p] * 0.8, f"{centers[i1]:.1f}°",
                     ha="left", va="top", fontsize=14)


    # ③二つのヒストグラムの凡例/軸設定

    # 凡例の作成
    axes[0].legend()
    axes[1].legend()

    # 上段（raw）
    ymax_raw = hist.max() + 20
    # 下段（smoothed）
    ymax_smooth = w.max() + 20
    # 共通の ymax（大きい方に合わせる）
    ymax_common = max(ymax_raw, ymax_smooth)

    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(0, ymax_common)
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(0, ymax_common)

    axes[0].tick_params(axis="x", labelbottom=True)
        # 上段にも横軸のラベルをつける

    return


# -----------------------------


if __name__ == "__main__":

    # -----------------------------

    # パラメータ指定

    # 使用チャンネルを指定, 45,67,25,43
    ch_ant0 = 0
    ch_ant1 = 1
    # ノイズの閾値を指定
    THR_CONF = 700
    # 縦線ノイズ検出の割合を指定
    STRIPE_RATIO = 0.3
    # 位相差プロット/ヒストグラムの範囲を指定
    PHASE_ROI = (100, 500), (1, 15)

    HIST_ROI = (180, 380), (2, 11.8)

    # 位相差の表示範囲を指定
    PHASE_LIM = (1, 15), (100, 500)

    # -----------------------------

    # ファイル
    file = 'C:/Users/junse/Research/A_Ionogram_data2/202501/VP2TO_20250101160000.sav'
    data = sio.readsav(file)

    # ファイルの日時を抽出
    dt = file_information(file)

    # データ
    antenna0 = data['pulse_i'][ch_ant0] + 1j * data['pulse_q'][ch_ant0]
    antenna1 = data['pulse_i'][ch_ant1] + 1j * data['pulse_q'][ch_ant1]

    # 余計な信号を除去
    antenna0[0:100, :] = 0  # make them small values to avoid direct
    antenna0[800:1000, :] = 0  # make it small value to avoid cal signal
    antenna1[0:100, :] = 0  # make them small values to avoid direct
    antenna1[800:1000, :] = 0  # make it small value to avoid cal signal

    # rgt_arrayとfreq_arrayを作成
    num_rgt = data['pulse_i'][0].shape[0]
    rgt_array = np.arange(num_rgt)

    num_freq = data['pulse_i'][0].shape[1]
    freq_array = np.arange(
        data['sct'].frequency[0][0][1] / 1000.,
        data['sct'].frequency[0][0][2] / 1000. + 0.02,
        0.02
    )

    (phase_rgt, phase_freq) = PHASE_ROI
    (hist_rgt, hist_freq) = HIST_ROI

    (xlim, ylim) = PHASE_LIM

    #ノイズマスクを作成
    mask1 = make_ROI_mask(phase_rgt,
                          phase_freq,
                          freq_array,
                          antenna0.shape
                          )

    mask2 = make_noise_mask(antenna0, antenna1, THR_CONF)

    mask3 = make_vertical_stripe_mask(antenna0, antenna1,
                                      phase_rgt,
                                      phase_freq,
                                      freq_array,
                                      mask2,
                                      STRIPE_RATIO,
                                      )

    phase_diff_mask = mask1 | mask2 | mask3

    # 位相差をプロット
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    plot_phase_difference(antenna0, antenna1,
                          rgt_array, freq_array,
                          xlim,
                          ylim,
                          phase_diff_mask,
                          ax,
                          dt, ch_ant0, ch_ant1
                          )

    mask4 = make_ROI_mask(hist_rgt,
                          hist_freq,
                          freq_array,
                          antenna0.shape
                          )

    draw_histogram_ROI(hist_rgt,
                       hist_freq,
                       ax
                       )

    plt.show()

    histogram_mask = mask2 | mask3 | mask4

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    make_histogram(antenna0, antenna1,
                   histogram_mask,
                   axes,
                   dt, ch_ant0, ch_ant1
                   )

    plt.show()