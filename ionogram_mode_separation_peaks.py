import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# -------------------------------------
file = 'C:/Users/junse/Research/A_Ionogram_data2/2024/202401/05/vertical/VP2TO_20240105130000.sav'
data = sio.readsav(file)

num_rgt = len(data['pulse_i'][0])  # 1000
rgt_array = np.arange(0, num_rgt)  # 0 kara 999
freq_array = np.arange(data['sct'].frequency[0][0][1] / 1000., data['sct'].frequency[0][0][2] / 1000. + 0.02, 0.02)
print(freq_array, len(freq_array))  # 1 kara 30
num_freq = len(freq_array)  # 1451

X_i = 0
X_q = 0
O_i = 0
O_q = 0

# O = (A+jB) + j(C+jD)
# X = j(A+jB) + (C+jD)
for i in range(0, 8):

    if i % 2 == 0: # X軸 C+jD

        print("added for channel " + str(i))

        # j(C+jD) = -D+jC
        O_i = O_i - data['pulse_q'][i] #-D
        O_q = O_q + data['pulse_i'][i] #C

        # C+jD
        X_i = X_i + data['pulse_i'][i] #C
        X_q = X_q + data['pulse_q'][i] #D

    else: # Y軸 A+jB

        print("added for channel " + str(i))

        # A+jB
        O_i = O_i + data['pulse_i'][i] #A
        O_q = O_q + data['pulse_q'][i] #B

        # j(A+jB) = -B+jA
        X_i = X_i - data['pulse_q'][i] #-B
        X_q = X_q + data['pulse_i'][i] #A

X_norm = (X_i ** 2 + X_q ** 2)  # **0.5 # 1000 x 1451
O_norm = (O_i ** 2 + O_q ** 2)  # **0.5 # 1000 x 1451

X_norm = X_norm[:, :1451]
X_norm[0:100, :] = 1  # make them small values to avoid direct
X_norm[800:1000, :] = 1  # make it small value to avoid cal signal
X_norm = 10 * np.log10(X_norm)

O_norm = O_norm[:, :1451]
O_norm[0:100, :] = 1  # make them small values to avoid direct
O_norm[800:1000, :] = 1  # make it small value to avoid cal signal
O_norm = 10 * np.log10(O_norm)


##----noise reduction----
for i_freq in range(0, num_freq - 1):
    X_tmp = X_norm[:, i_freq]
    O_tmp = O_norm[:, i_freq]

    # ヒストグラムを計算
    hist, bins = np.histogram(X_tmp[100:800], bins=100)
    peaks, _ = find_peaks(hist)
    noise_level = bins[peaks[0]]
    X_norm[:, i_freq] = X_tmp - noise_level

    # ヒストグラムを計算
    hist, bins = np.histogram(O_tmp[100:800], bins=100)
    peaks, _ = find_peaks(hist)
    noise_level = bins[peaks[0]]
    O_norm[:, i_freq] = O_tmp - noise_level


##----plot------
fig, axes = plt.subplots(1, 2, figsize=(6, 12))

cmin = 0.8 * np.median(O_norm[50:800, 100:1400])

cmax = 2.0 * np.median(O_norm[50:800, 100:1400])

axes[0].pcolormesh(freq_array, rgt_array, O_norm, vmin=cmin, vmax=cmax)

axes[0].set_xlim([1, 14])
axes[0].set_xlabel("frequency(MHz)")

axes[0].set_ylim([100, 800])
axes[0].set_ylabel("range gate")

axes[0].set_title("O mode")

axes[1].pcolormesh(freq_array, rgt_array, X_norm, vmin=cmin, vmax=cmax)

axes[1].set_xlim([1, 14])
axes[1].set_xlabel("frequency(MHz)")

axes[1].set_ylim([100, 800])
axes[1].set_ylabel("range gate")

axes[1].set_title("X mode")


# 周波数 f_i でのピーク点の表示

# 周波数 f_i の指定
freq_data = 7.8

# Xモード
i = int(round((freq_data+0.02-1)/0.02))
profile_X = X_norm[:, i]
profile_X_valid = profile_X[200:500]

peaks, properties = find_peaks(
    profile_X_valid,
    height = 48,
    prominence = 15,
    distance = 10,
    width = 10
)

peaks_rgs = peaks + 200
for rgs in peaks_rgs:
    peak_value = X_norm[rgs, i]
    print(f"X_mode 周波数インデックス{i}: 遅延時間番号 {rgs}, ピーク値 {peak_value}")

for rgs in peaks_rgs:
    axes[1].hlines(y=rgs, xmin=0, xmax=num_freq, colors='red', linestyles='--', linewidth=1)

axes[1].vlines(x=freq_data, ymin=0, ymax=num_rgt, colors='red', linestyles='--', linewidth=1.5)

# Oモード
i = int(round((freq_data+0.02-1)/0.02))
profile_O = O_norm[:, i]
profile_O_valid = profile_O[200:500]

peaks, properties = find_peaks(
    profile_O_valid,
    height = 48,
    prominence = 15,
    distance = 10,
    width = 10
)

peaks_rgs = peaks + 200
for rgs in peaks_rgs:
    peak_value = O_norm[rgs, i]
    print(f"O_mode 周波数インデックス{i}: 遅延時間番号 {rgs}, ピーク値 {peak_value}")

for rgs in peaks_rgs:
    axes[0].hlines(y=rgs, xmin=0, xmax=num_freq, colors='red', linestyles='--', linewidth=1)

axes[0].vlines(x=freq_data, ymin=0, ymax=num_rgt, colors='red', linestyles='--', linewidth=1.5)

# 表示
plt.show()
