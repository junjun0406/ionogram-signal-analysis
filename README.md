# Ionogram Signal Analysis

Python tools for **O/X mode separation**, **peak detection**, and **inter-antenna phase analysis** of ionogram data.

This repository provides a simplified public version of my ionosonde signal-processing pipeline developed during undergraduate research in radio science and ionospheric analysis.

---

## ✨ Features

* O/X mode separation from complex antenna signals
* Histogram-based noise reduction
* Peak detection along range-gate profiles
* Inter-antenna phase difference visualization
* ROI-based histogram analysis
* Command-line interface (CLI)

---

## 📁 Repository Structure

```
mode_separation.py
    Core pipeline for:
    - loading .sav ionogram data
    - O/X mode separation
    - noise reduction
    - visualization

mode_peaks.py
    Peak detection example using separated O/X modes.

ionogram_phase_analysis.py
    Phase-difference analysis between antennas.
    Includes:
        - ROI masking
        - confidence threshold mask
        - vertical stripe noise detection
```

---

## 🔧 Requirements

Python 3.10+

```
pip install numpy scipy matplotlib
```

---

## 🚀 Usage

### 1️⃣ O/X Mode Separation

```
python mode_separation.py --file path/to/ionogram.sav
```

This will:

* Load ionogram data
* Perform mode separation
* Apply noise reduction
* Plot O-mode and X-mode ionograms

---

### 2️⃣ Peak Detection Example

```
python mode_peaks.py --file path/to/ionogram.sav --freq 8.0
```

Detect peaks along a selected frequency slice.

Parameters:

* `--freq` : Target frequency in MHz

---

### 3️⃣ Phase Difference Analysis

```
python ionogram_phase_analysis.py --file path/to/ionogram.sav
```

This tool generates:

* Phase difference ionogram
* Histogram analysis inside ROI

Optional parameters:

```
--thr_conf 700
--stripe_ratio 0.3
--phase_rgt 100 500
--phase_freq 1.0 15.0
```

---

## 📊 Data Format

Input files must be ionogram `.sav` files containing:

```
pulse_i
pulse_q
sct.frequency
```

---

## 🧠 Background

Ionosonde signals often exhibit elliptical polarization due to geomagnetic effects, which causes imperfect O/X mode separation.

This repository demonstrates:

* Complex signal combination
* Phase-rotation-based mode separation
* Histogram-based noise estimation
* Phase difference analysis between orthogonal antennas

---

## ⚠️ Notes

This is a **public research-friendly version**.

The original private research code includes:

* timestamp extraction
* batch processing
* advanced calibration routines

which are intentionally omitted here.

---

## 👤 Author

Junsei Fujita
Electrical & Electronic Engineering → Informatics (Network Optimization)

Research interests:

* Signal processing
* Network optimization
* Infrastructure & cloud engineering

---
