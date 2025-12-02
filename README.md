# **pgm-tremors-modelling**

**Hybrid ST-GCN/HMM Tremor-State Detector for a Wearable MR-Fluid Exoskeletons**

> End-to-end modelling pipeline for low-latency tremor detection using **Spatio-Temporal Graph Convolutional Networks (ST-GCN)** fused with a **2-state Hidden Markov Model (HMM)**.
> Designed for safe, real-time deployment in a **battery-powered MR-fluid tremor-suppression orthosis**.
>
> *(Corresponds to the IEEE BSN 2025 submission by Jrab)*

---

## **Paper & External Resources**

* [**IEEE BSN 2025 Paper (PDF)**](https://github.com/toufibotics/pgm-tremors-modelling/blob/main/BSN%202025%20-%20Full%20Paper.pdf)

* [**IEEE BSN 2025 Poster (PDF)**](https://github.com/toufibotics/pgm-tremors-modelling/blob/main/BSN%202025%20poster.pdf)

* [**IEEE BSN 2025 Awardee Talk (PDF)**](https://github.com/toufibotics/pgm-tremors-modelling/blob/main/BSN25-Toufic-Jrab-Awardee.pdf)

* [**Initial Project Report - COMP 588 (PDF)**](https://github.com/toufibotics/pgm-tremors-modelling/blob/main/Initial%20Research%20Report%20-%20COMP%20588.pdf)

* [**Dataset DOI**](https://datadryad.org/dataset/doi:10.5061/dryad.fbg79cp1d)

---

## **Overview**

This repository contains the complete workflow described in the BSN 2025 paper, including preprocessing, baseline models, probabilistic graphical models, the lightweight ST-GCN encoder, and the hybrid fusion with an HMM back-end.

The modelling approach is grounded in the safety constraints of wearable robotic tremor-suppression systems:

* **<80 ms total control-loop budget**
* **Physiological dwell time (~100 ms) for tremor onset/offset**
* **Posteriors must be calibrated** to avoid accidental 15 Nm damper stiffening
* **Low-energy real-time inference** on Jetson-class embedded hardware

The final hybrid model achieves:

* **AUC = 0.70 ± 0.01** on held-out subjects (ADL data)
* **Sub-15 ms projected inference** (INT8, Jetson Nano CPU)
* **Significant calibration improvement** and reduction of spurious spikes
* **Best NLL** among all tested detectors


---

## **Pipeline Summary**

### **1. Signal Processing**

* 200 Hz **6-channel IMU** (lower-arm)
* 0.5–20 Hz band-pass filtering
* 1.28 s windows (256 samples, 50% overlap)
* Channel-wise ℓ2 normalization
* Dataset: **34 subjects × 3 ADLs**, ~8k labeled windows


### **2. Baselines**

| Class             | Models                                 |
| ----------------- | -------------------------------------- |
| **Rule-based**    | Welch PSD dominant-frequency threshold |
| **Probabilistic** | Bayesian Network (D, RMS features)     |
| **Temporal PGM**  | BN-driven HMM                          |
| **Neural**        | Two-layer LSTM                         |
| **Graph-based**   | ST-GCN encoder (22k params)            |

### **3. Proposed Model**

#### **ST-GCN Encoder**

* 6-node spatial graph
* Intra-modal + cross-axis edges
* Temporal kernel (k = 3)
* 3 residual GCN blocks
* Global average pooling → 32-D embedding
* Fully connected classifier head


#### **Bayesian Network (BN)**

* Dominant frequency & RMS
* Multinomial–Dirichlet closed-form posterior
* Robust under low-SNR ADL motions


#### **HMM Fusion Back-End**

* 2-state HMM (tremor vs voluntary motion)
* Encodes onset/offset probabilities
* Enforces a **100 ms physiological dwell-time**
* Forward recursion in **0.05 ms**
* Fusion via ( \ell_t = θ p_t + (1−θ) o_t ), ( θ=0.6 )


---

## **Key Results (Held-Out Subjects)**

| Model             | Prec    | Rec | F1  | AUC      | Latency (ms) |
| ----------------- | ------- | --- | --- | -------- | ------------ |
| Rule              | .26     | .49 | .34 | .55      | 0.02         |
| Bayesian Net      | .25     | .79 | .38 | .62      | 0.03         |
| LSTM              | .33     | .42 | .37 | .64      | 21.6         |
| ST-GCN            | .37     | .65 | .47 | .68      | 16.6         |
| HMM-BN            | .28     | .98 | .43 | .38      | 0.05         |
| **Hybrid (Ours)** | **.41** | .32 | .36 | **0.70** | **15.2**     |
|                   |         |     |     |          |              |

**Highlights**
✔ Best **AUC** and **NLL**
✔ Large improvement in **posterior calibration**
✔ Major reduction in **false high-confidence spikes** (≈3× fewer)
✔ Meets all real-time constraints for safe exoskeleton control


---

## **Repository Structure**

```
pgm-tremors-modelling/
│
├── notebooks/
│   ├── 00_processing.ipynb       # Preprocessing, feature engineering, EDA
│   └── 01_modelling.ipynb        # Baselines, ST-GCN, BN, HMM, hybrid fusion
│
├── src/
│   ├── models/                   # ST-GCN, BN, HMM classes
│   ├── utils/                    # Filtering, windowing, feature extraction
│   └── viz/                      # Reproducible figures & plots
│
├── data/                         # DVC-managed dataset (not stored in repo)
└── requirements.txt
```

---

## **Installation & Reproduction**

### 1. Clone

```bash
git clone https://github.com/toufibotics/pgm-tremors-modelling.git
cd pgm-tremors-modelling
```

### 2. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull Dataset (DVC)

```bash
dvc pull
```

### 4. Run Notebooks

Start with:

* **`00_processing.ipynb`** — data ingestion, windowing, filtering
* **`01_modelling.ipynb`** — all models + hybrid fusion + metrics

> **Note**: Update notebook paths if executing from outside repo root.

---

## **Research Contributions**

**Important to note that this project was envisioned within McGill's COMP 588 (Probabilistic Graphical Models), taught by Dr. Siamak Ravanbakhsh, whose inputs and lectures were instrumental to making this project a reality.** 

This repository implements all components of the BSN 2025 study, contributing:

1. **First reported fusion of ST-GCN with probabilistic temporal smoothing**
   for wearable tremor suppression under ADLs.
2. **Hardware-aware design** achieving sub-15 ms projected inference.
3. **Calibration-focused pipeline** enabling safe MR-damper triggering.
4. **Open-source, fully reproducible preprocessing** for the five-sensor Dryad dataset.


---

## **Contact**

**Toufic Jrab**

toufic.jrab@mail.mcgill.ca



