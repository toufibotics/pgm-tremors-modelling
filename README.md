# pgm-tremors

An adaptive, hybrid PGM + GNN pipeline for real-time tremor detection and suppression in Parkinson’s Disease and Essential Tremor.

## Overview

This repo implements the full data-to-inference workflow for a magnetorheological-fluid exoskeleton prototype:

1. **Data ingestion & preprocessing** via DVC  
2. **Feature extraction** from 200 Hz IMU & sEMG streams  
3. **Baselines**: rule-based FFT threshold, LSTM classifier  
4. **PGM models**: static Bayesian Net, 2-state HMM  
5. **GNN**: Spatio-Temporal Graph Convolution (ST-GCN)  
6. **Hybrid**: convex fusion of ST-GCN + HMM  
7. **Evaluation**: F1, RMSE, latency, tremor suppression potential

## Installation

```bash
# 1. Clone code
git clone https://github.com/Laith309/pgm-tremors.git
cd pgm-tremors

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Fetch large data via DVC
dvc pull
```

Main files are in: 00_processing.ipynb and 01_modelling.ipynb with visualizations and core project.

Dataset DOI: https://doi.org/10.5061/dryad.fbg79cp1d

