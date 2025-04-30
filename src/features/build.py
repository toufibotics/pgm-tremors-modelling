"""
build.py - runs through time & frequency features code files.

Usage
-----
python -m features.build [--sr 200]

Outputs
-------
data/processed/features.npz
    X : (N, F)  feature matrix
    y : labels  (if present in X_windows.npz)
    names : string array of feature names
"""

import argparse, numpy as np, pathlib, json
from . import imu_time, imu_freq

WIN_PATH = pathlib.Path("data/processed/X_windows.npz")
OUT_FP   = pathlib.Path("data/processed/features.npz")

def load_windows():
    d = np.load(WIN_PATH, allow_pickle=True)
    X = d["X"]              # (N, win, D)
    meta = {k:d[k] for k in d.files if k != "X"}  # subj, maybe y
    return X, meta

def main(fs):
    Xwin, meta = load_windows()
    Xt, name_t = imu_time.time_features(Xwin)
    Xf, name_f = imu_freq.freq_features(Xwin, fs)
    Xfeat = np.concatenate([Xt, Xf], axis=1)
    names = np.array(name_t + name_f)
    save_dict = {"X": Xfeat, "names": names}
    save_dict.update(meta)      # copy subj / label arrays
    OUT_FP.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_FP, **save_dict)
    print(f"[features] {Xfeat.shape} → {OUT_FP}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=float, default=200.0,
                    help="sampling rate Hz (default 200)")
    args = ap.parse_args()
    main(args.sr)
