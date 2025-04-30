"""
Frequency-domain IMU features.

Exports
-------
freq_features(X, fs=200.0) -> (feat_matrix, feat_names)
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def _dominant_freq(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=len(sig))
    idx = np.argmax(Pxx, axis=-1)
    dom_freq = f[idx]
    dom_mag  = Pxx[np.arange(sig.shape[0]), idx]
    return dom_freq, dom_mag

def _median_freq(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=len(sig))
    cumsum = np.cumsum(Pxx, axis=-1)
    med_idx = (cumsum >= cumsum[..., -1:] / 2).argmax(axis=-1)
    return f[med_idx]

def _spec_entropy(sig, fs):
    f, Pxx = welch(sig, fs=fs, nperseg=len(sig))
    Pxx_norm = Pxx / (Pxx.sum(axis=-1, keepdims=True) + 1e-12)
    return entropy(Pxx_norm, base=2, axis=-1)

def freq_features(X, fs=200.0):
    """
    X : (N, win, D)
    Returns (N, D*4)
    """
    N, win, D = X.shape
    feats = []
    names = []
    for d in range(D):
        sig = X[..., d]
        dom_f, dom_mag = _dominant_freq(sig, fs)
        med_f          = _median_freq(sig, fs)
        spec_ent       = _spec_entropy(sig, fs)
        feats.extend([dom_f, dom_mag, med_f, spec_ent])
        names.extend([f"axis{d}_domF", f"axis{d}_domMag",
                      f"axis{d}_medF", f"axis{d}_specEnt"])
    F = np.stack(feats, axis=-1).reshape(N, -1)
    return F.astype(np.float32), names
