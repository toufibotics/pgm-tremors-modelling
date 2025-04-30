"""
Time-domain IMU feature extraction.

Exports
-------
time_features(X) -> (feat_matrix, feat_names)

X : np.ndarray  shape (N, win, D)
    Sliding windows of IMU data.

Returned shape: (N, D * 4)   # RMS, VAR, ZCR, JERK per axis
"""

import numpy as np

def _rms(x, axis=-1):
    return np.sqrt(np.mean(np.square(x), axis=axis))

def _var(x, axis=-1):
    return np.var(x, axis=axis)

def _zcr(x, axis=-1):
    # sign changes / (len-1)
    s = np.sign(x)
    zc = np.sum(s[..., 1:] * s[..., :-1] < 0, axis=axis)
    return zc / (x.shape[axis] - 1)

def _jerk(x, axis=-1):
    # mean absolute diff (first‐order derivative magnitude)
    diff = np.diff(x, axis=axis)
    return np.mean(np.abs(diff), axis=axis)

def time_features(X):
    """
    Parameters
    ----------
    X : (N, win, D)

    Returns
    -------
    F  : (N, D*4)   in order [axis0_RMS, axis0_VAR, axis0_ZCR, axis0_JERK, axis1_RMS, ...]
    names : list[str]
    """
    feats = []
    names = []
    for d in range(X.shape[2]):
        sig = X[..., d]
        feats.extend([
            _rms(sig), _var(sig), _zcr(sig), _jerk(sig)
        ])
        names.extend([
            f"axis{d}_rms", f"axis{d}_var", f"axis{d}_zcr", f"axis{d}_jerk"
        ])
    F = np.stack(feats, axis=-1).reshape(X.shape[0], -1)
    return F.astype(np.float32), names
