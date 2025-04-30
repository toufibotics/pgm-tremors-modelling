"""
window.py – slide 256/128 windows over each trial
input : data/interim_pre/*.npy
output: data/processed/X_windows.npz  (X, subj)
"""

import numpy as np, pathlib, tqdm, re, json

WIN, HOP = 256, 128
SRC  = pathlib.Path("data/interim_pre")
OUT  = pathlib.Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
META = json.load(open("data/metadata.json"))
SUBJ_RE = re.compile(r"(CT|PD)\d{1,3}", re.I)

def sliding(arr):
    n = (arr.shape[0] - WIN)//HOP + 1
    s0,s1 = arr.strides
    return np.lib.stride_tricks.as_strided(arr,
            shape=(n,WIN,arr.shape[1]), strides=(HOP*s0,s0,s1))

def main():
    X_list, subj_list = [], []
    files = list(SRC.rglob("*.npy"))
    if not files:
        raise RuntimeError("No .npy files in data/interim_pre")

    for npy in tqdm.tqdm(files, desc="windows"):
        sig = np.load(npy)
        Xw  = sliding(sig)
        X_list.append(Xw)

        subj_id = npy.parent.name                 # now always PDxxx or CTxxx
        if not SUBJ_RE.fullmatch(subj_id):
            # skip unexpected folders (e.g., 'misc')
            print(f"[warn] skip {npy} (parent={subj_id})")
            continue
        subj_list.extend([subj_id]*len(Xw))

    if not X_list:
        raise RuntimeError("No valid windows found after filtering.")
    X = np.vstack(X_list).astype(np.float32)
    np.savez_compressed(OUT/"X_windows.npz", X=X, subj=np.array(subj_list))
    print("[window]", X.shape, "saved")


if __name__ == "__main__":
    main()
