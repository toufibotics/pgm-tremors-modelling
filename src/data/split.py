"""
split.py – subject-wise GroupKFold 5×
label comes from metadata.json (pd field)
"""

import numpy as np, json, pathlib
from sklearn.model_selection import GroupKFold

PROCESSED = pathlib.Path("data/processed")
META = {m["subject"]: m["pd"] for m in json.load(open("data/metadata.json"))}

def main(folds=5, fold_idx=0):
    d = np.load(PROCESSED/"X_windows.npz")
    X, subj = d["X"], d["subj"]
    y = np.array([META[s] for s in subj], dtype=np.int8)

    unique_groups = np.unique(subj)
    n_splits = min(folds, len(unique_groups))       
    if n_splits < 2:
        raise ValueError("Need at least 2 unique subjects; got "
                         f"{len(unique_groups)}")

    gkf = GroupKFold(n_splits=n_splits)            
    train_idx, test_idx = next(gkf.split(X, y, groups=subj))

    inner = GroupKFold(n_splits=min(5, n_splits))
    tr_i, val_i = next(inner.split(X[train_idx], y[train_idx],
                                   groups=subj[train_idx]))
    SPLIT_DIR = PROCESSED / "split"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True) 

    np.savez(SPLIT_DIR/"train.npz", X=X[train_idx][tr_i], y=y[train_idx][tr_i])
    np.savez(SPLIT_DIR/"val.npz",   X=X[train_idx][val_i], y=y[train_idx][val_i])
    np.savez(SPLIT_DIR/"test.npz",  X=X[test_idx],         y=y[test_idx])
    print("[split] done:",
          {k: np.load(SPLIT_DIR/f"{k}.npz")["X"].shape[0]
           for k in ("train","val","test")})


if __name__ == "__main__":
    main()
