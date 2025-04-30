"""
preproc.py  –  band-pass 0.5–20 Hz on each axis
reads data/raw/**.pkl   =>   writes data/interim_pre/*.npy
"""

import pathlib, pickle, numpy as np, scipy.signal as sg, tqdm

HP, LP, ORDER = 0.5, 20.0, 2
RAW  = pathlib.Path("data/raw")
DST = pathlib.Path("data/interim_pre")     
DST.mkdir(parents=True, exist_ok=True)

def butter_band(fs):
    return sg.butter(ORDER, [HP/fs*2, LP/fs*2], btype="band")

def process():
    pkls = list(RAW.rglob("*.pkl"))
    for fp in tqdm.tqdm(pkls, desc="preproc"):
        df  = pickle.loads(fp.read_bytes())
        sig = df[["Cal1","Cal2","Cal3","Cal4","Cal5","Cal6"]].to_numpy()
        fs  = 200.0
        b, a = butter_band(fs)
        sig  = sg.detrend(sig, axis=0)
        sig  = sg.filtfilt(b, a, sig, axis=0).astype(np.float32)

        # ── new: mirror subject directory ─────────────────────
        rel_dir  = fp.relative_to(RAW).parent        # e.g. PD004
        out_dir  = DST / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{fp.stem}.npy", sig)
    print(f"[preproc] {len(pkls)} files  →  {DST}")


if __name__ == "__main__":
    process()
