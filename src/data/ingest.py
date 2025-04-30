"""
ingest.py  –  Pick only LOWER LEFT IMU sheets from the PD-IMU dataset and
store each trial as data/raw/{subject}/{activity}.pkl
Also writes data/metadata.json.
"""

import pandas as pd, pathlib, pickle, json, re, tqdm

# ---------- configured ---------------------------------
LOWER_RE = re.compile(r"^Lower\s*Left$", re.I)   # only *Lower Left* sheet
RAW_DIR  = pathlib.Path("data/raw");     RAW_DIR.mkdir(parents=True, exist_ok=True)
META_FP  = pathlib.Path("data/metadata.json")
# ----------------------------------------------------------

def ingest_xls(dataset_root: pathlib.Path):
    meta = []
    dataset_root = pathlib.Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    for subj_path in tqdm.tqdm(sorted(dataset_root.iterdir()), desc="subjects"):
        if not subj_path.is_dir(): 
            continue
        subj_id   = subj_path.name                  # CT003  or PD012
        pd_label  = 1 if subj_id.upper().startswith("PD") else 0

        for xls in subj_path.glob("*.xls*"):
            activity = xls.stem.replace(" ", "_")   # "Cardigan_1"
            # choose the correct engine
            if xls.suffix.lower() == ".xls":
                df_dict = pd.read_excel(xls, sheet_name=None, engine="xlrd")   # ← binary XLS
            else:
                df_dict = pd.read_excel(xls, sheet_name=None, engine="openpyxl") # find Lower Left sheet
            
            # read all sheets quickly
            try:
                sheet_name, sheet_df = next(
                    (n, d) for n, d in df_dict.items() if LOWER_RE.match(n.strip())
                )
            except StopIteration:
                print(f"[warn] no Lower Left in {xls.name}")
                continue

            # dump to pickle
            out_dir = RAW_DIR / subj_id
            out_dir.mkdir(exist_ok=True)
            out_fp  = out_dir / f"{activity}.pkl"
            with out_fp.open("wb") as f:
                pickle.dump(sheet_df, f)

            meta.append(dict(subject=subj_id, pd=int(pd_label),
                             activity=activity, sheet=sheet_name,
                             pkl=str(out_fp)))

    META_FP.write_text(json.dumps(meta, indent=2))
    print(f"[ingest] {len(meta)} trials  →  {RAW_DIR}")

# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="path to PD_IMU_XLS/Data")
    args = ap.parse_args()
    ingest_xls(pathlib.Path(args.dataset))
