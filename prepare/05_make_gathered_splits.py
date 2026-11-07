# -*- coding: utf-8 -*-
"""Step 5: propagate the scenario roles of the operating points to the gathered rows.

Every gathered row takes the role (train / val / test) of its operating point (anchor_id) in each of
the 15 scenarios, so the rows of one operating point are never split between training and test.
Input   data/processed/rp1043_steady_normal.csv, data/processed/splits/splits_steady_normal.parquet,
        data/processed/gathered_normal/steady_normal_gathered01.csv
Output  data/processed/splits/splits_gathered_normal_1.parquet
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import PROC_DIR

SPL = os.path.join(PROC_DIR, "splits")
TOLS = [1]

sn = pd.read_csv(os.path.join(PROC_DIR, "rp1043_steady_normal.csv"), encoding="utf-8-sig")   # row order = anchor_id
spn = pd.read_parquet(os.path.join(SPL, "splits_steady_normal.parquet"))
QCOLS = [c for c in spn.columns if c.startswith("Q_")]
sn_roles = sn[["row_id"]].merge(spn, on="row_id", how="left")

print(f"scenarios: {len(QCOLS)}\n  {QCOLS}\n")
for tol in TOLS:
    aug = pd.read_csv(os.path.join(PROC_DIR, "gathered_normal", f"steady_normal_gathered{tol:02d}.csv"), encoding="utf-8-sig")
    aid = aug["anchor_id"].values
    out = pd.DataFrame({"row_id": aug["row_id"].values})
    for sc in QCOLS:
        arole = sn_roles[sc].astype(object).values
        rrole = np.array([arole[a] for a in aid], dtype=object)
        out[sc] = np.array([r if r in ("train", "val", "test") else "" for r in rrole], dtype=object)
    fn = os.path.join(SPL, f"splits_gathered_normal_{tol}.parquet")
    out.to_parquet(fn, index=False)
    sc0 = QCOLS[0]
    vc = pd.Series(out[sc0]).value_counts().to_dict()
    print(f"+/-{tol}%  rows={len(out):,}  [{sc0}] train={vc.get('train', 0):,} val={vc.get('val', 0):,} "
          f"test={vc.get('test', 0):,}  -> {fn}")
