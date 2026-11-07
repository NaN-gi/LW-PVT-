# -*- coding: utf-8 -*-
"""Step 4: gather the time-series data around each operating point (Section 4.1 of the paper).

For every normal operating point (anchor), the rows of the normal-operation time series whose three
inputs are all within +/-1% of the anchor are collected. The anchors themselves are included.
Input   data/processed/rp1043_steady_normal.csv, data/processed/rp1043_complete.parquet
Output  data/processed/gathered_normal/steady_normal_gathered01.csv
        columns: row_id, source_file, row_seq, fault_type, Qe_kW, TEO_C, TCI_C, TEI_C, TCO_C, kW, COP,
                 anchor_id (index of the operating point), is_anchor (1 = the operating point itself)
The study also produced the tolerances 3-50% (TOLS); only +/-1% is used in the paper.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import PROC_DIR

OUTDIR = os.path.join(PROC_DIR, "gathered_normal")
os.makedirs(OUTDIR, exist_ok=True)

TOLS = [1]                                                    # +/- percent
INPUTS = ["Qe_kW", "TEO_C", "TCI_C"]
KEEP = ["row_id", "source_file", "row_seq", "fault_type",
        "Qe_kW", "TEO_C", "TCI_C", "TEI_C", "TCO_C", "kW", "COP"]

sn = pd.read_csv(os.path.join(PROC_DIR, "rp1043_steady_normal.csv"), encoding="utf-8-sig")
c = pd.read_parquet(os.path.join(PROC_DIR, "rp1043_complete.parquet"))
iv = c["is_valid"]
if iv.dtype == object:
    iv = iv.astype(str).str.lower().isin(["true", "1"])
cn = c[(c["fault_type"] == "normal") & iv.astype(bool)].copy()

# anchor rows are copied from the operating-point table; their row_id is 'a<n>' so that it cannot
# collide with the integer row_id of the time-series table
anch = sn.copy()
anch["anchor_id"] = np.arange(len(sn))
anch["is_anchor"] = 1
anch["row_id"] = ["a" + str(i) for i in range(len(sn))]
cand = cn.copy()

# nearest anchor of every time-series row (inputs scaled by their standard deviation)
A = sn[INPUTS].values.astype(float)
sd = A.std(0)
tree = cKDTree(A / sd)
Rc = cand[INPUTS].values.astype(float)
_, nn = tree.query(Rc / sd, k=1)
rel = np.abs(Rc - A[nn]) / np.abs(A[nn])                      # relative distance per input

summary = []
for pct in TOLS:
    tol = pct / 100.0
    inbox = np.all(rel <= tol, axis=1)                        # all three inputs within +/-tol
    box = cand[inbox].copy()
    box["anchor_id"] = nn[inbox]
    box["is_anchor"] = 0
    out = pd.concat([anch, box], ignore_index=True)
    cols = [k for k in KEEP if k in out.columns] + ["anchor_id", "is_anchor"]
    out = out[cols].sort_values(["anchor_id", "is_anchor"]).reset_index(drop=True)
    fn = os.path.join(OUTDIR, f"steady_normal_gathered{pct:02d}.csv")
    out.to_csv(fn, index=False, encoding="utf-8-sig")
    if len(box):
        dev = (np.abs(box["COP"].values - sn["COP"].values[box["anchor_id"].values])
               / np.abs(sn["COP"].values[box["anchor_id"].values]))
        cop_dev = np.median(dev) * 100
    else:
        cop_dev = 0.0
    summary.append((pct, len(out), len(box), round(len(box) / len(sn), 1), round(cop_dev, 2)))

print(f"{len(sn)} operating points + gathered rows -> {OUTDIR}\n")
print(f"{'+/-%':>4} {'rows':>8} {'gathered':>8} {'per point':>9} {'COP dev %':>9}  file")
for pct, tot, nb, per, dev in summary:
    print(f"{pct:>4} {tot:>8,} {nb:>8,} {per:>9} {dev:>9}  steady_normal_gathered{pct:02d}.csv")
