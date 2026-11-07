# -*- coding: utf-8 -*-
"""Step 3: the 15 extrapolation scenarios of the operating points (Section 4.2 of the paper).

Input   data/processed/rp1043_steady.csv
Output  data/processed/splits/splits_steady_normal.parquet   row_id x scenario, value in {train, val, test}
        data/processed/splits/splits_summary.csv             counts and definition of each scenario

For each split variable v in {Q_e, T_chw,out, T_cw,in} (quartiles Q1, Q2, Q3 of the valid normal
operating points):
    q1_up     train v <= Q1          test v > Q1
    half_up   train v <= Q2          test v > Q2
    mid_out   train Q1 <= v <= Q3    test v < Q1 or v > Q3
    half_dn   train v >= Q2          test v < Q2
    q3_dn     train v >= Q3          test v < Q3
15% of the training points are drawn at random as the validation set of a scenario (inside the training
range). The random seed of a scenario is 42 + index, where the index is the position of the scenario in
the original family list of the study (FAMILY_INDEX); this reproduces the split file used in the paper.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import PROC_DIR

OUT = os.path.join(PROC_DIR, "splits")
os.makedirs(OUT, exist_ok=True)

SEED = 42
VAL_FRAC = 0.15
TASK_VARS = ["Qe_kW", "TEO_C", "TCI_C"]
TARGET = "COP"
FAMILY_INDEX = {
    "Q_Qe_q1_up": 1, "Q_Qe_half_up": 2, "Q_Qe_half_dn": 3, "Q_Qe_mid_out": 4,
    "Q_TEO_q1_up": 5, "Q_TEO_half_up": 6, "Q_TEO_half_dn": 7, "Q_TEO_mid_out": 8,
    "Q_TCI_q1_up": 9, "Q_TCI_half_up": 10, "Q_TCI_half_dn": 11, "Q_TCI_mid_out": 12,
    "Q_Qe_q3_dn": 23, "Q_TEO_q3_dn": 24, "Q_TCI_q3_dn": 25,
}
VAR_OF = {"Qe": "Qe_kW", "TEO": "TEO_C", "TCI": "TCI_C"}


def carve_val(role, train_mask, rng):
    """Move VAL_FRAC of the training points to the validation set (the test region is untouched)."""
    idx = np.where(train_mask)[0]
    n_val = max(1, int(round(len(idx) * VAL_FRAC)))
    val_idx = rng.choice(idx, size=n_val, replace=False)
    role[val_idx] = "val"
    return role


def make_quantile(df, var, mode, rng):
    x = df[var].to_numpy()
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    if mode == "q1_up":
        tr, desc = x <= q1, f"train {var}<=Q1({q1:.2f}) -> test above"
    elif mode == "half_up":
        tr, desc = x <= q2, f"train {var}<=Q2({q2:.2f}) -> test above"
    elif mode == "half_dn":
        tr, desc = x >= q2, f"train {var}>=Q2({q2:.2f}) -> test below"
    elif mode == "mid_out":
        tr = (x >= q1) & (x <= q3)
        desc = f"train Q1({q1:.2f})<={var}<=Q3({q3:.2f}) -> test both sides"
    elif mode == "q3_dn":
        tr, desc = x >= q3, f"train {var}>=Q3({q3:.2f}) -> test below"
    else:
        raise ValueError(mode)
    role = np.where(tr, "train", "test").astype(object)
    role = carve_val(role, role == "train", rng)
    return role, desc


def main():
    steady = pd.read_csv(os.path.join(PROC_DIR, "rp1043_steady.csv"), encoding="utf-8-sig")
    df = steady[steady["fault_type"] == "normal"]
    df = df[df["is_valid"]].dropna(subset=TASK_VARS + [TARGET]).reset_index(drop=True)

    mat = pd.DataFrame({"row_id": df["row_id"].to_numpy()})
    summary = []
    for name, i in FAMILY_INDEX.items():
        _, v, mode = name.split("_", 2)
        rng = np.random.default_rng(SEED + i)
        role, desc = make_quantile(df, VAR_OF[v], mode, rng)
        mat[name] = role
        vc = pd.Series(role).value_counts()
        summary.append([name, int(vc.get("train", 0)), int(vc.get("val", 0)), int(vc.get("test", 0)), desc])
    mat.to_parquet(os.path.join(OUT, "splits_steady_normal.parquet"), index=False)
    sm = pd.DataFrame(summary, columns=["scenario", "n_train", "n_val", "n_test", "definition"])
    sm.to_csv(os.path.join(OUT, "splits_summary.csv"), index=False, encoding="utf-8-sig")
    print(f"operating points: {len(df)}, scenarios: {len(FAMILY_INDEX)}")
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
