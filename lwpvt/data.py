# -*- coding: utf-8 -*-
"""Data of one extrapolation scenario.

The scripts in prepare/ write
  data/processed/gathered_normal/steady_normal_gathered01.csv   the 35,641 data (rows) used in the paper
  data/processed/splits/splits_gathered_normal_1.parquet        role (train / val / test) of every row in
                                                                 each of the 15 scenarios
load_rows() merges the two files for one scenario; load_split() additionally standardises the inputs
and the output with the statistics of the training rows and returns tensors.

Inputs  X = (Qe_kW, TEO_C, TCI_C) = (Q_e, T_chw,out, T_cw,in);  output y = COP.
"""
import os
import numpy as np
import pandas as pd
import torch

from .paths import PROC_DIR

FEATURES = ["Qe_kW", "TEO_C", "TCI_C"]     # Q_e (kW), T_chw,out (degC), T_cw,in (degC)
TARGET = "COP"
ROWS_CSV = os.path.join(PROC_DIR, "gathered_normal", "steady_normal_gathered01.csv")
SPLIT_PARQUET = os.path.join(PROC_DIR, "splits", "splits_gathered_normal_1.parquet")

# Sign constraints used by the physics penalties of LW-PVT and by the comparison models that take a
# monotonic constraint: +1 increasing, -1 decreasing, 0 no constraint on that input.
# COP decreases with T_cw,in. No constraint is placed on T_chw,out (see Section 3.3 of the paper).
MONO = np.array([0.0, 0.0, -1.0], np.float32)
# Curvature constraint: COP is concave in Q_e (-1); no curvature constraint on the temperatures.
CURV = np.array([-1.0, 0.0, 0.0], np.float32)


class Scaler:
    """z-score standardisation with the statistics of the training rows; inverse() maps back."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def load_rows(scenario):
    """Rows of one scenario with a 'role' column (train / val / test)."""
    base = pd.read_csv(ROWS_CSV, encoding="utf-8-sig")
    spl = pd.read_parquet(SPLIT_PARQUET)
    if scenario not in spl.columns:
        raise ValueError(f"unknown scenario '{scenario}'; available: {[c for c in spl.columns if c != 'row_id']}")
    df = base.merge(spl[["row_id", scenario]], on="row_id", how="inner").rename(columns={scenario: "role"})
    df = df[df["role"].isin(["train", "val", "test"])]
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def load_split(scenario, device="cpu"):
    """Standardised tensors of one scenario.

    Returns a dict with the training tensors (Xtr, ytr), the validation and test inputs (Xva, Xte) with
    their outputs in COP units (yva_raw, yte_raw), the scalers, the input box of all rows in
    standardised coordinates (domain_std; used for the collocation points and the virtual samples), the
    sign vectors (mono, curv) and the row counts.
    """
    df = load_rows(scenario)
    X = df[FEATURES].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32).reshape(-1, 1)
    role = df["role"].to_numpy()
    tr, va, te = role == "train", role == "val", role == "test"
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"train ({int(tr.sum())}) or test ({int(te.sum())}) rows are missing")

    xs = Scaler(X[tr].mean(0), X[tr].std(0) + 1e-8)        # training rows only
    ys = Scaler(y[tr].mean(0), y[tr].std(0) + 1e-8)

    def to_t(a):
        return torch.tensor(np.asarray(a, np.float32), device=device)

    # Operating box: min/max of the inputs of all rows of the scenario (input positions only, no labels).
    domain_raw = np.stack([X.min(0), X.max(0)])
    domain_std = xs.transform(domain_raw).astype(np.float32)

    return {
        "Xtr": to_t(xs.transform(X[tr])), "ytr": to_t(ys.transform(y[tr])),
        "Xva": to_t(xs.transform(X[va])) if va.sum() > 0 else None,
        "yva_raw": y[va],
        "Xte": to_t(xs.transform(X[te])), "yte_raw": y[te],
        "x_scaler": xs, "y_scaler": ys,
        "domain_std": domain_std, "mono": MONO.copy(), "curv": CURV.copy(),
        "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
    }
