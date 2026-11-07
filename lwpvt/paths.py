# -*- coding: utf-8 -*-
"""Folder layout of the repository and the run-folder convention.

    <repo>/data/raw/ASHRAE_RP-1043/Chiller Data/   original RP-1043 .xls files (not redistributed)
    <repo>/data/processed/                          files written by the scripts in prepare/
    <repo>/output/<model>/<scenario>/s<seed>/       one folder per training run

The data and output folders can be moved with the environment variables LWPVT_DATA and LWPVT_OUTPUT.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("LWPVT_DATA", os.path.join(ROOT, "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw", "ASHRAE_RP-1043", "Chiller Data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
OUT_DIR = os.environ.get("LWPVT_OUTPUT", os.path.join(ROOT, "output"))

# The 15 extrapolation scenarios: split variable x split mode (column names of the split file).
SCENARIOS = [f"Q_{v}_{m}" for v in ("Qe", "TEO", "TCI")
             for m in ("q1_up", "half_up", "mid_out", "half_dn", "q3_dn")]
_VAR = {"Qe": "Q_e", "TEO": "T_chw,out", "TCI": "T_cw,in"}
_MODE = {"q1_up": "Q1 up", "half_up": "Half up", "mid_out": "Mid out", "half_dn": "Half down", "q3_dn": "Q3 down"}
SCENARIO_LABEL = {s: f"{_VAR[s.split('_')[1]]} {_MODE[s.split('_', 2)[2]]}" for s in SCENARIOS}

SEEDS = list(range(42, 52))          # ten trainings per scenario


def run_dir(model, scenario, seed):
    """Folder of one training run."""
    return os.path.join(OUT_DIR, model, scenario, f"s{seed}")


def batch_suffix(bs):
    """Folder suffix of a batch size: 0 = full batch."""
    return "_bfull" if bs == 0 else f"_b{bs:03d}"
