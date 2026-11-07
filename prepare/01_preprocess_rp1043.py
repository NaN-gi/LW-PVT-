# -*- coding: utf-8 -*-
"""Step 1: pool the ASHRAE RP-1043 test files.

Input   data/raw/ASHRAE_RP-1043/Chiller Data/<fault folder>/*.xls  (plus "read me first.xls")
Output  data/processed/rp1043_steady.csv         "Steady States" sheets of all files (27 operating
                                                 conditions per file)
        data/processed/rp1043_complete.parquet   "Complete Data Set" sheets (10-second time series)
        data/processed/preprocess_report.txt     parsing report
Columns added: file metadata (source_file, fault_type, severity, file_flag, test_date), SI units
(Qe_kW, TEO_C, TCI_C, TEI_C, TCO_C) and a validity flag is_valid (kW > 5, 0.5 < COP < 12, Evap Tons > 1).
"""
import os
import re
import sys
import glob
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import RAW_DIR, PROC_DIR

os.makedirs(PROC_DIR, exist_ok=True)

FAULT_MAP = {                                    # folder name -> fault code
    "Benchmark Tests": "normal",
    "Condenser fouling": "cf",
    "Reduced condenser water flow": "fwc",
    "Reduced evaporator water flow": "fwe",
    "Refrigerant leak": "rl",
    "Refrigerant overcharge": "ro",
    "Excess oil": "eo",
    "Non-condensables in refrigerant": "nc",
    "Defective Pilot Valve": "dpv",
    "Multiple faults": "multi",
}
TON_TO_KW = 3.51685                              # 1 refrigeration ton = 3.51685 kW


def f_to_c(s):
    return (pd.to_numeric(s, errors="coerce") - 32.0) * 5.0 / 9.0


def load_readme_dates():
    """file name -> test date, from 'read me first.xls'."""
    path = os.path.join(RAW_DIR, "read me first.xls")
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)
    df.columns = ["file_name", "test_date", "comments"]
    df["key"] = df["file_name"].astype(str).str.strip().str.lower()
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    return dict(zip(df["key"], df["test_date"]))


def find_sheet(xl, keyword):
    for s in xl.sheet_names:
        if keyword.lower() in s.lower():
            return s
    return None


def parse_meta(path):
    folder = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    fault = FAULT_MAP.get(folder, "unknown")
    m = re.search(r"(\d+)", stem)                # severity = number at the end of the file name
    severity = int(m.group(1)) if (m and fault != "normal") else np.nan
    flags = []
    low = stem.lower()
    for kw in ["alt", "aborted", "unsteady", "modified", "trace"]:
        if kw in low:
            flags.append(kw)
    return dict(source_file=stem, fault_type=fault, severity=severity,
                file_flag="|".join(flags) if flags else "")


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def add_si_columns(df):
    if "Evap Tons" in df.columns:
        df["Qe_kW"] = pd.to_numeric(df["Evap Tons"], errors="coerce") * TON_TO_KW
    for src, dst in [("TEO", "TEO_C"), ("TCI", "TCI_C"), ("TEI", "TEI_C"), ("TCO", "TCO_C")]:
        if src in df.columns:
            df[dst] = f_to_c(df[src])
    if "COP" in df.columns:
        df["COP"] = pd.to_numeric(df["COP"], errors="coerce")
    return df


def quality_flags(df):
    """Row validity flag (rows are kept; the flag is used downstream)."""
    kw = pd.to_numeric(df.get("kW"), errors="coerce")
    cop = pd.to_numeric(df.get("COP"), errors="coerce")
    qe = pd.to_numeric(df.get("Evap Tons"), errors="coerce")
    ok = (
        kw.notna() & (kw > 5.0)                       # chiller off / no load
        & cop.notna() & (cop > 0.5) & (cop < 12.0)    # non-physical COP
        & qe.notna() & (qe > 1.0)
    )
    df["is_valid"] = ok
    return df


def main():
    dates = load_readme_dates()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.xls"), recursive=True))
    files = [f for f in files if "read me" not in os.path.basename(f).lower()]
    if not files:
        raise SystemExit(f"no .xls file found under {RAW_DIR}")

    steady_parts, complete_parts, report = [], [], []
    for i, path in enumerate(files, 1):
        meta = parse_meta(path)
        key = meta["source_file"].lower()
        tdate = dates.get(key)
        if tdate is None:                              # retry without a suffix such as 'alt'
            key2 = re.sub(r"\s*(alt|--.*)$", "", key).strip()
            tdate = dates.get(key2, pd.NaT)
        meta["test_date"] = tdate

        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            report.append(f"[FAIL-OPEN] {meta['source_file']}: {e}")
            continue

        s_name = find_sheet(xl, "steady")
        if s_name:
            try:
                df = normalize_columns(xl.parse(s_name, header=0))
                if "COP" not in df.columns:
                    raise ValueError(f"no COP column (cols={list(df.columns)[:8]}...)")
                df = add_si_columns(df)
                df = quality_flags(df)
                for k, v in meta.items():
                    df[k] = v
                steady_parts.append(df)
            except Exception as e:
                report.append(f"[FAIL-STEADY] {meta['source_file']}: {e}")
        else:
            report.append(f"[NO-STEADY-SHEET] {meta['source_file']}: sheets={xl.sheet_names}")

        c_name = find_sheet(xl, "complete")
        if c_name:
            try:
                df = normalize_columns(xl.parse(c_name, header=0))
                df = add_si_columns(df)
                df = quality_flags(df)
                df["row_seq"] = np.arange(len(df))     # time order inside the file
                for k, v in meta.items():
                    df[k] = v
                complete_parts.append(df)
            except Exception as e:
                report.append(f"[FAIL-COMPLETE] {meta['source_file']}: {e}")
        else:
            report.append(f"[NO-COMPLETE-SHEET] {meta['source_file']}: sheets={xl.sheet_names}")

        print(f"({i}/{len(files)}) {meta['source_file']}", flush=True)

    steady = pd.concat(steady_parts, ignore_index=True)
    steady.insert(0, "row_id", np.arange(len(steady)))
    steady.to_csv(os.path.join(PROC_DIR, "rp1043_steady.csv"), index=False, encoding="utf-8-sig")

    complete = pd.concat(complete_parts, ignore_index=True)
    complete.insert(0, "row_id", np.arange(len(complete)))
    complete.to_parquet(os.path.join(PROC_DIR, "rp1043_complete.parquet"), index=False)

    lines = [f"files: {len(files)}",
             f"steady   : {len(steady):,} rows (valid {int(steady['is_valid'].sum()):,})",
             f"complete : {len(complete):,} rows (valid {int(complete['is_valid'].sum()):,})",
             "", "[steady rows per fault type]", steady.groupby("fault_type").size().to_string(),
             "", "[key variables (steady, valid rows)]"]
    sv = steady[steady["is_valid"]]
    for c in ["Qe_kW", "TEO_C", "TCI_C", "COP"]:
        q = sv[c].quantile([0, 0.25, 0.5, 0.75, 1.0]).round(2).tolist()
        lines.append(f"  {c:7s} min/Q1/med/Q3/max = {q}")
    lines += ["", "[warnings]"] + (report if report else ["(none)"])
    txt = "\n".join(lines)
    with open(os.path.join(PROC_DIR, "preprocess_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
