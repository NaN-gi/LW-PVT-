# -*- coding: utf-8 -*-
"""Step 2: keep the normal-operation operating points.

Input   data/processed/rp1043_steady.csv
Output  data/processed/rp1043_steady_normal.csv   the 486 operating points of the benchmark tests
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import PROC_DIR

steady = pd.read_csv(os.path.join(PROC_DIR, "rp1043_steady.csv"), encoding="utf-8-sig")
normal = steady[steady["fault_type"] == "normal"].reset_index(drop=True)
out = os.path.join(PROC_DIR, "rp1043_steady_normal.csv")
normal.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(normal)} normal operating points -> {out}")
