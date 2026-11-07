# -*- coding: utf-8 -*-
"""MP: multivariate quadratic polynomial regression (10 coefficients) fitted by least squares to the
training rows of each scenario (standardised inputs). The fit is deterministic, so one run per
scenario is written under seed 42.

Usage (from the repository root):  python run/mp.py
Output: output/MP/<scenario>/s42/metrics.json, best_model.pth, pred_test.csv
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import FEATURES, TARGET, load_rows
from lwpvt.baselines import mp_design
from lwpvt.metrics import regression_metrics
from lwpvt.paths import SCENARIOS, run_dir
from lwpvt.saveart import save_artifacts, json_save_retry

mapes = []
for sc in SCENARIOS:
    d = load_rows(sc)
    tr, va, te = (d[d["role"] == r] for r in ("train", "val", "test"))
    mu, sd = tr[FEATURES].values.mean(0), tr[FEATURES].values.std(0) + 1e-8
    A = mp_design((tr[FEATURES].values - mu) / sd)
    beta, *_ = np.linalg.lstsq(A, tr[TARGET].values, rcond=None)

    def pred(D):
        return mp_design((D[FEATURES].values - mu) / sd) @ beta

    res = {"test": regression_metrics(te[TARGET].values.reshape(-1, 1), pred(te).reshape(-1, 1)),
           "val": regression_metrics(va[TARGET].values.reshape(-1, 1), pred(va).reshape(-1, 1))}
    od = run_dir("MP", sc, 42)
    save_artifacts(od, X_test=te[FEATURES].values, y_true=te[TARGET].values, y_pred=pred(te),
                   state=None, extra={"beta": beta, "x_mean": mu, "x_std": sd},
                   X_valid=va[FEATURES].values, y_valid=va[TARGET].values, p_valid=pred(va))
    json_save_retry(dict(scenario=sc, model="MP", method="MP (quadratic polynomial, 10 coefficients)", seed=42,
                         n=dict(train=len(tr), val=len(va), test=len(te)),
                         beta=[float(b) for b in beta], metrics=res),
                    os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
    mapes.append(res["test"]["MAPE"])
    print(f"[{sc}] test MAPE {res['test']['MAPE']:.2f}%  R2 {res['test']['R2']:.3f}", flush=True)
print(f"MP: mean test MAPE over 15 scenarios {np.mean(mapes):.2f}% | worst {max(mapes):.2f}%")
