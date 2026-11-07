# -*- coding: utf-8 -*-
"""Main table of the paper (Table 3): test MAPE (%) and R2 of the ten models in the 15 extrapolation
scenarios, the mean over the scenarios (Avg) and the mean rank over the scenarios (Rank).

Each cell is the mean over the ten seeds of a scenario. For a model trained with a grid (batch size or
the physics coefficient of PI-WGAN), the variant with the best mean rank of the MAPE over the 15
scenarios is selected (ties: lower mean MAPE), as described in Section 4.3 of the paper. Avg and Rank
are computed on the cell values as printed (MAPE rounded to 0.01, R2 to 0.001; MAPE ranked ascending,
R2 descending, ties share the average rank). A Friedman test and Wilcoxon signed-rank tests against
LW-PVT with Holm correction (Demsar, 2006) are printed for the unrounded MAPE.

Usage (from the repository root):  python results/main_table.py
Writes output/main_table_mape.csv, output/main_table_r2.csv, output/main_table_selection.csv
"""
import os
import sys
import json
import glob
import math

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import OUT_DIR, SCENARIOS, SCENARIO_LABEL

# model name of the paper -> folder patterns under output/ (all variants of the grid)
METHODS = [
    ("MP", ["MP"]),
    ("MLP", ["MLP_b*"]),
    ("ICNN-PGNN", ["ICNN-PGNN_b*"]),
    ("MonoPINN", ["MonoPINN_b*"]),
    ("TS-PINN", ["TS-PINN_b*"]),
    ("Carnot-PINN", ["Carnot-PINN_b*"]),
    ("E-ANN", ["E-ANN_b*"]),
    ("Hybrid TL", ["Hybrid-TL_b*"]),
    ("PI-WGAN", ["PI-WGAN_lam*"]),
    ("LW-PVT", ["LW-PVT"]),
]
PROPOSED = "LW-PVT"
SEED_GLOB = "s*"                      # run folders inside <model>/<scenario>/
# Nemenyi critical values q_0.05 for k = 2 ... 12 models
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def folders(out_dir, patterns):
    fs = set()
    for p in patterns:
        fs.update(os.path.basename(x) for x in glob.glob(os.path.join(out_dir, p)) if os.path.isdir(x))
    return sorted(fs)


def load_folder(out_dir, folder):
    """Per scenario: (mean test MAPE, mean test R2, number of seeds); None when a scenario is missing."""
    out = {}
    for sc in SCENARIOS:
        files = glob.glob(os.path.join(out_dir, folder, sc, SEED_GLOB, "metrics.json"))
        if not files:
            return None
        mapes, r2s = [], []
        for f in files:
            m = json.load(open(f, encoding="utf-8"))["metrics"]["test"]
            mapes.append(m["MAPE"])
            r2s.append(m.get("R2", float("nan")))
        out[sc] = (float(np.mean(mapes)), float(np.nanmean(r2s)), len(files))
    return out


def mean_rank(table, higher_is_better=False):
    """table: {name: {scenario: value}} -> {name: mean rank over the scenarios} (rank 1 = best)."""
    names = list(table)
    M = np.array([[table[n][sc] for n in names] for sc in SCENARIOS])
    R = np.array([stats.rankdata(-M[i] if higher_is_better else M[i]) for i in range(len(SCENARIOS))])
    return dict(zip(names, R.mean(0))), R


def build(out_dir, methods=METHODS):
    chosen, data, sel_rows = {}, {}, []
    for lab, pats in methods:
        tabs = {f: load_folder(out_dir, f) for f in folders(out_dir, pats)}
        tabs = {f: t for f, t in tabs.items() if t is not None}
        if not tabs:
            raise SystemExit(f"{lab}: no complete run folder matches {pats} under {out_dir}")
        if len(tabs) == 1:
            best = next(iter(tabs))
            ar = {best: float("nan")}
        else:
            ar, _ = mean_rank({f: {sc: t[sc][0] for sc in SCENARIOS} for f, t in tabs.items()})
            best = min(tabs, key=lambda f: (round(ar[f], 6), np.mean([tabs[f][sc][0] for sc in SCENARIOS])))
        chosen[lab], data[lab] = best, tabs[best]
        for f in tabs:
            sel_rows.append(dict(model=lab, variant=f,
                                 mean_rank_in_grid=(round(ar[f], 3) if not math.isnan(ar[f]) else ""),
                                 mean_mape=round(np.mean([tabs[f][sc][0] for sc in SCENARIOS]), 3),
                                 n_seeds=min(tabs[f][sc][2] for sc in SCENARIOS),
                                 selected=("yes" if f == best else "")))
    labs = [l for l, _ in methods]
    idx = [SCENARIO_LABEL[s] for s in SCENARIOS]
    MAPE = pd.DataFrame({l: [data[l][sc][0] for sc in SCENARIOS] for l in labs}, index=idx)
    R2 = pd.DataFrame({l: [data[l][sc][1] for sc in SCENARIOS] for l in labs}, index=idx)
    # ranks on the values as printed in the table (ties share the average rank)
    rank_m, RM = mean_rank({l: {sc: round(data[l][sc][0], 2) for sc in SCENARIOS} for l in labs})
    rank_r, _ = mean_rank({l: {sc: round(data[l][sc][1], 3) for sc in SCENARIOS} for l in labs}, higher_is_better=True)

    def with_summary(df, dec, ranks):
        t = df.round(dec)
        t.loc["Avg"] = t.mean().round(dec)
        t.loc["Rank"] = pd.Series({l: round(ranks[l], 2) for l in labs})
        return t

    T_mape = with_summary(MAPE, 2, rank_m)
    T_r2 = with_summary(R2, 3, rank_r)
    return T_mape, T_r2, pd.DataFrame(sel_rows), chosen, MAPE.values, labs, RM


def friedman_block(M, labs, proposed):
    """Friedman and Iman-Davenport tests over the models, Nemenyi critical difference, and Wilcoxon
    signed-rank tests of the proposed model against every other model with Holm correction."""
    k, N = M.shape[1], M.shape[0]
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_ff = 1 - stats.f.cdf(FF, k - 1, (k - 1) * (N - 1))
    R = np.array([stats.rankdata(M[i]) for i in range(N)])
    avg = R.mean(0)
    CD = Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
    pv = labs.index(proposed)
    print(f"\n[MAPE] Friedman chi2={chi2:.2f} p={p:.2e} | Iman-Davenport F={FF:.2f} p={p_ff:.2e} | Nemenyi CD={CD:.2f}")
    order = np.argsort(avg)
    print("  mean rank: " + ", ".join(f"{labs[j]} {avg[j]:.2f}" for j in order))
    ps = sorted((stats.wilcoxon(M[:, pv], M[:, j])[1], labs[j], int(np.sum(M[:, pv] < M[:, j]))) for j in range(k) if j != pv)
    m = len(ps)
    holm = []
    for i, (pw, l, wins) in enumerate(ps):
        ph = min(1.0, max((m - r) * ps[r][0] for r in range(i + 1)))
        holm.append((l, wins, pw, ph))
    print(f"  Wilcoxon + Holm vs {proposed}: " + ", ".join(f"{l} wins {w}/15 p={ph:.3f}{'*' if ph < 0.05 else ''}"
                                                           for l, w, pw, ph in holm))


if __name__ == "__main__":
    argv = sys.argv[1:]
    out_dir = argv[argv.index("--output") + 1] if "--output" in argv else OUT_DIR
    T_mape, T_r2, sel, chosen, M, labs, _ = build(out_dir)
    print("[selected variants] " + ", ".join(f"{l}: {chosen[l]}" for l in labs))
    print("\n[test MAPE (%)]")
    print(T_mape.to_string())
    print("\n[test R2]")
    print(T_r2.to_string())
    friedman_block(M, labs, PROPOSED)
    T_mape.to_csv(os.path.join(out_dir, "main_table_mape.csv"), encoding="utf-8-sig")
    T_r2.to_csv(os.path.join(out_dir, "main_table_r2.csv"), encoding="utf-8-sig")
    sel.to_csv(os.path.join(out_dir, "main_table_selection.csv"), index=False, encoding="utf-8-sig")
    print(f"\nwritten: {out_dir}/main_table_mape.csv, main_table_r2.csv, main_table_selection.csv")
