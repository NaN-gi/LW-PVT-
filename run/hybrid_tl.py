# -*- coding: utf-8 -*-
"""Hybrid TL (Kim et al., 2025, Energy and Buildings 341): a neural network pretrained on synthetic
data from an empirical model and fine-tuned on the measured training rows.

Method as in its paper: synthetic COP data are generated on the operating box (5,000 uniform points),
their level is calibrated with K = sum(COP_measured) / sum(COP_model) over the training inputs, an ANN
3-32-32-32-1 (ReLU, dropout 0.2) is pretrained on them (500 epochs, Adam lr 0.001) and then fine-tuned
on the training rows with the first layers frozen and only the last layer trained (100 epochs,
lr 0.0001). The checkpoint with the lowest validation error is kept in each stage. Because RP-1043
provides no manufacturer performance points, the empirical model is the Gordon-Ng regression fitted
to the training rows (the same model that labels the virtual samples of LW-PVT). The batch size
follows the grid of run/nn_batch_grid.py.

Usage (from the repository root):
    python run/hybrid_tl.py [--bs 0,8,16,32,64,128,256] [--seeds 10] [--scenarios ...] [--nsyn 5000] [--dry] [--smoke]
Selected setting of the paper: --bs 32.
Output: output/Hybrid-TL_b<bs>/<scenario>/s<seed>/ (full batch: Hybrid-TL_bfull)
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import FEATURES, TARGET, load_rows
from lwpvt.gn import fit_gn, gn_predict
from lwpvt.metrics import regression_metrics
from lwpvt.paths import SCENARIOS, SEEDS, run_dir, batch_suffix
from lwpvt.saveart import save_artifacts, run_done, json_save_retry

HID, DROP = 32, 0.2
EP_PRE, LR_PRE, EP_FT, LR_FT = 500, 1e-3, 100, 1e-4


def ann():
    return nn.Sequential(nn.Linear(3, HID), nn.ReLU(), nn.Dropout(DROP),
                         nn.Linear(HID, HID), nn.ReLU(), nn.Dropout(DROP),
                         nn.Linear(HID, HID), nn.ReLU(), nn.Dropout(DROP),
                         nn.Linear(HID, 1))


def gn_teacher(tr):
    """GN regression fitted to the training rows; returns (beta, training CVRMSE in %)."""
    X, y = tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy(float)
    beta = fit_gn(X, y)["beta"]
    cv = float(np.sqrt(np.mean((gn_predict(beta, X) - y) ** 2)) / y.mean() * 100)
    return beta, cv


def synth_box(beta, df_all, n, rng):
    """Uniform samples in the operating box (min/max of every input over all rows) with GN labels."""
    rngs = [(float(df_all[c].min()), float(df_all[c].max())) for c in FEATURES]
    X, y = [], []
    while sum(len(x) for x in X) < n:
        m = n * 2
        qe, teo, tci = (rng.uniform(*rngs[j], m) for j in range(3))
        cop = gn_predict(beta, np.column_stack([qe, teo, tci]))
        ok = (cop > 0.5) & (cop < 12.0)
        if not ok.any():
            raise RuntimeError("no valid synthetic sample; check the input ranges")
        X.append(np.column_stack([qe, teo, tci])[ok])
        y.append(cop[ok])
    return np.vstack(X)[:n].astype(np.float32), np.concatenate(y)[:n].astype(np.float32)


def fit(net, X, y, Xva, yva, epochs, lr, batch, dev, params=None):
    """MSE training of all parameters (params=None) or of the given parameters; returns the state with
    the lowest validation MSE."""
    opt = torch.optim.Adam(params if params is not None else net.parameters(), lr=lr)
    n = len(X)
    best_v, best_state = float("inf"), None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        bsz = n if batch == 0 else batch
        for i in range(0, n, bsz):
            b = perm[i:i + bsz]
            opt.zero_grad()
            loss = ((net(X[b]) - y[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xva) - yva) ** 2).mean())
        if v < best_v - 1e-12:
            best_v, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def run_one(sc, seed, dev, beta, df_all, nsyn, batch):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = load_rows(sc)
    tr, va, te = (df[df["role"] == r] for r in ("train", "val", "test"))
    Xtr, ytr = tr[FEATURES].to_numpy(np.float32), tr[TARGET].to_numpy(np.float32)
    Xva, yva = va[FEATURES].to_numpy(np.float32), va[TARGET].to_numpy(float)
    Xte, yte = te[FEATURES].to_numpy(np.float32), te[TARGET].to_numpy(float)
    # synthetic data (per-seed generator) and level calibration K on the training inputs
    rng = np.random.default_rng(seed)
    Xs, ys = synth_box(beta, df_all, nsyn, rng)
    cop_tr = gn_predict(beta, Xtr.astype(float))
    K = float(ytr.sum() / cop_tr.sum())
    ys = ys * K
    # standardisation with the synthetic and the measured inputs (shared by both stages)
    xm, xsd = np.vstack([Xs, Xtr]).mean(0), np.vstack([Xs, Xtr]).std(0) + 1e-8
    ym, ysd = float(ys.mean()), float(ys.std() + 1e-8)
    T = lambda X: torch.tensor(((X - xm) / xsd).astype(np.float32), device=dev)
    Ty = lambda y: torch.tensor(((y - ym) / ysd).reshape(-1, 1).astype(np.float32), device=dev)
    net = ann().to(dev)
    # pretraining on the synthetic data (validation rows monitor the checkpoint)
    net = fit(net, T(Xs), Ty(ys), T(Xva), Ty(yva), EP_PRE, LR_PRE, batch, dev)
    # fine-tuning of the last layer only
    for p in net.parameters():
        p.requires_grad_(False)
    last = net[-1]
    for p in last.parameters():
        p.requires_grad_(True)
    net = fit(net, T(Xtr), Ty(ytr), T(Xva), Ty(yva), EP_FT, LR_FT, batch, dev, params=list(last.parameters()))
    net.eval()

    def pred(X):
        with torch.no_grad():
            return net(T(X)).cpu().numpy().ravel() * ysd + ym
    pva, pte = pred(Xva), pred(Xte)
    art = dict(state=net.state_dict(), extra={"x_mean": xm, "x_std": xsd, "y_mean": ym, "y_std": ysd, "K": K},
               X_test=Xte, y_true=yte, y_pred=pte, X_valid=Xva, y_valid=yva, p_valid=pva)
    return {"val": regression_metrics(yva, pva), "test": regression_metrics(yte, pte)}, K, art


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = "--dry" in argv
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
    use_sc = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
    nsyn = int(argv[argv.index("--nsyn") + 1]) if "--nsyn" in argv else 5000
    bss = [int(x) for x in argv[argv.index("--bs") + 1].split(",")] if "--bs" in argv else [0, 8, 16, 32, 64, 128, 256]
    if "--smoke" in argv:                              # short run for testing the pipeline
        EP_PRE, EP_FT = 20, 5
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:n_seeds]
    df_all = load_rows(SCENARIOS[0])                   # operating box = all rows, independent of the role
    jobs = [(bs, sc, sd) for bs in bss for sc in use_sc for sd in seeds]
    print(f"Hybrid TL: batch {bss} x {len(use_sc)} scenarios x {len(seeds)} seeds = {len(jobs)} runs | "
          f"{nsyn} synthetic points, pretraining {EP_PRE} epochs (lr {LR_PRE}), fine-tuning {EP_FT} epochs "
          f"(lr {LR_FT}, last layer) | device={dev}", flush=True)
    t0 = time.time()
    teachers = {}
    for i, (bs, sc, sd) in enumerate(jobs, 1):
        name = "Hybrid-TL" + batch_suffix(bs)
        od = run_dir(name, sc, sd)
        if run_done(od):
            continue
        print(f"[{i}/{len(jobs)}] {sc} {name} seed {sd}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        if dry:
            continue
        if sc not in teachers:                         # one GN fit per scenario (training rows only)
            tr = load_rows(sc)
            teachers[sc] = gn_teacher(tr[tr["role"] == "train"])
            print(f"    GN label model: training CVRMSE {teachers[sc][1]:.2f}%", flush=True)
        beta, cvr = teachers[sc]
        m, K, art = run_one(sc, sd, dev, beta, df_all, nsyn, bs)
        save_artifacts(od, **art)
        json_save_retry(dict(scenario=sc, model=name, method="Hybrid TL (Kim et al., 2025) with GN labels", seed=sd,
                             settings=dict(hidden=HID, dropout=DROP, ep_pre=EP_PRE, lr_pre=LR_PRE, ep_ft=EP_FT, lr_ft=LR_FT,
                                           frozen_layers=2, finetuned_layers=1, batch=bs, n_synthetic=nsyn,
                                           label_model="GN regression fitted to the training rows",
                                           label_model_train_cvrmse_pct=round(cvr, 3), K_calibration=K),
                             metrics=m),
                        os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
        print(f"    test MAPE {m['test']['MAPE']:.2f}%  R2 {m['test']['R2']:.3f}  (K={K:.3f})", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
