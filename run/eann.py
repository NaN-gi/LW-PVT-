# -*- coding: utf-8 -*-
"""E-ANN (Zhang et al., 2025, Journal of Building Engineering 111): a neural network trained on the
measured training rows merged with synthetic data generated from an empirical performance model.

Method as in its paper: synthetic inputs are drawn uniformly with the split variable over the test
range and the other inputs over the training range (5,800 samples per scenario, drawn once with seed
42), labelled by the empirical model, merged with the training rows without weighting, and used to
train an ANN 3-37-35-49-1 (ReLU, Adam lr 0.01, 55 epochs, checkpoint with the lowest validation RMSE).
Because RP-1043 provides no manufacturer performance data, the labels come from the Gordon-Ng
regression fitted to the training rows (the same model that labels the virtual samples of LW-PVT);
synthetic samples with a label outside (0.5, 12) are rejected. The batch size follows the grid of
run/nn_batch_grid.py; the paper's own value is 32.

Usage (from the repository root):
    python run/eann.py [--bs 0,8,16,32,64,128,256] [--seeds 10] [--scenarios ...] [--vol 5800] [--dry]
Selected setting of the paper: --bs 0 (full batch).
Output: output/E-ANN_b<bs>/<scenario>/s<seed>/ (full batch: E-ANN_bfull)
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

N_VOL = 5800                          # number of synthetic samples (optimum reported in the paper)
EPOCHS, LR = 55, 0.01


def gn_teacher(tr):
    """GN regression fitted to the training rows; returns (beta, training CVRMSE in %)."""
    X, y = tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy(float)
    beta = fit_gn(X, y)["beta"]
    cv = float(np.sqrt(np.mean((gn_predict(beta, X) - y) ** 2)) / y.mean() * 100)
    return beta, cv


def generate_synthetic(beta, tr, te, axis, n, rng):
    """Uniform inputs (split variable: test range, other inputs: training range) with GN labels."""
    rngs = []
    for j, c in enumerate(FEATURES):
        src = te if j == axis else tr
        rngs.append((float(src[c].min()), float(src[c].max())))
    X, y = [], []
    while sum(len(x) for x in X) < n:
        m = n * 2
        qe = rng.uniform(*rngs[0], m)
        teo = rng.uniform(*rngs[1], m)
        tci = rng.uniform(*rngs[2], m)
        cop = gn_predict(beta, np.column_stack([qe, teo, tci]))
        ok = (cop > 0.5) & (cop < 12.0)
        X.append(np.column_stack([qe, teo, tci])[ok])
        y.append(cop[ok])
        if not ok.any():
            raise RuntimeError("no valid synthetic sample; check the input ranges")
    X, y = np.vstack(X)[:n], np.concatenate(y)[:n]
    return X.astype(np.float32), y.astype(np.float32)


def train_ann(Xm, ym, Xva, yva, Xte, yte, seed, dev, batch=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xm, xs = Xm.mean(0), Xm.std(0) + 1e-8
    ym_, ys = ym.mean(), ym.std() + 1e-8
    Xt = torch.tensor((Xm - xm) / xs, device=dev)
    yt = torch.tensor(((ym - ym_) / ys).reshape(-1, 1), device=dev)
    net = nn.Sequential(nn.Linear(3, 37), nn.ReLU(), nn.Linear(37, 35), nn.ReLU(),
                        nn.Linear(35, 49), nn.ReLU(), nn.Linear(49, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(Xt)
    best_v, best_state = float("inf"), None

    def pred(Xr):
        net.eval()
        with torch.no_grad():
            z = net(torch.tensor((Xr - xm) / xs, device=dev)).cpu().numpy().ravel()
        return z * ys + ym_

    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        bsz = n if batch == 0 else batch                  # 0 = full batch
        for i in range(0, n, bsz):
            b = perm[i:i + bsz]
            opt.zero_grad()
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        v = regression_metrics(yva, pred(Xva))["RMSE"]
        if v < best_v - 1e-9:
            best_v, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    pva, pte = pred(Xva), pred(Xte)
    art = dict(state=best_state, extra={"x_mean": xm, "x_std": xs, "y_mean": float(ym_), "y_std": float(ys)},
               X_test=Xte, y_true=yte, y_pred=pte, X_valid=Xva, y_valid=yva, p_valid=pva)
    return {"val": regression_metrics(yva, pva), "test": regression_metrics(yte, pte)}, art


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = "--dry" in argv
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
    use_sc = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
    vol = int(argv[argv.index("--vol") + 1]) if "--vol" in argv else N_VOL
    bss = [int(x) for x in argv[argv.index("--bs") + 1].split(",")] if "--bs" in argv else [0, 8, 16, 32, 64, 128, 256]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:n_seeds]
    print(f"E-ANN: {len(use_sc)} scenarios x {len(seeds)} seeds x batch {bss} | {vol} synthetic samples "
          f"with GN labels | device={dev}", flush=True)
    t0 = time.time()
    for sc in use_sc:
        df = load_rows(sc)
        tr, va, te = (df[df["role"] == r] for r in ("train", "val", "test"))
        axis = {"Qe": 0, "TEO": 1, "TCI": 2}[sc.split("_")[1]]
        beta, cvr = gn_teacher(tr)
        print(f"[{sc}] GN label model: training CVRMSE {cvr:.2f}%  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        if not dry:
            rng = np.random.default_rng(42)
            Xc, yc = generate_synthetic(beta, tr, te, axis, vol, rng)
            Xm = np.vstack([tr[FEATURES].to_numpy(np.float32), Xc])
            ym = np.concatenate([tr[TARGET].to_numpy(np.float32), yc])
            Xva_, yva_ = va[FEATURES].to_numpy(np.float32), va[TARGET].to_numpy(float)
            Xte_, yte_ = te[FEATURES].to_numpy(np.float32), te[TARGET].to_numpy(float)
        for bs in bss:
            name = "E-ANN" + batch_suffix(bs)
            for sd in seeds:
                od = run_dir(name, sc, sd)
                if run_done(od):
                    continue
                if dry:
                    print(f"    (dry) {name} seed {sd}")
                    continue
                m, art = train_ann(Xm, ym, Xva_, yva_, Xte_, yte_, sd, dev, batch=bs)
                save_artifacts(od, **art)
                json_save_retry(dict(scenario=sc, model=name, method="E-ANN (Zhang et al., 2025) with GN labels", seed=sd,
                                     settings=dict(hidden=[37, 35, 49], lr=LR, epochs=EPOCHS, batch=bs, n_synthetic=vol,
                                                   label_model="GN regression fitted to the training rows",
                                                   label_model_train_cvrmse_pct=round(cvr, 3),
                                                   synthetic_cop_range=[float(yc.min()), float(yc.max())]),
                                     metrics=m),
                                os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
            if not dry:
                print(f"    {name}: {len(seeds)} seeds done  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
