# -*- coding: utf-8 -*-
"""LW-PVT (proposed method): train one network per scenario and seed.

Loss in standardised COP units (Sections 3.1-3.4 of the paper):
    L = L_data + lambda1 * L_virtual + L_phys
    L_data    = mean_i (f(x_i) - y_i)^2                      training rows
    L_virtual = sum_j p_j (f(x_j) - y_j)^2                   N virtual samples with GN labels y_j and
                                                              inverse-variance shares p_j (sum = 1)
    L_phys    = L_mono + L_curv + L_bound                    512 collocation points redrawn every epoch
Protocol: MLP 3-128-128-128-1 (SiLU), full-batch Adam (lr 0.001, 1,000 epochs), checkpoint with the
lowest validation RMSE, ten seeds (42-51) per scenario. Only the training rows and the operating box
are used to build the virtual samples.

Usage (from the repository root):
    python run/lwpvt.py [--seeds 10] [--scenarios Q_Qe_q1_up,...] [--epochs 1000] [--N 1024]
                        [--lambda1 0.0005] [--dry]
Output: output/LW-PVT/<scenario>/s<seed>/  metrics.json, best_model.pth, pred_test.csv, virtual_samples.csv
A run whose metrics.json exists is skipped, so an interrupted grid can be resumed.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import FEATURES, TARGET, MONO, CURV, Scaler, load_rows
from lwpvt.model import MLP
from lwpvt.physics import sample_collocation, physics_penalty, cop_physics_bounds
from lwpvt.gn import sample_virtual_inputs
from lwpvt.weights import label_pack
from lwpvt.metrics import regression_metrics
from lwpvt.paths import SCENARIOS, SEEDS, run_dir
from lwpvt.saveart import run_done, torch_save_retry, csv_save_retry, json_save_retry

N_VIRTUAL = 1024          # number of virtual samples
LAMBDA1 = 0.0005          # weight of the virtual-sample loss (total weight W = lambda1 * n_train)
EPOCHS = 1000
N_COLLOC = 512            # collocation points per epoch for the physics loss
LR = 1e-3
MIN_DIST = 0.15           # exclusion distance of the virtual samples from the training rows (std units)
HIDDEN = (128, 128, 128)


def run_one(scenario, seed, N, lambda1, epochs, dev, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = load_rows(scenario)
    X_all = df[FEATURES].to_numpy(np.float32)                 # operating box = min/max of all rows
    tr, va, te = (df[df["role"] == r] for r in ("train", "val", "test"))
    Xtr_raw = tr[FEATURES].to_numpy(np.float32)
    ytr_raw = tr[TARGET].to_numpy(np.float32)
    xs = Scaler(Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-8)
    ys = Scaler(np.array([ytr_raw.mean()], np.float32), np.array([ytr_raw.std() + 1e-8], np.float32))
    domain_std = xs.transform(np.stack([X_all.min(0), X_all.max(0)])).astype(np.float32)

    # -- virtual samples: positions, GN labels, uncertainties and weight shares (training rows only)
    T_std = np.asarray(sample_virtual_inputs(xs.transform(Xtr_raw).astype(np.float32), domain_std, N, seed,
                                             min_dist=MIN_DIST, n_cand=max(20000, 20 * N)))
    T_raw = xs.inverse(T_std)
    pack = label_pack(Xtr_raw, ytr_raw, T_raw)
    n_train = len(Xtr_raw)
    W = lambda1 * n_train
    if verbose:
        print(f"  virtual samples {len(T_std)} | s_train={pack['s_train']:.4f} "
              f"| sigma_win median {np.median(pack['sigma_win']):.3f}, max {pack['sigma_win'].max():.3f} "
              f"| W = lambda1 * n_train = {lambda1:g} * {n_train} = {W:.2f}", flush=True)

    Xtr = torch.tensor(xs.transform(Xtr_raw), device=dev)
    ytr = torch.tensor(ys.transform(ytr_raw.reshape(-1, 1)), device=dev)
    Tt = torch.tensor(T_std, device=dev)
    labt = torch.tensor(ys.transform(pack["labels"].reshape(-1, 1)).astype(np.float32), device=dev)
    pt = torch.tensor(pack["p"].astype(np.float32).reshape(-1, 1), device=dev)

    model = MLP(in_dim=3, hidden=HIDDEN).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_v, best_state = float("inf"), None
    Xva = torch.tensor(xs.transform(va[FEATURES].to_numpy(np.float32)), device=dev)
    yva_raw = va[TARGET].to_numpy(float)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = ((model(Xtr) - ytr) ** 2).mean()                                   # L_data
        loss = loss + (W / n_train) * (pt * (model(Tt) - labt) ** 2).sum()        # lambda1 * L_virtual
        xc = sample_collocation(domain_std, N_COLLOC, dev)
        l_mono, l_curv = physics_penalty(model, xc, MONO, CURV)
        loss = loss + l_mono + l_curv + cop_physics_bounds(xc, model(xc), xs, ys)  # L_phys
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = ys.inverse(model(Xva).cpu().numpy()).ravel()
        vr = float(np.sqrt(np.mean((pv - yva_raw) ** 2)))
        if vr < best_v - 1e-9:
            best_v = vr
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)

    out, preds = {}, {}
    for name, D in [("val", va), ("test", te)]:
        Xr = torch.tensor(xs.transform(D[FEATURES].to_numpy(np.float32)), device=dev)
        model.eval()
        with torch.no_grad():
            preds[name] = ys.inverse(model(Xr).cpu().numpy()).ravel()
        out[name] = regression_metrics(D[TARGET].to_numpy(float), preds[name])
    art = dict(state=best_state, x_mean=xs.mean, x_std=xs.std, y_mean=float(ys.mean[0]), y_std=float(ys.std[0]),
               T_raw=T_raw, labels=pack["labels"], sigma=pack["sigma"], p=pack["p"],
               Xte=te[FEATURES].to_numpy(np.float32), yte=te[TARGET].to_numpy(float), pte=preds["test"])
    diag = dict(n_train=int(n_train), n_val=int(len(va)), n_test=int(len(te)), N=int(len(T_std)),
                lambda1=lambda1, W=float(W), s_train=pack["s_train"],
                sigma_win_median=float(np.median(pack["sigma_win"])),
                label_min=float(pack["labels"].min()), label_max=float(pack["labels"].max()),
                mono_signs=MONO.tolist(), curv_signs=CURV.tolist(), n_colloc=N_COLLOC, lr=LR, epochs=epochs,
                hidden=list(HIDDEN), best_val_rmse=best_v)
    return out, diag, art


def save_run(od, art):
    os.makedirs(od, exist_ok=True)
    torch_save_retry({"state_dict": art["state"], "hidden": list(HIDDEN), "features": FEATURES,
                      "x_mean": np.asarray(art["x_mean"], np.float32).tolist(),
                      "x_std": np.asarray(art["x_std"], np.float32).tolist(),
                      "y_mean": art["y_mean"], "y_std": art["y_std"]}, os.path.join(od, "best_model.pth"))
    csv_save_retry(pd.DataFrame({**{c: art["Xte"][:, i] for i, c in enumerate(FEATURES)},
                                 "y_true": art["yte"], "y_pred": art["pte"]}),
                   os.path.join(od, "pred_test.csv"), encoding="utf-8-sig")
    csv_save_retry(pd.DataFrame({**{c: art["T_raw"][:, i] for i, c in enumerate(FEATURES)},
                                 "label": art["labels"], "sigma": art["sigma"], "p": art["p"]}),
                   os.path.join(od, "virtual_samples.csv"), encoding="utf-8-sig")


def _fmt(v):
    return ("%.10f" % v).rstrip("0").rstrip(".").replace(".", "p")


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = "--dry" in argv
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
    use_sc = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else EPOCHS
    N = int(argv[argv.index("--N") + 1]) if "--N" in argv else N_VIRTUAL
    lambda1 = float(argv[argv.index("--lambda1") + 1]) if "--lambda1" in argv else LAMBDA1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:n_seeds]
    folder = "LW-PVT" + ("" if N == N_VIRTUAL else f"_N{N}") + ("" if lambda1 == LAMBDA1 else f"_lambda1_{_fmt(lambda1)}")
    print(f"LW-PVT: {len(use_sc)} scenarios x {len(seeds)} seeds | N={N} lambda1={lambda1:g} "
          f"| full batch {epochs} epochs | device={dev} | output/{folder}", flush=True)
    t0 = time.time()
    for sc in use_sc:
        first = True
        for sd in seeds:
            od = run_dir(folder, sc, sd)
            if run_done(od):
                continue
            print(f"[{sc}] seed {sd}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
            if dry:
                continue
            m, diag, art = run_one(sc, sd, N, lambda1, epochs, dev, verbose=first)
            first = False
            save_run(od, art)
            json_save_retry(dict(scenario=sc, model=folder, method="LW-PVT", seed=sd, settings=diag, metrics=m),
                            os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
            print(f"    test MAPE {m['test']['MAPE']:.2f}%  R2 {m['test']['R2']:.3f}", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
