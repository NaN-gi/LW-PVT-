# -*- coding: utf-8 -*-
"""MonoPINN (Tang et al., 2026, Energy and Buildings 359): monotonic penalty on the training samples
with feature weights and a penalty coefficient tuned by a genetic algorithm (GA).

Method as in its paper: network 3-256-256-256-1 (ReLU), Adam (lr 0.003), up to 2,000 epochs, penalty
lambda * sum_i w_i * mean ReLU(-s_i dCOP/dx_i) on the training batch, (w, lambda) found by a GA
(10 individuals, 6 generations, tournament selection, elitism 2, arithmetic crossover, mutation).
Settings not given in the paper: the GA fitness is the validation MAPE of a shortened training (at most
300 epochs, early stopping with patience 30, batch 64); the final trainings use the batch-size grid of
run/nn_batch_grid.py with early stopping (patience 50). With the sign vector of this task only
T_cw,in carries a constraint, so the GA searches lambda (w = 1 on T_cw,in). The GA runs once per
scenario (seed 42) and its result is stored in output/MonoPINN/<scenario>/ga_weights.json.

Usage (from the repository root):
    python run/monopinn.py [--final_bs 0,8,16,32,64,128,256] [--seeds 10] [--scenarios ...]
                           [--search_ep 300] [--final_ep 2000] [--patience 50] [--search_patience 30] [--dry]
Selected setting of the paper: --final_bs 0 (full batch).
Output: output/MonoPINN_b<bs>/<scenario>/s<seed>/ (full batch: MonoPINN_bfull)
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import load_split
from lwpvt.metrics import regression_metrics
from lwpvt.paths import SCENARIOS, SEEDS, OUT_DIR, run_dir, batch_suffix
from lwpvt.saveart import save_artifacts, run_done, json_save_retry

LR, BATCH = 0.003, 64                     # GA search uses batch 64
POP, GEN, ELITE, TOUR = 10, 6, 2, 3
LAM_LO, LAM_HI = 0.5, 50.0


def build_net(dev):
    return nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)).to(dev)


def mono_pen_per_feature(model, X, active_signs):
    x = X.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(model(x).sum(), x, create_graph=True)[0]
    return {j: torch.relu(-float(s) * g[:, j]).mean() for j, s in active_signs.items()}


def train_once(data, w_map, lam, epochs, seed, dev, patience=0, batch=BATCH):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_net(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    Xtr, ytr = data["Xtr"], data["ytr"]
    ys = data["y_scaler"]
    n = len(Xtr)
    bs = n if batch <= 0 else batch                       # 0 = full batch
    best_v, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = ((net(Xtr[b]) - ytr[b]) ** 2).mean()
            pens = mono_pen_per_feature(net, Xtr[b], w_map["signs"])
            loss = loss + lam * sum(w_map["w"][j] * p for j, p in pens.items())
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = ys.inverse(net(data["Xva"]).cpu().numpy()).ravel()
        v = float(np.mean(np.abs((pv - data["yva_raw"].ravel()) / data["yva_raw"].ravel())) * 100)
        if v < best_v - 1e-6:
            best_v = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            wait = 0
        elif patience:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net, best_v


def metrics_of(net, data, preds):
    ys = data["y_scaler"]
    out = {}
    for name, X, yr in [("val", data["Xva"], data["yva_raw"].ravel()),
                        ("test", data["Xte"], data["yte_raw"].ravel())]:
        net.eval()
        with torch.no_grad():
            preds[name] = ys.inverse(net(X).cpu().numpy()).ravel()
        out[name] = regression_metrics(yr, preds[name])
    return out


def ga_search(data, active, search_ep, dev, log, patience=30):
    rng = np.random.default_rng(42)
    k = len(active)
    signs = {j: float(data["mono"][j]) for j in active}

    def indiv(w, lam):
        return {"w": {j: float(wi) for j, wi in zip(active, w)}, "lam": float(lam)}

    pop = [indiv(np.ones(k) / k, 10.5)]                      # reference point of the paper
    while len(pop) < POP:
        pop.append(indiv(rng.dirichlet(np.ones(k)),
                         float(np.exp(rng.uniform(np.log(LAM_LO), np.log(LAM_HI))))))
    cache = {}

    def fit(ind):
        key = tuple(round(ind["w"][j], 4) for j in active) + (round(ind["lam"], 3),)
        if key not in cache:
            _, v = train_once(data, {"signs": signs, "w": ind["w"]}, ind["lam"],
                              search_ep, 42, dev, patience=patience)
            cache[key] = v
        return cache[key]

    for g in range(GEN):
        scored = sorted(pop, key=fit)
        log(f"    GA generation {g + 1}/{GEN}: best val MAPE {fit(scored[0]):.3f} "
            f"(w={[round(scored[0]['w'][j], 3) for j in active]}, lambda={scored[0]['lam']:.2f})")
        if g == GEN - 1:
            return scored[0], fit(scored[0])
        nxt = scored[:ELITE]
        while len(nxt) < POP:
            a, b = (min(rng.choice(len(pop), TOUR, replace=False), key=lambda i: fit(pop[i]))
                    for _ in range(2))
            pa, pb = pop[a], pop[b]
            t = rng.uniform()
            w = np.array([t * pa["w"][j] + (1 - t) * pb["w"][j] for j in active])
            lam = float(np.exp(t * np.log(pa["lam"]) + (1 - t) * np.log(pb["lam"])))
            if rng.uniform() < 0.3:                          # mutation
                w = rng.dirichlet(w / max(w.sum(), 1e-9) * 20 + 0.5)
                lam = float(np.clip(lam * np.exp(rng.normal(0, 0.3)), LAM_LO, LAM_HI))
            w = np.clip(w, 1e-4, None)
            nxt.append(indiv(w / w.sum(), lam))
        pop = nxt


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = "--dry" in argv
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
    use_sc = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
    search_ep = int(argv[argv.index("--search_ep") + 1]) if "--search_ep" in argv else 300
    final_ep = int(argv[argv.index("--final_ep") + 1]) if "--final_ep" in argv else 2000
    pat = int(argv[argv.index("--patience") + 1]) if "--patience" in argv else 50
    s_pat = int(argv[argv.index("--search_patience") + 1]) if "--search_patience" in argv else 30
    fbs = ([int(x) for x in argv[argv.index("--final_bs") + 1].split(",")]
           if "--final_bs" in argv else [0, 8, 16, 32, 64, 128, 256])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:n_seeds]
    print(f"MonoPINN: {len(use_sc)} scenarios | GA ({POP} individuals x {GEN} generations, <= {search_ep} epochs, "
          f"patience {s_pat}, batch 64) + final {len(seeds)} seeds x batch {fbs} x <= {final_ep} epochs, "
          f"patience {pat} | device={dev}", flush=True)
    t0 = time.time()
    for sc in use_sc:
        gw_path = os.path.join(OUT_DIR, "MonoPINN", sc, "ga_weights.json")
        data = None
        if os.path.exists(gw_path):
            best = json.load(open(gw_path, encoding="utf-8"))
        else:
            print(f"[{sc}] GA search  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
            if dry:
                continue
            data = load_split(sc, device=dev)
            active = [j for j, s in enumerate(data["mono"]) if s != 0]
            best_ind, best_fit = ga_search(data, active, search_ep, dev, lambda m: print(m, flush=True), patience=s_pat)
            best = {"w": best_ind["w"], "lam": best_ind["lam"], "search_val_mape": best_fit,
                    "search_ep": search_ep, "search_patience": s_pat, "pop": POP, "gen": GEN}
            os.makedirs(os.path.dirname(gw_path), exist_ok=True)
            json_save_retry(best, gw_path, indent=1)
        for bs in fbs:
            name = "MonoPINN" + batch_suffix(bs)
            for sd in seeds:
                od = run_dir(name, sc, sd)
                if run_done(od):
                    continue
                print(f"[{sc}] final {name} seed {sd}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
                if dry:
                    continue
                if data is None:
                    data = load_split(sc, device=dev)
                signs = {int(j): float(data["mono"][int(j)]) for j in best["w"].keys()}
                wmap = {"signs": signs, "w": {int(j): v for j, v in best["w"].items()}}
                net, _ = train_once(data, wmap, best["lam"], final_ep, sd, dev, patience=pat, batch=bs)
                preds = {}
                m = metrics_of(net, data, preds)
                xs_, ys_ = data["x_scaler"], data["y_scaler"]
                save_artifacts(od, X_test=xs_.inverse(data["Xte"].detach().cpu().numpy()), y_true=data["yte_raw"].ravel(),
                               y_pred=preds["test"], state=net.state_dict(),
                               extra={"x_mean": xs_.mean, "x_std": xs_.std, "y_mean": float(np.asarray(ys_.mean).ravel()[0]),
                                      "y_std": float(np.asarray(ys_.std).ravel()[0]), "lam": float(best["lam"]), "w": best["w"]},
                               X_valid=xs_.inverse(data["Xva"].detach().cpu().numpy()), y_valid=data["yva_raw"].ravel(),
                               p_valid=preds["val"])
                json_save_retry(dict(scenario=sc, model=name, method="MonoPINN (Tang et al., 2026) with GA", seed=sd,
                                     batch=bs, epochs=final_ep, patience=pat, lr=LR,
                                     mono_signs=[float(x) for x in data["mono"]], ga=best, metrics=m),
                                os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
                print(f"    test MAPE {m['test']['MAPE']:.2f}%  R2 {m['test']['R2']:.3f}", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
