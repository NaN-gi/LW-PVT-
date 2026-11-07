# -*- coding: utf-8 -*-
"""Train one neural-network comparison model in one scenario: MLP, ICNN-PGNN, TS-PINN or Carnot-PINN.

Model presets (network structure, epochs, learning rate and physics coefficient of each paper):
    MLP          3-128-128-128-1, SiLU; data loss only
    ICNN-PGNN    input-convex network 3-64-64-1 + monotonic penalty (lambda 0.1) on a fixed grid
    TS-PINN      structural network 3-32-32-32-32-1 (ReLU) with Gaussian NLL data loss
                 + trend penalty (lambda 1) on fake sequences regenerated every epoch
    Carnot-PINN  3-128-128-1 (ReLU) + physics-residual loss (lambda 0.3) with an entropy network (32);
                 1,200 epochs, early stopping with patience 50
Common protocol: Adam (lr 0.001), checkpoint with the lowest validation RMSE, mini-batches shuffled
every epoch (batch_size 0 = full batch). The batch size is selected by run/nn_batch_grid.py.

Usage (from the repository root):
    python run/train_nn.py --model MLP --scenario Q_Qe_q1_up --batch_size 8 --epochs 12 --seed 42
                           [--patience 0] [--out_name MLP_b008]
Output: output/<out_name or model>/<scenario>/s<seed>/  metrics.json, best_model.pth, pred_test.csv,
        pred_valid.csv, train_log.csv
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import load_split, FEATURES
from lwpvt.model import MLP, ICNN
from lwpvt.baselines import (icnn_grid, monotone_penalty_at, tspinn_sequence_penalty,
                             carnot_residual_loss, EntropyNet, TSPINNBackbone, GaussianNLLHead)
from lwpvt.metrics import regression_metrics
from lwpvt.paths import run_dir
from lwpvt.saveart import save_artifacts, json_save_retry, csv_save_retry

PRESETS = {
    "MLP":         dict(hidden=(128, 128, 128), act="silu"),
    "ICNN-PGNN":   dict(hidden=(64, 64), lam=0.1),
    "TS-PINN":     dict(hidden=(32, 32, 32, 32), lam=1.0, n_seq=1, interval_degC=0.2),
    "Carnot-PINN": dict(hidden=(128, 128), act="relu", lam=0.3, entropy_hidden=32),
}
LR = 1e-3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_raw(model, X, y_scaler):
    model.eval()
    return y_scaler.inverse(model(X).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PRESETS))
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--batch_size", type=int, default=0, help="0 = full batch (one step per epoch)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the validation RMSE has not improved for this many epochs (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_name", default="", help="output folder name (default: the model name)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = PRESETS[args.model]
    set_seed(args.seed)
    dev = args.device

    data = load_split(args.scenario, device=dev)
    mono = data["mono"]
    print(f"[{args.scenario}] n={data['n']}  model={args.model}  batch={args.batch_size}  "
          f"epochs={args.epochs}  device={dev}", flush=True)

    # -- network
    if args.model == "ICNN-PGNN":
        model = ICNN(in_dim=len(FEATURES), hidden=cfg["hidden"])
    elif args.model == "TS-PINN":
        model = TSPINNBackbone(data["x_scaler"], data["y_scaler"], in_dim=len(FEATURES), hidden=cfg["hidden"])
        model = GaussianNLLHead(model, in_dim=len(FEATURES))
    else:
        model = MLP(in_dim=len(FEATURES), hidden=cfg["hidden"], act=cfg["act"])
    model = model.to(dev)
    grid_pts = icnn_grid(data["Xtr"].detach().cpu().numpy(), device=dev) if args.model == "ICNN-PGNN" else None
    dsnet = EntropyNet(in_dim=len(FEATURES), hidden=cfg["entropy_hidden"]).to(dev) if args.model == "Carnot-PINN" else None
    seq_lens = None
    if args.model == "TS-PINN":                       # sequence length = range / 0.2 degC on the constrained inputs
        xsd = np.asarray(data["x_scaler"].std, dtype=float).ravel()
        dlo = np.asarray(data["domain_std"][0], dtype=float)
        dhi = np.asarray(data["domain_std"][1], dtype=float)
        seq_lens = {j: max(int(np.ceil((dhi[j] - dlo[j]) * xsd[j] / cfg["interval_degC"])) + 1, 2)
                    for j, s in enumerate(mono) if s != 0}
    params = list(model.parameters()) + (list(dsnet.parameters()) if dsnet is not None else [])
    opt = torch.optim.Adam(params, lr=LR)
    mse = torch.nn.MSELoss()

    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva_raw = data["Xva"], data["yva_raw"]
    ys = data["y_scaler"]
    best_val, best_state, log, wait = float("inf"), None, [], 0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.batch_size > 0:                       # mini-batches: shuffle every epoch
            perm = torch.randperm(len(Xtr), device=dev)
            batch_idx = [perm[i:i + args.batch_size] for i in range(0, len(perm), args.batch_size)]
        else:
            batch_idx = [None]
        for bi in batch_idx:
            Xb = Xtr if bi is None else Xtr[bi]
            yb = ytr if bi is None else ytr[bi]
            opt.zero_grad()
            if args.model == "TS-PINN":
                loss_data = model.nll(Xb, yb)
            else:
                loss_data = mse(model(Xb), yb)
            loss = loss_data
            if args.model == "ICNN-PGNN":
                loss = loss + cfg["lam"] * monotone_penalty_at(model, grid_pts, mono)
            elif args.model == "TS-PINN":
                loss = loss + cfg["lam"] * tspinn_sequence_penalty(model, data["domain_std"], mono, seq_lens,
                                                                    n_seq=cfg["n_seq"], device=dev)
            elif args.model == "Carnot-PINN":
                loss = loss + cfg["lam"] * carnot_residual_loss(model, dsnet, Xb, data["x_scaler"], ys, dev)
            loss.backward()
            opt.step()

        # -- checkpoint on the validation RMSE (COP units)
        pva = predict_raw(model, Xva, ys)
        val_rmse = float(np.sqrt(np.mean((pva.ravel() - yva_raw.ravel()) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        log.append((ep, float(loss_data.item()), val_rmse))
        if ep == 1 or ep % 100 == 0:
            print(f"  ep {ep:4d} | data {loss_data.item():.4f} | val_rmse {val_rmse:.3f} | best {best_val:.3f}", flush=True)
        if args.patience > 0 and wait >= args.patience:
            print(f"  early stop at epoch {ep} (no improvement for {args.patience} epochs)", flush=True)
            break

    # -- evaluation of the best checkpoint
    model.load_state_dict(best_state)
    pte = predict_raw(model, data["Xte"], ys)
    pva = predict_raw(model, Xva, ys)
    res = {"test": regression_metrics(data["yte_raw"], pte), "val": regression_metrics(yva_raw, pva)}

    name = args.out_name or args.model
    out_dir = run_dir(name, args.scenario, args.seed)
    xs = data["x_scaler"]
    extra = {"x_mean": xs.mean, "x_std": xs.std, "y_mean": float(ys.mean[0]), "y_std": float(ys.std[0]),
             "hidden": list(cfg["hidden"])}
    if dsnet is not None:
        extra["entropy_net_state"] = {k: v.detach().cpu() for k, v in dsnet.state_dict().items()}
    save_artifacts(out_dir, X_test=xs.inverse(data["Xte"].cpu().numpy()), y_true=data["yte_raw"], y_pred=pte,
                   state=best_state, extra=extra,
                   X_valid=xs.inverse(Xva.cpu().numpy()), y_valid=yva_raw, p_valid=pva)
    json_save_retry(dict(scenario=args.scenario, model=name, method=args.model, seed=args.seed,
                         settings=dict(hidden=list(cfg["hidden"]), lr=LR, epochs=args.epochs,
                                       epochs_run=len(log), batch_size=args.batch_size,
                                       patience=args.patience, lam=cfg.get("lam"),
                                       mono_signs=[float(x) for x in mono]),
                         n=data["n"], best_val_rmse=best_val, metrics=res),
                    os.path.join(out_dir, "metrics.json"), ensure_ascii=False, indent=1)
    csv_save_retry(pd.DataFrame(log, columns=["epoch", "loss_data", "val_rmse"]),
                   os.path.join(out_dir, "train_log.csv"), encoding="utf-8-sig")
    print(f"[done] {name} @ {args.scenario}  test MAPE {res['test']['MAPE']:.2f}%  R2 {res['test']['R2']:.3f}  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
