# -*- coding: utf-8 -*-
"""PI-WGAN (Wang et al., 2025, Energy Conversion and Management 346): data augmentation with a
physics-informed Wasserstein GAN followed by an 8-coefficient polynomial regression.

Method as in its paper (same RP-1043 chiller and the same three inputs):
  1. WGAN-GP: generator and critic with three hidden layers of 256 units (LeakyReLU, normal
     initialisation). The generator outputs (T_chw,out, T_cw,in, Q_e) and an internal COP layer
     (8 tanh units) adds the COP. L_D = E[D(x)] - E[D(G(z))] + lambda_gp GP; L_G = -E[D(G(z))] + lambda L_phys.
  2. Physics penalty on K = 5 generated samples per iteration: for every pair, the violations of
     "COP increases with T_chw,out" and "COP decreases with T_cw,in" are weighted by a Gaussian
     similarity of the other inputs (sigma = 0.2).
  3. Post filter of the generated samples: T_chw,out < T_cw,in and 1 < COP < 0.3 COP_Carnot.
  4. Adam (lr 0.001), 5,000 iterations; five times the number of training rows are generated, merged
     with the training rows, and the 8-coefficient polynomial
     COP = b0 + b1 Qe + b2 Teo + b3 Tci + b4 Qe^2 + b5 Qe Teo + b6 Qe Tci + b7 Teo Tci is refitted.
Settings not given in the paper: noise dimension 16, batch 64, five critic updates per generator
update, lambda_gp = 10, Adam betas (0.5, 0.9), training in standardised units. The physics coefficient
lambda was tuned per task in the paper, so the grid {0.2, 1, 10, 32} is run and the value with the best
mean rank is kept (results/main_table.py).

Usage (from the repository root):
    python run/piwgan.py [--lam 0.2,1,10,32] [--iters 5000] [--mult 5] [--seeds 10] [--scenarios ...] [--dry] [--smoke]
Selected setting of the paper: --lam 32.
Output: output/PI-WGAN_lam<lambda>/<scenario>/s<seed>/
"""
import os
import sys
import time
import itertools
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.data import FEATURES, TARGET, load_rows
from lwpvt.metrics import regression_metrics
from lwpvt.paths import SCENARIOS, SEEDS, run_dir
from lwpvt.saveart import save_artifacts, run_done, json_save_retry

ZDIM, HID, BATCH, N_CRITIC, LAM_GP, LR, SIG_PHY, K_PAIR, H_CARNOT = 16, 256, 64, 5, 10.0, 1e-3, 0.2, 5, 0.3
COLS = ["TEO_C", "TCI_C", "Qe_kW"]                      # order of the paper: T_chw,out, T_cw,in, Q_e (+ COP)


def init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(ZDIM, HID), nn.LeakyReLU(0.2), nn.Linear(HID, HID), nn.LeakyReLU(0.2),
                                  nn.Linear(HID, HID), nn.LeakyReLU(0.2), nn.Linear(HID, 3))
        self.cop = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))   # internal COP layer
        self.apply(init_normal)

    def forward(self, z):
        x = self.body(z)
        return torch.cat([x, self.cop(x)], 1)


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, HID), nn.LeakyReLU(0.2), nn.Linear(HID, HID), nn.LeakyReLU(0.2),
                                 nn.Linear(HID, HID), nn.LeakyReLU(0.2), nn.Linear(HID, 1))
        self.apply(init_normal)

    def forward(self, x):
        return self.net(x)


PAIRS = list(itertools.combinations(range(K_PAIR), 2))


def physics_loss(xg):
    """Pairwise soft monotonic constraints. xg: [K, 4] standardised generated samples
    (T_chw,out, T_cw,in, Q_e, COP)."""
    i = torch.tensor([p[0] for p in PAIRS], device=xg.device)
    j = torch.tensor([p[1] for p in PAIRS], device=xg.device)
    d = xg[i] - xg[j]                                        # [n_pairs, 4]
    dTeo, dTci, dQ, dCOP = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    s1 = torch.exp(-(dTci ** 2 + dQ ** 2) / SIG_PHY ** 2)    # rule 1: other inputs (T_cw,in, Q_e)
    s2 = torch.exp(-(dTeo ** 2 + dQ ** 2) / SIG_PHY ** 2)    # rule 2: other inputs (T_chw,out, Q_e)
    l1 = s1 * torch.relu(-dTeo * dCOP)                       # COP must move with T_chw,out
    l2 = s2 * torch.relu(dTci * dCOP)                        # COP must move against T_cw,in
    return (l1 + l2).mean()


def gradient_penalty(Dn, xr, xf):
    a = torch.rand(xr.size(0), 1, device=xr.device)
    xh = (a * xr + (1 - a) * xf).requires_grad_(True)
    g = torch.autograd.grad(Dn(xh).sum(), xh, create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def mp8(X):
    q, t1, t2 = X[:, 0], X[:, 1], X[:, 2]                    # FEATURES order: Qe, TEO, TCI
    return np.column_stack([np.ones(len(X)), q, t1, t2, q * q, q * t1, q * t2, t1 * t2])


def run_one(sc, seed, dev, lam, iters, mult):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = load_rows(sc)
    tr, va, te = (df[df["role"] == r] for r in ("train", "val", "test"))
    R = tr[COLS + [TARGET]].to_numpy(np.float32)             # [n, 4]
    mu, sd = R.mean(0), R.std(0) + 1e-8
    Xr = torch.tensor((R - mu) / sd, device=dev)
    n = len(Xr)
    Gn, Dn = G().to(dev), D().to(dev)
    oG = torch.optim.Adam(Gn.parameters(), lr=LR, betas=(0.5, 0.9))
    oD = torch.optim.Adam(Dn.parameters(), lr=LR, betas=(0.5, 0.9))
    for it in range(iters):
        for _ in range(N_CRITIC):
            xr = Xr[torch.randint(0, n, (BATCH,), device=dev)]
            xf = Gn(torch.randn(BATCH, ZDIM, device=dev)).detach()
            lD = -(Dn(xr).mean() - Dn(xf).mean()) + LAM_GP * gradient_penalty(Dn, xr, xf)
            oD.zero_grad()
            lD.backward()
            oD.step()
        xf = Gn(torch.randn(BATCH, ZDIM, device=dev))
        lphy = physics_loss(xf[torch.randperm(BATCH, device=dev)[:K_PAIR]])
        lG = -Dn(xf).mean() + lam * lphy
        oG.zero_grad()
        lG.backward()
        oG.step()
    # generation and post filter
    Gn.eval()
    want = mult * n
    keep, tries = [], 0
    with torch.no_grad():
        while sum(len(k) for k in keep) < want and tries < 50:
            xf = Gn(torch.randn(max(want, 4096), ZDIM, device=dev)).cpu().numpy() * sd + mu
            teo, tci, qe, cop = xf[:, 0], xf[:, 1], xf[:, 2], xf[:, 3]
            carnot = (teo + 273.15) / np.clip(tci - teo, 1e-3, None)
            ok = (teo < tci) & (cop > 1.0) & (cop < H_CARNOT * carnot) & (qe > 0)
            keep.append(xf[ok])
            tries += 1
    S = np.vstack(keep)[:want]
    n_syn = len(S)
    Xs = np.column_stack([S[:, 2], S[:, 0], S[:, 1]]).astype(np.float32)   # FEATURES order (Qe, TEO, TCI)
    ys = S[:, 3].astype(np.float32)
    Xtr, ytr = tr[FEATURES].to_numpy(np.float32), tr[TARGET].to_numpy(np.float32)
    Xva, yva = va[FEATURES].to_numpy(np.float32), va[TARGET].to_numpy(float)
    Xte, yte = te[FEATURES].to_numpy(np.float32), te[TARGET].to_numpy(float)
    xm, xsd = Xtr.mean(0), Xtr.std(0) + 1e-8                 # standardisation for the polynomial fit
    Z = lambda X: (X - xm) / xsd
    beta_aug, *_ = np.linalg.lstsq(mp8(Z(np.vstack([Xtr, Xs]))), np.concatenate([ytr, ys]), rcond=None)
    beta_no, *_ = np.linalg.lstsq(mp8(Z(Xtr)), ytr, rcond=None)
    pva, pte = mp8(Z(Xva)) @ beta_aug, mp8(Z(Xte)) @ beta_aug
    out = {"val": regression_metrics(yva, pva), "test": regression_metrics(yte, pte)}
    art = dict(state=Gn.state_dict(), extra={"beta_aug": beta_aug, "beta_noaug": beta_no, "x_mean": xm, "x_std": xsd,
                                             "gan_mu": mu, "gan_sd": sd, "n_synth": int(n_syn),
                                             "note": "predictor = 8-coefficient polynomial beta_aug on standardised inputs; state_dict = generator"},
               X_test=Xte, y_true=yte, y_pred=pte, X_valid=Xva, y_valid=yva, p_valid=pva)
    diag = {"polynomial_without_augmentation": {"val": regression_metrics(yva, mp8(Z(Xva)) @ beta_no),
                                                "test": regression_metrics(yte, mp8(Z(Xte)) @ beta_no)},
            "n_train": int(n), "n_synth": int(n_syn), "filter_tries": tries, "phys_loss_last": float(lphy.detach().item())}
    return out, diag, art


if __name__ == "__main__":
    argv = sys.argv[1:]
    dry = "--dry" in argv
    n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
    use_sc = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
    lams = [float(x) for x in argv[argv.index("--lam") + 1].split(",")] if "--lam" in argv else [0.2, 1.0, 10.0, 32.0]
    iters = int(argv[argv.index("--iters") + 1]) if "--iters" in argv else 5000
    mult = int(argv[argv.index("--mult") + 1]) if "--mult" in argv else 5
    if "--smoke" in argv:                                  # short run for testing the pipeline
        iters = 200
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:n_seeds]
    jobs = [(lam, sc, sd) for lam in lams for sc in use_sc for sd in seeds]
    print(f"PI-WGAN: lambda {lams} x {len(use_sc)} scenarios x {len(seeds)} seeds = {len(jobs)} runs | "
          f"{iters} iterations, {mult}x synthetic | device={dev}", flush=True)
    t0 = time.time()
    for i, (lam, sc, sd) in enumerate(jobs, 1):
        name = "PI-WGAN_lam" + ("%g" % lam).replace(".", "p")
        od = run_dir(name, sc, sd)
        if run_done(od):
            continue
        print(f"[{i}/{len(jobs)}] {sc} {name} seed {sd}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
        if dry:
            continue
        m, diag, art = run_one(sc, sd, dev, lam, iters, mult)
        save_artifacts(od, **art)
        json_save_retry(dict(scenario=sc, model=name, method="PI-WGAN augmentation + 8-coefficient polynomial (Wang et al., 2025)",
                             seed=sd, settings=dict(lam_physics=lam, iters=iters, mult=mult, zdim=ZDIM, hidden=HID, batch=BATCH,
                                                    n_critic=N_CRITIC, lam_gp=LAM_GP, lr=LR, sigma_physical=SIG_PHY,
                                                    K=K_PAIR, H_carnot=H_CARNOT),
                             diag=diag, metrics=m),
                        os.path.join(od, "metrics.json"), ensure_ascii=False, indent=1)
        print(f"    test MAPE {m['test']['MAPE']:.2f}%  R2 {m['test']['R2']:.3f}  "
              f"(polynomial without augmentation {diag['polynomial_without_augmentation']['test']['MAPE']:.2f}%, "
              f"{diag['n_synth']} synthetic)", flush=True)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)
