# -*- coding: utf-8 -*-
"""Physics terms of the comparison models, written for the three-input COP task of the paper.

ICNN-PGNN   Guo et al. (2024), Applied Energy 354: input-convex network (model.ICNN) with a monotonic
            penalty (lambda = 0.1) on a fixed grid of points outside the training range of the
            constrained inputs.
TS-PINN     Liang et al. (2025), Applied Energy 390: structural network (inverse-Carnot power with a
            network for the entropy generation) trained with a Gaussian negative log-likelihood and a
            trend penalty on sequences of fake inputs that are regenerated every epoch (lambda = 1).
Carnot-PINN Zhu et al. (2025), Energies 18: physics-residual loss between the prediction and the COP of
            an inverse-Carnot power model with a network for the entropy term (lambda = 0.3).
MP          multivariate quadratic polynomial with 10 coefficients (design matrix mp_design).
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- ICNN-PGNN
def icnn_grid(train_X_std, n_side=6, n_settings=5, device="cpu"):
    """Fixed penalty grid of ICNN-PGNN on the constrained inputs (T_chw,out = 1, T_cw,in = 2).

    For each constrained input, n_side grid values are placed outside the training range (half of them
    below, half above, up to one range width away) for n_settings joint settings of the other inputs at
    the training quantiles {0, .25, .5, .75, 1}. The grid is fixed during training.
    """
    X = np.asarray(train_X_std, dtype=np.float32)
    tr_lo, tr_hi = X.min(0), X.max(0)
    span = tr_hi - tr_lo
    pts = []
    half = n_side // 2
    for j in (1, 2):
        below = tr_lo[j] - span[j] * np.linspace(1.0, 1.0 / half, half, dtype=np.float32)
        above = tr_hi[j] + span[j] * np.linspace(1.0 / half, 1.0, half, dtype=np.float32)
        axis = np.concatenate([below, above])
        for q in np.linspace(0.0, 1.0, n_settings, dtype=np.float32):
            base = tr_lo + q * span
            for a in axis:
                p = base.copy()
                p[j] = a
                pts.append(p)
    return torch.tensor(np.stack(pts), device=device)


def monotone_penalty_at(model, X, mono):
    """Mean monotonic violation at the given points (first-order part of physics.physics_penalty)."""
    x = X.detach().clone().requires_grad_(True)
    y = model(x)
    g = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    v = x.new_zeros(x.shape[0])
    for j, s in enumerate(mono):
        if s != 0:
            v = v + torch.relu(-float(s) * g[:, j])
    return v.mean()


# ---------------------------------------------------------------- TS-PINN
def tspinn_sequence_penalty(model, domain_std, mono, seq_lens, n_seq=1, device="cpu"):
    """Trend penalty of TS-PINN: for each constrained input, n_seq fake sequences sweep the whole range
    of that input on a fixed grid (seq_lens[j] points) while the other inputs take random values; the
    sign violations of the differences between neighbouring predictions are penalised. The sequences
    are discarded after the step."""
    lo = torch.as_tensor(domain_std[0], device=device, dtype=torch.float32)
    hi = torch.as_tensor(domain_std[1], device=device, dtype=torch.float32)
    total = torch.zeros((), device=device)
    cnt = 0
    for j, s in enumerate(mono):
        if s == 0:
            continue
        L = int(seq_lens[j])
        grid = torch.linspace(0.0, 1.0, L, device=device)
        u = torch.rand(n_seq, 1, lo.shape[0], device=device)
        x = (lo + u * (hi - lo)).expand(n_seq, L, lo.shape[0]).clone()
        x[:, :, j] = lo[j] + grid.unsqueeze(0) * (hi[j] - lo[j])
        y = model(x.reshape(-1, lo.shape[0])).reshape(n_seq, L)
        d = y[:, 1:] - y[:, :-1]
        total = total + torch.relu(-float(s) * d).mean()
        cnt += 1
    return total / max(cnt, 1)


class TSPINNBackbone(nn.Module):
    """Structural network of TS-PINN written for the COP output.

    P = (T_c - T_e) / T_e * Q_e + ReLU(NN(x)) * T_c  (inverse-Carnot power plus entropy generation),
    COP = Q_e / P, returned in standardised units. The output bias of the network is initialised to 0.1
    so that the ReLU does not start dead.
    """

    def __init__(self, x_scaler, y_scaler, in_dim=3, hidden=(32, 32, 32, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].bias.fill_(0.1)
        self.register_buffer("xm", torch.tensor(np.asarray(x_scaler.mean, np.float32)).reshape(1, -1))
        self.register_buffer("xsd", torch.tensor(np.asarray(x_scaler.std, np.float32)).reshape(1, -1))
        self.register_buffer("ym", torch.tensor(float(np.asarray(y_scaler.mean).ravel()[0])))
        self.register_buffer("ysd", torch.tensor(float(np.asarray(y_scaler.std).ravel()[0])))

    def forward(self, x_std):
        raw = x_std * self.xsd + self.xm
        qe = torch.clamp(raw[:, 0:1], min=1.0)
        te = raw[:, 1:2] + 273.15
        tc = torch.maximum(raw[:, 2:3] + 273.15, te + 0.5)
        p = (tc - te) / te * qe + torch.relu(self.net(x_std)) * tc
        cop = qe / torch.clamp(p, min=1e-3)
        return (cop - self.ym) / self.ysd


class GaussianNLLHead(nn.Module):
    """Gaussian negative log-likelihood output of TS-PINN: the backbone predicts the mean and a small
    head predicts the heteroscedastic log variance. forward() returns the mean only, so validation and
    test predictions are unchanged; nll() is used as the data loss."""

    def __init__(self, backbone, in_dim=3, hidden=32):
        super().__init__()
        self.backbone = backbone
        self.logvar = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.backbone(x)

    def nll(self, x, y):
        mu = self.backbone(x)
        lv = torch.clamp(self.logvar(x), -8.0, 4.0)
        return 0.5 * (lv + (y - mu) ** 2 / torch.exp(lv)).mean()


# ---------------------------------------------------------------- Carnot-PINN
class EntropyNet(nn.Module):
    """Auxiliary network for the entropy term, Delta S(x) >= 0 (two hidden layers, softplus output)."""

    def __init__(self, in_dim=3, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


def carnot_residual_loss(model, dsnet, X_std, x_scaler, y_scaler, device):
    """Physics residual of Carnot-PINN at the training inputs:
    P_phy = (T_c - T_e) / T_e * Q_e + Delta S * T_c,  COP_phy = Q_e / P_phy,
    loss = mean (f(x) - COP_phy)^2 in standardised units."""
    xm = torch.as_tensor(x_scaler.mean, device=device, dtype=torch.float32).reshape(1, -1)
    xs = torch.as_tensor(x_scaler.std, device=device, dtype=torch.float32).reshape(1, -1)
    raw = X_std * xs + xm
    qe = torch.clamp(raw[:, 0:1], min=1.0)
    te = raw[:, 1:2] + 273.15
    tc = torch.maximum(raw[:, 2:3] + 273.15, te + 0.5)
    p_phy = (tc - te) / te * qe + dsnet(X_std) * tc
    cop_phy = qe / torch.clamp(p_phy, min=1e-3)
    ym = float(y_scaler.mean[0])
    ysd = float(y_scaler.std[0])
    cop_phy_std = (cop_phy - ym) / ysd
    return torch.mean((model(X_std) - cop_phy_std) ** 2)


# ---------------------------------------------------------------- MP
def mp_design(X):
    """Design matrix of the 10-coefficient quadratic polynomial:
    [1, q, t1, t2, q^2, t1^2, t2^2, q t1, q t2, t1 t2] for X = (Q_e, T_chw,out, T_cw,in), standardised."""
    q, t1, t2 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([np.ones(len(X)), q, t1, t2, q * q, t1 * t1, t2 * t2,
                            q * t1, q * t2, t1 * t2])
