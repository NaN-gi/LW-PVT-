# -*- coding: utf-8 -*-
"""Network architectures.

MLP   fully connected network used by LW-PVT (3 -> 128 -> 128 -> 128 -> 1, SiLU), by the MLP baseline
      and by Carnot-PINN (3 -> 128 -> 128 -> 1, ReLU)
ICNN  input-convex network of ICNN-PGNN (Guo et al., 2024)
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Input -> hidden layers -> 1 output. act = 'silu' (default) or 'relu'."""

    def __init__(self, in_dim=3, hidden=(128, 128, 128), out_dim=1, act="silu"):
        super().__init__()
        A = nn.ReLU if act == "relu" else nn.SiLU
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), A()]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PosLinear(nn.Module):
    """Linear layer whose weights are kept positive through W = softplus(raw)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Positive weights cannot cancel, so a Kaiming-type scale would grow layer by layer;
        # the rows are initialised to a sum of about 1.25 (unit gain).
        w = torch.abs(torch.randn(out_dim, in_dim)) * (1.25 / in_dim)
        self.raw = nn.Parameter(torch.log(torch.expm1(w.clamp(min=1e-4))))    # inverse softplus
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ torch.nn.functional.softplus(self.raw).T + self.bias


class ICNN(nn.Module):
    """Input-convex network of ICNN-PGNN: the adjusted COP (c - COP) is convex in the inputs, so the
    predicted COP is concave.

    z_1 = softplus(W_0 x + b_0),  z_{k+1} = softplus(W_k^+ z_k + V_k x + b_k)  with W_k^+ >= 0.
    u(x) = W_out^+ z + V_out x is convex; the prediction (standardised COP) is c - u(x) with a learned
    scalar c. The monotonic constraint is not structural; it is imposed by a penalty on a fixed grid
    (baselines.icnn_grid).
    """

    def __init__(self, in_dim=3, hidden=(64, 64)):
        super().__init__()
        self.first = nn.Linear(in_dim, hidden[0])
        self.z = nn.ModuleList([PosLinear(a, b) for a, b in zip(hidden[:-1], hidden[1:])])
        self.xs = nn.ModuleList([nn.Linear(in_dim, h, bias=False) for h in hidden[1:]])
        self.out_z = PosLinear(hidden[-1], 1)
        self.out_x = nn.Linear(in_dim, 1, bias=False)
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = torch.nn.functional.softplus(self.first(x))
        for lz, lx in zip(self.z, self.xs):
            z = torch.nn.functional.softplus(lz(z) + lx(x))
        return self.c - (self.out_z(z) + self.out_x(x))
