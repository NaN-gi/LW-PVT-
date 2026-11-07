# -*- coding: utf-8 -*-
"""Physics loss of LW-PVT (Section 3.3 of the paper).

At collocation points drawn uniformly in the operating box, the derivatives of the network are obtained
by automatic differentiation and only the sign violations are penalised (no labels are needed):
  monotonic (first order)   s_j * dy/dx_j >= 0 is required; violation = ReLU(-s_j * dy/dx_j)
  curvature (second order)  c_j * d2y/dx_j2 >= 0 is required; violation = ReLU(-c_j * d2y/dx_j2)
  range (zeroth order)      1 <= COP <= COP_Carnot
Standardisation is a positive affine map, so the signs of the derivatives are the same as in physical
units and the penalties can be evaluated in standardised coordinates.
"""
import torch


def sample_collocation(domain_std, n, device):
    """n points drawn uniformly in the standardised operating box, with requires_grad=True."""
    lo = torch.as_tensor(domain_std[0], device=device, dtype=torch.float32)
    hi = torch.as_tensor(domain_std[1], device=device, dtype=torch.float32)
    u = torch.rand(n, lo.shape[0], device=device)
    x = lo + u * (hi - lo)
    x.requires_grad_(True)
    return x


def physics_penalty(model, x_colloc, mono, curv):
    """Mean monotonic and curvature violations at the collocation points.

    mono, curv: sign vectors over the inputs (+1 / -1 / 0 = no constraint).
    Returns (L_mono, L_curv) as scalars for backpropagation.
    """
    n = x_colloc.shape[0]
    y = model(x_colloc)                                                # [N, 1]
    g = torch.autograd.grad(y.sum(), x_colloc, create_graph=True)[0]   # [N, F] first derivatives
    mono_t = torch.as_tensor(mono, device=x_colloc.device, dtype=g.dtype)
    curv_t = torch.as_tensor(curv, device=x_colloc.device, dtype=g.dtype)

    v_mono = x_colloc.new_zeros(n)
    for j in range(g.shape[1]):
        s = mono_t[j]
        if s != 0:
            v_mono = v_mono + torch.relu(-s * g[:, j])

    v_curv = x_colloc.new_zeros(n)
    for j in range(g.shape[1]):
        c = curv_t[j]
        if c != 0:
            d2 = torch.autograd.grad(g[:, j].sum(), x_colloc, create_graph=True)[0][:, j]
            v_curv = v_curv + torch.relu(-c * d2)

    return v_mono.mean(), v_curv.mean()


def cop_physics_bounds(x_colloc, y_pred, x_scaler, y_scaler, teo_idx=1, tci_idx=2):
    """Range penalty: COP >= 1 and COP <= COP_Carnot = (T_chw,out + 273.15) / (T_cw,in - T_chw,out).

    x_colloc [N, F] standardised inputs, y_pred [N, 1] standardised prediction. The bounds are
    functions of the inputs only and are treated as constants (detached).
    """
    dev, dt = x_colloc.device, y_pred.dtype
    xm = torch.as_tensor(x_scaler.mean, device=dev, dtype=dt)
    xs = torch.as_tensor(x_scaler.std, device=dev, dtype=dt)
    ym, ysd = float(y_scaler.mean[0]), float(y_scaler.std[0])
    teo = (x_colloc[:, teo_idx] * xs[teo_idx] + xm[teo_idx]).detach()
    tci = (x_colloc[:, tci_idx] * xs[tci_idx] + xm[tci_idx]).detach()
    carnot = 1.0 * (teo + 273.15) / torch.clamp(tci - teo, min=0.5)
    lo_std = (1.0 - ym) / ysd                                        # COP >= 1 in standardised units
    hi_std = ((carnot - ym) / ysd).unsqueeze(1)                      # [N, 1] Carnot bound
    return torch.relu(lo_std - y_pred).mean() + torch.relu(y_pred - hi_std).mean()
