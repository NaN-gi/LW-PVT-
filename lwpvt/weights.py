# -*- coding: utf-8 -*-
"""Labels, label uncertainty and inverse-variance weights of the virtual samples (Section 3.2).

For virtual sample j at input x_j:
    label          y_j = COP_GN(x_j)                          GN model fitted to the training rows
    uncertainty    sigma_j^2 = s_train^2 (1 + h_j) + sigma_win,j^2
        s_train        RMSE of the GN model on the training rows
        h_j            leverage of x_j in the GN design matrix (evaluated at the predicted COP, no label)
        sigma_win,j    standard deviation of COP_GN(x_j) over refits of the GN model to the training rows
                       with the lower or the upper quartile of one input removed (window perturbation)
    weight share   p_j = sigma_j^-2 / sum_k sigma_k^-2            (sum_j p_j = 1)
The total weight of the virtual samples is W = lambda1 * n_train (lambda1 = 0.0005 in the paper), so
that the loss term is (W / n_train) * sum_j p_j (f(x_j) - y_j)^2 = lambda1 * sum_j p_j (f(x_j) - y_j)^2.
"""
import numpy as np

from .gn import fit_gn, gn_predict, gn_terms


def window_betas(Xtr_raw, ytr_raw, min_rows=200, frac=0.25):
    """GN coefficients of the full fit and of the refits with one edge window of one input removed
    (3 inputs x 2 ends; a refit is kept only when at least min_rows rows remain)."""
    X = np.asarray(Xtr_raw, float)
    y = np.asarray(ytr_raw, float).ravel()
    betas = [fit_gn(X, y)["beta"]]
    for j in range(X.shape[1]):
        a = X[:, j]
        for lo, hi in [(np.quantile(a, frac), np.inf), (-np.inf, np.quantile(a, 1.0 - frac))]:
            m = (a >= lo) & (a <= hi)
            if m.sum() >= min_rows:
                betas.append(fit_gn(X[m], y[m])["beta"])
    return betas


def window_sigma(betas, X_raw):
    """Standard deviation of the GN prediction over the committee of coefficient vectors (COP units)."""
    P = np.stack([gn_predict(b, X_raw) for b in betas])
    return P.std(0)


def label_pack(Xtr_raw, ytr_raw, T_raw, win_frac=0.25, min_rows=200):
    """Labels, uncertainties and weight shares of the virtual samples at inputs T_raw (physical units).

    Only the training rows (Xtr_raw, ytr_raw) are used.
    Returns dict(labels, sigma, p, s_train, sigma_pred, sigma_win, beta).
    """
    Xf, yf = np.asarray(Xtr_raw, float), np.asarray(ytr_raw, float).ravel()
    gnm = fit_gn(Xf, yf)
    resid = gn_predict(gnm["beta"], Xf) - yf
    s2 = float(np.mean(resid ** 2))
    s_train = float(np.sqrt(s2))
    cop_hat = gn_predict(gnm["beta"], T_raw)
    F, _ = gn_terms(T_raw, cop_hat)
    h = np.clip(np.einsum("ij,jk,ik->i", F, gnm["FtF_inv"], F), 0.0, None)
    sigma_pred = np.sqrt(s2 * (1.0 + h))                         # prediction error of the regression
    sigma_win = window_sigma(window_betas(Xf, yf, min_rows=min_rows, frac=win_frac), T_raw)
    sigma = np.sqrt(sigma_pred ** 2 + sigma_win ** 2)
    v = (1.0 / np.clip(sigma, 1e-12, None)) ** 2                  # inverse variance
    p = v / v.sum()
    return dict(labels=gn_predict(gnm["beta"], T_raw), sigma=sigma, p=p, s_train=s_train,
                sigma_pred=sigma_pred, sigma_win=sigma_win, beta=gnm["beta"])
