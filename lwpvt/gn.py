# -*- coding: utf-8 -*-
"""Gordon-Ng (GN) regression and the positions of the virtual samples (Section 3.1 of the paper).

GN model (Gordon and Ng; three coefficients, absolute temperatures):
    (T_e / T_c) (1 + 1/COP) - 1 = b1 T_e / Q_e + b2 (T_c - T_e) / (T_c Q_e) + b3 (Q_e / T_c) (1 + 1/COP)
with T_e = T_chw,out + 273.15 K and T_c = T_cw,in + 273.15 K. The model is linear in b, so it is
fitted by least squares to the training rows, where COP is known.
"""
import numpy as np
from scipy.spatial import cKDTree


def gn_terms(X_raw, cop):
    """Design matrix F [N, 3] and left-hand side of the GN model (inputs in physical units)."""
    qe = np.clip(X_raw[:, 0], 1.0, None)
    te = X_raw[:, 1] + 273.15
    tc = X_raw[:, 2] + 273.15
    k = 1.0 + 1.0 / np.clip(cop, 0.5, None)
    F = np.column_stack([te / qe, (tc - te) / (tc * qe), (qe / tc) * k])
    lhs = (te / tc) * k - 1.0
    return F, lhs


def fit_gn(X_raw, y_raw):
    """Least-squares fit of the GN coefficients. Returns dict(beta, FtF_inv); FtF_inv gives the leverage."""
    F, lhs = gn_terms(X_raw, y_raw)
    beta, *_ = np.linalg.lstsq(F, lhs, rcond=None)
    FtF = F.T @ F
    try:
        FtF_inv = np.linalg.inv(FtF)
    except np.linalg.LinAlgError:
        FtF_inv = np.linalg.pinv(FtF)
    return dict(beta=beta, FtF_inv=FtF_inv)


def gn_predict(beta, X_raw):
    """COP predicted by the GN model, clipped to [1, 12].

    k = [1 + b1 T_e/Q_e + b2 (T_c - T_e)/(T_c Q_e)] / [(T_e - b3 Q_e)/T_c],  COP = 1 / (k - 1).
    """
    qe = np.clip(X_raw[:, 0], 1.0, None)
    te = X_raw[:, 1] + 273.15
    tc = X_raw[:, 2] + 273.15
    num = 1.0 + beta[0] * te / qe + beta[1] * (tc - te) / (tc * qe)
    den = (te - beta[2] * qe) / tc
    k = num / np.where(np.abs(den) < 1e-9, 1e-9, den)
    return 1.0 / np.clip(k - 1.0, 1.0 / 12.0, 1.0)


def sample_virtual_inputs(Xtr_std, domain_std, n, seed, min_dist=0.15, n_cand=20000):
    """Positions of the virtual samples in standardised coordinates.

    Candidates are drawn uniformly in the operating box; those within min_dist (Euclidean distance in
    standardised coordinates) of a training row are discarded, and n of the remaining candidates are
    kept. A dedicated random generator is used so that the global NumPy stream is not consumed.
    Returns [m, F] with m <= n.
    """
    rng = np.random.default_rng(10000 + seed)
    lo, hi = np.asarray(domain_std[0]), np.asarray(domain_std[1])
    cand = lo + rng.random((n_cand, len(lo))).astype(np.float32) * (hi - lo)
    d, _ = cKDTree(Xtr_std).query(cand, k=1)
    far = cand[d > min_dist]
    if len(far) > n:
        far = far[rng.choice(len(far), n, replace=False)]
    return far.astype(np.float32)
