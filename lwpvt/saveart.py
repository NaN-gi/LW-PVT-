# -*- coding: utf-8 -*-
"""Saving of the artifacts of one run.

Every run folder receives
  metrics.json     metrics on the validation and test rows and the run settings
  pred_test.csv    test rows in their original order: the three inputs, y_true, y_pred
  pred_valid.csv   the same for the validation rows (when available)
  best_model.pth   model state (state_dict or regression coefficients) with the scalers
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch

FEATURES = ["Qe_kW", "TEO_C", "TCI_C"]


def _np(a):
    return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float64)


def retry_io(fn, tries=6, first_wait=1.0):
    """Retry a file operation that may fail while the file is locked by another process."""
    wait = first_wait
    for k in range(tries):
        try:
            return fn()
        except (PermissionError, OSError) as e:
            if k == tries - 1:
                raise
            print(f"    [retry {k + 1}/{tries - 1}] {type(e).__name__}: {e}; waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 16)


def torch_save_retry(obj, path):
    """Write to a temporary file and replace, so that a partial file is never left behind."""
    tmp = path + ".tmp"

    def _w():
        torch.save(obj, tmp)
        os.replace(tmp, path)
    return retry_io(_w)


def csv_save_retry(df, path, **kw):
    tmp = path + ".tmp"

    def _w():
        df.to_csv(tmp, index=False, **kw)
        os.replace(tmp, path)
    retry_io(_w)
    return df


def json_save_retry(obj, path, **kw):
    tmp = path + ".tmp"

    def _w():
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **kw)
        os.replace(tmp, path)
    return retry_io(_w)


def save_artifacts(out_dir, X_test, y_true, y_pred, state=None, extra=None, X_valid=None, y_valid=None, p_valid=None):
    """Write pred_test.csv (and pred_valid.csv) and best_model.pth into out_dir.

    X_test: inputs in physical units [n, 3] in the FEATURES order.  state: state_dict or None.
    extra: scalers, coefficients and other values needed to reproduce the predictions.
    """
    os.makedirs(out_dir, exist_ok=True)
    Xt = _np(X_test)
    df = {c: Xt[:, i] for i, c in enumerate(FEATURES[:Xt.shape[1]])}
    df.update({"y_true": _np(y_true).ravel(), "y_pred": _np(y_pred).ravel()})
    csv_save_retry(pd.DataFrame(df), os.path.join(out_dir, "pred_test.csv"), encoding="utf-8-sig")
    if X_valid is not None and p_valid is not None:
        Xv = _np(X_valid)
        dv = {c: Xv[:, i] for i, c in enumerate(FEATURES[:Xv.shape[1]])}
        dv.update({"y_true": _np(y_valid).ravel(), "y_pred": _np(p_valid).ravel()})
        csv_save_retry(pd.DataFrame(dv), os.path.join(out_dir, "pred_valid.csv"), encoding="utf-8-sig")
    ck = {"features": FEATURES}
    if state is not None:
        ck["state_dict"] = {k: v.detach().cpu() for k, v in state.items()}
    for k, v in (extra or {}).items():
        ck[k] = v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, (np.floating,)) else v)
    torch_save_retry(ck, os.path.join(out_dir, "best_model.pth"))


def run_done(out_dir):
    """True when the run folder already holds metrics.json (used to resume an interrupted grid)."""
    return os.path.exists(os.path.join(out_dir, "metrics.json"))
