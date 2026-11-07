# -*- coding: utf-8 -*-
"""Regression metrics in COP units."""
import numpy as np


def regression_metrics(y_true, y_pred):
    """RMSE, MAE, MAPE (%) and R2."""
    y_true = np.asarray(y_true, np.float64).ravel()
    y_pred = np.asarray(y_pred, np.float64).ravel()
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1e-8, None)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
