# -*- coding: utf-8 -*-
"""LW-PVT: lightly weighted physics-labeled virtual sample training for chiller COP extrapolation.

Modules
-------
data       RP-1043 rows of the 15 extrapolation scenarios, standardisation
model      MLP and the input-convex network used by ICNN-PGNN
physics    collocation sampling and the physics penalties of LW-PVT
gn         Gordon-Ng regression and the positions of the virtual samples
weights    label uncertainty and inverse-variance sample weights of the virtual samples
baselines  physics terms of the comparison models (ICNN-PGNN, TS-PINN, Carnot-PINN) and the MP design matrix
metrics    RMSE, MAE, MAPE, R2
paths      folder layout and the run-folder convention
saveart    saving of predictions, model state and metrics
"""
