# LW-PVT: lightly weighted physics-labeled virtual sample training

Code of the paper *Lightly weighted physics-labeled virtual sample training for improving the
extrapolation of chiller COP prediction* (N. Kim and S. Sim).

The repository contains the data preparation, the 15 extrapolation scenarios, the proposed method
(LW-PVT) and the nine comparison models of the main results table (Table 3 of the paper), and the
script that assembles the table from the finished runs.

## Method in short

LW-PVT trains one multilayer perceptron with three loss terms,

    L = L_data + lambda1 * L_virtual + L_phys

L_data: mean squared error on the measured training data.
L_virtual: weighted squared error on N = 1,024 virtual samples placed in the unmeasured part of
  the operating range. Their labels come from a Gordon-Ng regression fitted to the training data only,
  and each sample carries an inverse-variance share p_j of the total weight W = lambda1 * n_train
  (lambda1 = 0.0005).
L_phys: hinge penalties at 512 collocation points: dCOP/dT_cw,in <= 0, d2COP/dQ_e2 <= 0 and
  1 <= COP <= COP_Carnot.

Only the training data and the operating range are used; no manufacturer data and no test
information enter the method.

## Repository layout

```
LW-PVT/
├── lwpvt/                core package
│   ├── data.py           rows of a scenario, standardisation
│   ├── model.py          MLP, input-convex network (ICNN-PGNN)
│   ├── physics.py        collocation points and physics penalties of LW-PVT
│   ├── gn.py             Gordon-Ng regression, positions of the virtual samples
│   ├── weights.py        labels, label uncertainty and inverse-variance weights of the virtual samples
│   ├── baselines.py      physics terms of ICNN-PGNN, TS-PINN, Carnot-PINN; MP design matrix
│   └── metrics.py, paths.py, saveart.py
├── prepare/              01 pool the RP-1043 files -> 02 normal operating points -> 03 scenarios
│                         -> 04 gather the +/-1% data -> 05 scenario roles of the gathered data
├── run/                  one script per model (see the table below)
├── results/main_table.py Table 3 from the finished runs
├── data/                 RP-1043 files go here (not redistributed; see data/README.md)
└── output/               run folders: output/<model>/<scenario>/s<seed>/
```

## Installation

Python 3.11 with the packages in `requirements.txt` (PyTorch 2.x; a GPU is used when available,
otherwise the CPU):

```
pip install -r requirements.txt
```

## Data

The ASHRAE RP-1043 data are not included. Obtain the data set from ASHRAE, copy the test files to
`data/raw/ASHRAE_RP-1043/Chiller Data/` (see `data/README.md`) and run, from the repository root,

```
python prepare/01_preprocess_rp1043.py
python prepare/02_make_steady_normal.py
python prepare/03_make_splits.py
python prepare/04_make_gathered_datasets.py
python prepare/05_make_gathered_splits.py
```

This produces the 486 normal operating points, the 35,641 data within +/-1% of an operating point,
and the roles (train / val / test) of every datum in the 15 scenarios
(`Q_<variable>_<mode>` with variable in {Qe, TEO, TCI} = {Q_e, T_chw,out, T_cw,in} and mode in
{q1_up, half_up, mid_out, half_dn, q3_dn}; the paper writes them "Q_e Q1 up", "T_chw,out Half down", ...).

## Models of Table 3

All scripts run the 15 scenarios with ten seeds (42-51) and skip runs whose `metrics.json` already
exists, so an interrupted grid can be resumed. The comparison models use the network structure,
epochs, learning rate and physics coefficient of their papers. Because their papers give no batch
size for this task, the seven neural-network comparison models other than PI-WGAN were trained with
seven batch sizes (full batch, 8, 16, 32, 64, 128, 256), and `results/main_table.py` keeps the batch
size with the best mean rank over the 15 scenarios. The same rule selects the physics coefficient of
PI-WGAN from {0.2, 1, 10, 32}. The second column gives the command of the full protocol; the third
column gives the command that runs only the setting selected in the paper.

| Model (paper) | full protocol | selected setting only | output folder |
|---|---|---|---|
| LW-PVT (proposed) | `python run/lwpvt.py` | same | `LW-PVT` |
| MP | `python run/mp.py` | same | `MP` |
| MLP | `python run/nn_batch_grid.py --models MLP` | `... --models MLP --bs 8` | `MLP_b008` |
| ICNN-PGNN | `python run/nn_batch_grid.py --models ICNN-PGNN` | `... --bs 256` | `ICNN-PGNN_b256` |
| TS-PINN | `python run/nn_batch_grid.py --models TS-PINN` | `... --bs 16` | `TS-PINN_b016` |
| Carnot-PINN | `python run/nn_batch_grid.py --models Carnot-PINN` | `... --bs 256` | `Carnot-PINN_b256` |
| MonoPINN | `python run/monopinn.py` | `python run/monopinn.py --final_bs 0` | `MonoPINN_bfull` |
| E-ANN | `python run/eann.py` | `python run/eann.py --bs 0` | `E-ANN_bfull` |
| Hybrid TL | `python run/hybrid_tl.py` | `python run/hybrid_tl.py --bs 32` | `Hybrid-TL_b032` |
| PI-WGAN | `python run/piwgan.py` | `python run/piwgan.py --lam 32` | `PI-WGAN_lam32` |

`run/nn_batch_grid.py` calls `run/train_nn.py` once per run; `run/train_nn.py` can also be called
directly for one model, scenario and seed. Each run folder receives `metrics.json` (validation and
test RMSE, MAE, MAPE, R2 and the settings), `pred_test.csv` (inputs, measured and predicted COP of the
test data) and `best_model.pth` (model state and scalers). LW-PVT also writes `virtual_samples.csv`
(inputs, label, uncertainty sigma_j and share p_j of every virtual sample).

## Table 3

```
python results/main_table.py
```

reads all finished runs, selects the grid variant of each model, and writes
`output/main_table_mape.csv` and `output/main_table_r2.csv` (15 scenarios x 10 models, with the rows
Avg and Rank of the paper) and `output/main_table_selection.csv` (mean rank of every grid variant).
It also prints the Friedman test and the Wilcoxon signed-rank tests against LW-PVT with Holm
correction reported in Section 4.5.

## Notes

* The data and output folders can be relocated with the environment variables `LWPVT_DATA` and
  `LWPVT_OUTPUT`.
* The original study ran on Windows 11 with Python 3.11.14, PyTorch 2.9.1 and CUDA 12.8. Results
  on another platform may differ in the last digits because of floating-point differences between
  devices.
* E-ANN and Hybrid TL generate their synthetic data from manufacturer performance information in
  their papers. RP-1043 provides no such information, so in this study their synthetic data are
  labeled by the same Gordon-Ng regression that labels the virtual samples of LW-PVT (Section 4.3 of
  the paper).

## License

MIT License (see `LICENSE`). The ASHRAE RP-1043 data are subject to the license of ASHRAE.
