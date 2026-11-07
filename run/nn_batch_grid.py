# -*- coding: utf-8 -*-
"""Batch-size grid of the neural-network comparison models (MLP, ICNN-PGNN, TS-PINN, Carnot-PINN).

The papers of the comparison models do not report a batch size for this task, so each model was
trained with seven batch sizes (full batch, 8, 16, 32, 64, 128, 256) and the batch size with the best
mean rank over the 15 scenarios was kept (results/main_table.py performs this selection). To keep the
number of parameter updates comparable, the number of epochs is 1,000 for the full batch and
max(10, round(100 * batch / 64)) otherwise (8 -> 12, 16 -> 25, 32 -> 50, 64 -> 100, 128 -> 200,
256 -> 400). Carnot-PINN keeps the 1,200 epochs of its paper with early stopping (patience 50).

Usage (from the repository root):
    python run/nn_batch_grid.py [--models MLP,ICNN-PGNN,TS-PINN,Carnot-PINN] [--bs 0,8,16,32,64,128,256]
                                [--seeds 10] [--scenarios ...] [--dry]
Selected settings of the paper only:
    python run/nn_batch_grid.py --models MLP --bs 8
    python run/nn_batch_grid.py --models ICNN-PGNN --bs 256
    python run/nn_batch_grid.py --models TS-PINN --bs 16
    python run/nn_batch_grid.py --models Carnot-PINN --bs 256
Output: output/<model>_b<bs>/<scenario>/s<seed>/ (full batch: <model>_bfull). Finished runs are skipped.
"""
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lwpvt.paths import SCENARIOS, SEEDS, run_dir, batch_suffix

PY = sys.executable
MODELS = ["MLP", "ICNN-PGNN", "TS-PINN", "Carnot-PINN"]
BATCHES = [0, 8, 16, 32, 64, 128, 256]


def epochs_of(model, bs):
    if model == "Carnot-PINN":
        return 1200
    return 1000 if bs == 0 else max(10, round(100 * bs / 64))


argv = sys.argv[1:]
dry = "--dry" in argv
models = argv[argv.index("--models") + 1].split(",") if "--models" in argv else MODELS
batches = [int(x) for x in argv[argv.index("--bs") + 1].split(",")] if "--bs" in argv else BATCHES
n_seeds = int(argv[argv.index("--seeds") + 1]) if "--seeds" in argv else len(SEEDS)
scenarios = argv[argv.index("--scenarios") + 1].split(",") if "--scenarios" in argv else SCENARIOS
seeds = SEEDS[:n_seeds]
for m in models:
    assert m in MODELS, f"unknown model {m}; choose from {MODELS}"

jobs = [(bs, m, sc, sd) for bs in batches for m in models for sc in scenarios for sd in seeds]
print(f"batch grid: {len(batches)} batch sizes {batches} x {len(models)} models x {len(scenarios)} scenarios "
      f"x {len(seeds)} seeds = {len(jobs)} runs", flush=True)
done = skip = fail = 0
t0 = time.time()
for i, (bs, m, sc, sd) in enumerate(jobs, 1):
    name = m + batch_suffix(bs)
    if os.path.exists(os.path.join(run_dir(name, sc, sd), "metrics.json")):
        skip += 1
        continue
    cmd = [PY, os.path.join(ROOT, "run", "train_nn.py"), "--model", m, "--scenario", sc, "--seed", str(sd),
           "--batch_size", str(bs), "--epochs", str(epochs_of(m, bs)), "--out_name", name]
    if m == "Carnot-PINN":
        cmd += ["--patience", "50"]
    print(f"[{i}/{len(jobs)}] {sc} {name} seed {sd}  ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    if dry:
        print("    " + " ".join(cmd))
        continue
    r = subprocess.run(cmd, cwd=ROOT)
    done += (r.returncode == 0)
    fail += (r.returncode != 0)
print(f"\nfinished {done} | skipped {skip} | failed {fail} | {(time.time() - t0) / 60:.1f} min", flush=True)
