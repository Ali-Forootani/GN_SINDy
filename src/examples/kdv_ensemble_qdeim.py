#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:10:36 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble Q-DEIM for KdV discovery (GN-SINDy)
- Sweeps over (n_d, tolerance, num_basis, num_samples)
- Repeats over seeds
- Votes on active terms across runs (majority or custom threshold)
- Aggregates coefficients (median over runs where term is active)
"""

import os, sys, time, json, random
import numpy as np
import torch
import pandas as pd

# ----------------- Paths & imports (assumes your project layout) -----------------
cwd = os.getcwd()
sys.path.append(cwd)



import numpy as np
import torch
import sys
import os
import scipy.io as sio

cwd = os.getcwd()
#sys.path.append(cwd + '/my_directory')
sys.path.append(cwd)


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(2)




from scipy.io import loadmat
import scipy.io as sio
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import STRidge
from GNSINDy.src.deepymod.model.constraint import STRidgeCons
from GNSINDy.src.deepymod.training import train
from GNSINDy.src.deepymod.training.sparsity_scheduler import TrainTestPeriodic
from GNSINDy.src.deepymod.data.DEIM_class import DEIM


# ----------------- Your DEIM class is assumed available in scope -----------------
# class DEIM: ...  (use the one you pasted)

# ----------------- Device -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Ensemble] device = {device}")

# ----------------- Library names for poly_order=2, diff_order=3 -----------------
# Order matches DeepMoD's Library1D for these settings (common layout):
# [1, ux, uxx, uxxx, u, u*ux, u*uxx, u*uxxx, u^2, u^2*ux, u^2*uxx, u^2*uxxx]
TERM_NAMES = [
    "1", "u_x", "u_xx", "u_xxx",
    "u", "u*u_x", "u*u_xx", "u*u_xxx",
    "u^2", "u^2*u_x", "u^2*u_xx", "u^2*u_xxx"
]

# ----------------- Grids & meta -----------------
# Keep tolerance <= 1e-5 and not ultra-low. Adjust as needed.
GRID = {
    "n_d":        [2, 3],
    "tolerance":  [3e-5, 1e-5, 5e-6, 2e-6],
    "num_basis":  [4, 8, 10, 12],
    "num_samples":[900, 1500, 3000]  # Subsample from greedy pool for training
}
SEEDS = [0, 1, 2]     # repeat per configuration
SUPPORT_TAU = 1e-3    # |coef| >= tau => active
VOTE_THRESHOLD = 0.6  # fraction of runs in which a term must be active

# Output
OUT_DIR = "./data/deepymod/KDV_ensemble"
os.makedirs(OUT_DIR, exist_ok=True)
RUNS_CSV = os.path.join(OUT_DIR, "ensemble_runs_kdv.csv")
VOTE_CSV = os.path.join(OUT_DIR, "ensemble_vote_kdv.csv")

# ----------------- Data loader factory (with DEIM knobs) -----------------
def load_kdv_mat(root_dir):
    # matches your paths in earlier script
    data = sio.loadmat(os.path.join(root_dir, "src/data/kdv.mat"))
    t_o = data["t"].flatten()[0:201, None]    # (201,1)
    x_o = data["x"].flatten()[:, None]        # (512,1)
    U   = np.real(data["usol"])               # (512,201)
    return U, t_o.squeeze(), x_o.squeeze()

def create_data_factory(root_dir, n_d, tolerance, num_basis):
    # Returns a no-arg function that Dataset will call
    U, t_vec, x_vec = load_kdv_mat(root_dir)

    def _create():
        deim = DEIM(U, n_d=n_d, t_o=t_vec, x_o=x_vec,
                    tolerance=float(tolerance), num_basis=int(num_basis))
        S_s, T_s, U_s = deim.execute()
        coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1)).float()
        data = torch.from_numpy(U_s.reshape(-1,1)).float()
        return coords, data
    return _create

# ----------------- Model factory -----------------
def make_model():
    # KdV config you used
    poly_order, diff_order = 2, 3
    net = NN(2, [32, 32, 32, 32], 1)
    library = Library1D(poly_order, diff_order)
    estimator = STRidge()         # non-differentiable sparse estimator
    constraint = STRidgeCons()    # refinement on constrained LS
    model = DeepMoD(net, library, estimator, constraint).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           betas=(0.99, 0.99), amsgrad=True, lr=1e-3)
    sched = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-4)
    return model, opt, sched, poly_order, diff_order

def get_coeffs_and_mask(model):
    # Robust fetch of coefficients and active mask for the first (and only) output
    with torch.no_grad():
        coeffs = model.estimator_coeffs()[0].detach().cpu().numpy().ravel()
    # Use magnitude threshold for support decision (also report STRidge mask if present)
    support = (np.abs(coeffs) >= SUPPORT_TAU)
    return coeffs, support

# ----------------- Single run -----------------
def run_single(root_dir, seed, n_d, tolerance, num_basis, num_samples):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    create_data = create_data_factory(root_dir, n_d, tolerance, num_basis)

    # Build dataset from greedy pool then subsample
    dataset = Dataset(
        create_data,
        preprocess_kwargs = {
            "noise_level": 0.0,
            "normalize_coords": False,
            "normalize_data": False,
        },
        subsampler = Subsample_random,
        subsampler_kwargs = {"number_of_samples": int(num_samples)},
        device = device
    )
    train_loader, test_loader = get_train_test_loader(dataset, train_test_split=0.99)

    model, opt, sched, poly_order, diff_order = make_model()

    # Short label dir for TB logs—keep separate per run
    log_dir = os.path.join(OUT_DIR, "tb",
                           f"nd{n_d}-tol{tolerance:.0e}-nb{num_basis}-N{num_samples}-s{seed}")
    os.makedirs(log_dir, exist_ok=True)

    t0 = time.time()
    train(model,
          train_loader,
          test_loader,
          opt,
          sched,
          log_dir=log_dir,
          exp_ID="EnsembleQDEIM",
          write_iterations=250,
          max_iterations=25000,
          delta=1e-4,
          patience=1000)
    runtime = time.time() - t0

    coeffs, support = get_coeffs_and_mask(model)

    # Package row
    row = {
        "seed": seed,
        "n_d": n_d,
        "tolerance": tolerance,
        "num_basis": num_basis,
        "num_samples": num_samples,
        "runtime_sec": runtime,
        "coeffs": json.dumps(list(map(float, coeffs))),
        "active_idx": json.dumps([int(i) for i in np.where(support)[0]]),
    }
    return row, coeffs, support

# ----------------- Ensemble driver -----------------
def product(*lists):
    if not lists:
        yield ()
    else:
        for x in lists[0]:
            for y in product(*lists[1:]):
                yield (x,) + y

def run_ensemble(root_dir):
    rows = []
    activations = []   # list of boolean arrays per run
    coeff_bank = []    # list of coeff arrays per run

    for (n_d, tol, nb, N) in product(GRID["n_d"], GRID["tolerance"], GRID["num_basis"], GRID["num_samples"]):
        for seed in SEEDS:
            print(f"[RUN] seed={seed} n_d={n_d} tol={tol:.0e} num_basis={nb} N={N}")
            try:
                row, coeffs, support = run_single(root_dir, seed, n_d, tol, nb, N)
                rows.append(row)
                activations.append(support.astype(np.int8))
                coeff_bank.append(coeffs)
            except Exception as e:
                print(f"[WARN] run failed: {e}")
                rows.append({
                    "seed": seed, "n_d": n_d, "tolerance": tol, "num_basis": nb, "num_samples": N,
                    "runtime_sec": np.nan, "coeffs": json.dumps([]), "active_idx": json.dumps([]),
                    "error": str(e)
                })

    df = pd.DataFrame(rows)
    df.to_csv(RUNS_CSV, index=False)
    print(f"[Ensemble] Wrote per-run results to {RUNS_CSV}")

    # Voting
    if len(activations) == 0:
        print("[Ensemble] No successful runs, skipping voting.")
        return

    A = np.vstack(activations)              # (R, D)
    C = np.vstack(coeff_bank)               # (R, D)
    R, D = A.shape

    freq = A.mean(axis=0)                   # fraction of runs active
    voted = (freq >= VOTE_THRESHOLD)        # final support set
    voted_idx = np.where(voted)[0].tolist()

    # Aggregate coefficients: median over runs where term active; else 0
    agg_coeffs = np.zeros(D, dtype=float)
    for j in range(D):
        mask = (A[:, j] == 1)
        if mask.any():
            agg_coeffs[j] = np.median(C[mask, j])
        else:
            agg_coeffs[j] = 0.0

    # Summaries with names
    vote_rows = []
    for j in range(D):
        vote_rows.append({
            "term_idx": j,
            "term_name": TERM_NAMES[j] if j < len(TERM_NAMES) else f"phi_{j}",
            "freq_active": float(freq[j]),
            "voted_active": bool(voted[j]),
            "coef_median": float(agg_coeffs[j]),
        })
    df_vote = pd.DataFrame(vote_rows)
    df_vote.to_csv(VOTE_CSV, index=False)
    print(f"[Ensemble] Wrote vote summary to {VOTE_CSV}")

    # Pretty print final PDE
    print("\n========== Ensemble Voted PDE (threshold = {:.0%}) ==========".format(VOTE_THRESHOLD))
    for j in np.where(voted)[0]:
        name = TERM_NAMES[j] if j < len(TERM_NAMES) else f"phi_{j}"
        print(f"{name:>10s}: {agg_coeffs[j]: .6f}  (freq={freq[j]:.2f})")
    print("=============================================================\n")

if __name__ == "__main__":
    # Your project places kdv.mat at <root_dir>/src/data/kdv.mat.
    # From a script under e.g. <root_dir>/src/examples, go up two levels:
    def setting_directory(depth):
        current_dir = os.path.abspath(os.getcwd())
        root_dir = current_dir
        for _ in range(depth):
            root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
            sys.path.append(os.path.dirname(root_dir))
        return root_dir
    root_dir = setting_directory(2)
    run_ensemble(root_dir)
