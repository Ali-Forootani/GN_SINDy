#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KdV equation discovery with GN-SINDy: Noise robustness study (hyperparams aligned)

- Loads kdv.mat (fields: x, t, usol)
- Q-DEIM sampling on the clean field
- Adds Gaussian / Laplace noise at levels 1%, 5%, 10% of signal std
- Trains GN-SINDy and reports precision/recall/F1 (support recovery) and coeff L2 error
- Saves per-run and summary CSVs; prints a LaTeX table

Hyperparameters are IDENTICAL to the non-noise script:
  DEIM: n_d=2, num_basis=1, tolerance=1e-5
  Network: NN(2, [32,32,32,32], 1)
  Estimator/Constraint: STRidge + STRidgeCons
  Optimizer: Adam(lr=1e-3, betas=(0.99,0.99), amsgrad=True)
  Scheduler: TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
  Train loop: max_iterations=25000, delta=1e-4, patience=1000, write_iterations=25
  Library: Library1D(poly_order=2, diff_order=3)

Author: forootani
"""

import os, sys, json
from typing import Dict, Tuple, List

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# -----------------------------
# Device and determinism
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

np.random.seed(42)
torch.manual_seed(50)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------
# Path setup
# -----------------------------
def setting_directory(depth: int):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(2)
cwd = os.getcwd()
sys.path.append(cwd)

# -----------------------------
# GNSINDy / DeePyMoD imports
# -----------------------------
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.model.constraint import STRidgeCons
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import STRidge
from GNSINDy.src.deepymod.training import train
from GNSINDy.src.deepymod.training.sparsity_scheduler import TrainTestPeriodic
from GNSINDy.src.deepymod.data.DEIM_class import DEIM
from GNSINDy.src.deepymod.utils.utilities import create_or_reset_directory
from GNSINDy.src.deepymod.analysis import load_tensorboard

# =============================================================================
# FIXED hyperparameters (identical to the non-noise script)
# =============================================================================
POLY_ORDER = 2
DIFF_ORDER = 3

DEIM_N_D = 2
DEIM_NUM_BASIS = 1
DEIM_TOL = 1e-5

NET_LAYERS = [32, 32, 32, 32]
OPT_LR = 1e-3
OPT_BETAS = (0.99, 0.99)

SCHED_PERIODICITY = 50
SCHED_PATIENCE = 1000
SCHED_DELTA = 1e-5

TRAIN_MAX_ITERS = 25000
TRAIN_DELTA = 1e-4
TRAIN_PATIENCE = 1000
WRITE_ITERS = 25

# =============================================================================
# KdV ground-truth mapping (adjust indices here if needed!)
# PDE: u_t = -6 u u_x - u_{xxx}
# Library1D(poly=2, diff=3) order typically matches (1, u, u^2, u_x, u_xx, u_xxx, u u_x, ...).
# In your prior code you highlighted indices 3 -> u_{xxx}, 5 -> u u_x. Keep these unless your library differs.
# =============================================================================
IDX_UXXX = 3
IDX_U_UX = 5
TRUE_COEFFS: Dict[int, float] = {
    IDX_UXXX: -1.0,   # coefficient of u_{xxx}
    IDX_U_UX: -6.0,   # coefficient of u u_x
}
SUPPORT_THR = 1e-3

# =============================================================================
# Data loading (KdV) + Q-DEIM sampling + optional noise
# =============================================================================
def _load_kdv_mat(kdv_path: str):
    """
    Expects variables:
      x   : (Nx, ) spatial grid
      t   : (Nt, ) temporal grid
      usol: (Nx, Nt) or (Nt, Nx); we convert to shape (Nx, Nt)
    """
    data = loadmat(kdv_path)
    x = np.asarray(data["x"]).squeeze()          # (Nx,)
    t = np.asarray(data["t"]).squeeze()          # (Nt,)
    U = np.real(np.asarray(data["usol"]))        # could be (Nt, Nx) or (Nx, Nt)
    if U.shape[0] == t.size and U.shape[1] == x.size:
        U = U.T  # make shape (Nx, Nt)
    assert U.shape == (x.size, t.size), "kdv.mat: unexpected 'usol' shape"
    return x, t, U

def create_data_KDV(noise_type: str = "gaussian", noise_level: float = 0.0, seed: int = None,
                    deim_n_d: int = DEIM_N_D, deim_num_basis: int = DEIM_NUM_BASIS, deim_tol: float = DEIM_TOL):
    """
    Returns (coords, data) sampled by Q-DEIM from the KdV field
    and then corrupted with optional noise.

    noise_level is relative to signal std (i.e., sigma = noise_level * std(data)).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    kdv_path = os.path.join(root_dir, "src", "data", "kdv.mat")
    x, t, U = _load_kdv_mat(kdv_path)

    # Q-DEIM sampling (fixed hyperparams)
    deim_instance = DEIM(U, deim_n_d, t, x, tolerance=deim_tol, num_basis=deim_num_basis)
    S_s, T_s, U_s = deim_instance.execute()  # S_s ~ x-samples, T_s ~ t-samples, U_s values

    coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1)).float()   # (N, 2) with (t, x)
    data   = torch.from_numpy(U_s.reshape(-1, 1)).float()              # (N, 1)

    # Add noise to observations
    if noise_level > 0:
        sigma = noise_level * data.std()
        nt = noise_type.strip().lower()
        if nt == "gaussian":
            noise = sigma * torch.randn_like(data)
        elif nt == "laplace":
            noise = torch.distributions.Laplace(0.0, sigma).sample(data.shape)
        else:
            raise ValueError("noise_type must be 'gaussian' or 'laplace'")
        data = data + noise

    return coords, data

# =============================================================================
# Model / training configuration (fixed to match non-noise script)
# =============================================================================
def make_model(poly_order=POLY_ORDER, diff_order=DIFF_ORDER) -> Tuple[DeepMoD, TrainTestPeriodic, torch.optim.Optimizer, Library1D]:
    network = NN(2, NET_LAYERS, 1)
    library = Library1D(poly_order, diff_order)
    constraint = STRidgeCons()
    estimator  = STRidge()
    model = DeepMoD(network, library, estimator, constraint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=OPT_BETAS, amsgrad=True, lr=OPT_LR)
    sparsity_scheduler = TrainTestPeriodic(periodicity=SCHED_PERIODICITY, patience=SCHED_PATIENCE, delta=SCHED_DELTA)
    return model, sparsity_scheduler, optimizer, library

def make_dataset_callable(noise_type, noise_level, seed,
                          number_of_samples=900,  # match your non-noise plots
                          normalize_coords=False, normalize_data=False, split=1.0, dev=device,
                          deim_n_d=DEIM_N_D, deim_num_basis=DEIM_NUM_BASIS, deim_tol=DEIM_TOL):
    """
    Returns train/test dataloaders created from a lambda wrapping `create_data_KDV` with chosen noise configuration.
    """
    ds = Dataset(
        lambda: create_data_KDV(noise_type=noise_type, noise_level=noise_level, seed=seed,
                                deim_n_d=deim_n_d, deim_num_basis=deim_num_basis, deim_tol=deim_tol),
        preprocess_kwargs={
            "noise_level": 0.00,            # we already add noise ourselves
            "normalize_coords": normalize_coords,
            "normalize_data": normalize_data,
        },
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": number_of_samples},
        device=dev,
    )
    train_loader, test_loader = get_train_test_loader(ds, train_test_split=split)
    return ds, train_loader, test_loader

# =============================================================================
# Metrics (support recovery and coefficient error)
# =============================================================================
def coeff_vector_from_model(model: DeepMoD) -> np.ndarray:
    c = model.estimator_coeffs()[0].detach().cpu().numpy().reshape(-1)
    return c

def support_from_coeffs(c: np.ndarray, thr: float) -> np.ndarray:
    return (np.abs(c) > thr).astype(int)

def precision_recall_f1(pred_support: np.ndarray, true_support: np.ndarray) -> Tuple[float, float, float]:
    tp = int(np.sum((pred_support == 1) & (true_support == 1)))
    fp = int(np.sum((pred_support == 1) & (true_support == 0)))
    fn = int(np.sum((pred_support == 0) & (true_support == 1)))
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

def make_true_support(n_terms: int, true_coeffs_map: Dict[int, float], thr: float) -> np.ndarray:
    s = np.zeros(n_terms, dtype=int)
    for k, v in true_coeffs_map.items():
        if k < n_terms and abs(v) > thr:
            s[k] = 1
    return s

def coeff_error_l2(est: np.ndarray, true_coeffs_map: Dict[int, float]) -> float:
    errs = []
    for k, v_true in true_coeffs_map.items():
        v_est = est[k] if k < len(est) else 0.0
        errs.append((v_est - v_true) ** 2)
    return float(np.sqrt(np.sum(errs)))

# =============================================================================
# Single run (one noise type/level/seed)
# =============================================================================
def run_single_experiment(noise_type: str,
                          noise_level: float,
                          seed: int,
                          num_samples: int,
                          poly_order: int,
                          diff_order: int,
                          log_dir: str,
                          deim_n_d: int = DEIM_N_D,
                          deim_num_basis: int = DEIM_NUM_BASIS,
                          deim_tol: float = DEIM_TOL,
                          max_iterations: int = TRAIN_MAX_ITERS,
                          patience: int = TRAIN_PATIENCE,
                          delta: float = TRAIN_DELTA) -> Dict:
    model, sparsity_scheduler, optimizer, library = make_model(poly_order, diff_order)

    ds, train_loader, test_loader = make_dataset_callable(
        noise_type=noise_type,
        noise_level=noise_level,
        seed=seed,
        number_of_samples=num_samples,
        normalize_coords=False,
        normalize_data=False,
        split=1.0,
        dev=device,
        deim_n_d=deim_n_d,
        deim_num_basis=deim_num_basis,
        deim_tol=deim_tol,
    )

    # Train (fixed settings)
    exp_id = f"KDV_noise_{noise_type}_lvl_{noise_level:.3f}_seed_{seed}"
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        sparsity_scheduler,
        log_dir=log_dir,
        exp_ID=exp_id,
        write_iterations=WRITE_ITERS,
        max_iterations=max_iterations,
        delta=delta,
        patience=patience,
    )

    # Extract coefficients & metrics
    c_est = coeff_vector_from_model(model)
    n_terms = len(c_est)
    pred_support = support_from_coeffs(c_est, SUPPORT_THR)
    true_support = make_true_support(n_terms, TRUE_COEFFS, SUPPORT_THR)

    prec, rec, f1 = precision_recall_f1(pred_support, true_support)
    l2err = coeff_error_l2(c_est, TRUE_COEFFS)

    result = {
        "equation": "KdV",
        "noise_type": "Gaussian" if noise_type.lower()=="gaussian" else "Laplace",
        "noise_level": noise_level,
        "seed": seed,
        "num_samples": num_samples,
        "poly_order": poly_order,
        "diff_order": diff_order,
        "deim_n_d": deim_n_d,
        "deim_num_basis": deim_num_basis,
        "deim_tol": deim_tol,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "coeff_l2": l2err,
        "coeff_vector_json": json.dumps(list(map(float, c_est))),
    }
    return result

# =============================================================================
# Grid over noise levels and seeds (stability study)
# =============================================================================
def run_noise_grid(
    noise_types: List[str],
    noise_levels: List[float],
    seeds: List[int],
    num_samples: int,
    poly_order: int,
    diff_order: int,
    out_dir: str,
    deim_n_d: int = DEIM_N_D,
    deim_num_basis: int = DEIM_NUM_BASIS,
    deim_tol: float = DEIM_TOL,
    max_iterations: int = TRAIN_MAX_ITERS,
    patience: int = TRAIN_PATIENCE,
    delta: float = TRAIN_DELTA
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for nt in noise_types:
        for nl in noise_levels:
            for sd in seeds:
                print(f"=== KdV | {nt} noise, level={nl}, seed={sd} ===")
                res = run_single_experiment(
                    noise_type=nt,
                    noise_level=nl,
                    seed=sd,
                    num_samples=num_samples,
                    poly_order=poly_order,
                    diff_order=diff_order,
                    log_dir=out_dir,
                    deim_n_d=deim_n_d,
                    deim_num_basis=deim_num_basis,
                    deim_tol=deim_tol,
                    max_iterations=max_iterations,
                    patience=patience,
                    delta=delta,
                )
                all_rows.append(res)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "kdv_noise_robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")
    return df

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["noise_type", "noise_level"]).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        coeff_l2_mean=("coeff_l2", "mean"),
        coeff_l2_std=("coeff_l2", "std")
    ).reset_index().sort_values(["noise_type", "noise_level"])
    return agg

def latex_table(df_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Noise robustness of GN-SINDy on KdV: precision/recall/F1 for support recovery and coefficient $\\ell_2$ error across multiple noise realizations. Mean$\\pm$std over seeds.}")
    lines.append("\\label{tab:kdv_noise_robustness}")
    lines.append("\\begin{tabular}{l c c c c c}")
    lines.append("\\toprule")
    lines.append("Noise & Level & Precision & Recall & F1 & $\\ell_2$ coef. err.\\\\")
    lines.append("\\midrule")
    for _, r in df_summary.iterrows():
        nt = r["noise_type"]
        lvl = r["noise_level"]
        def fmt(m,s): return f"{m:.3f}$\\pm${(0.0 if pd.isna(s) else s):.3f}"
        row = f"{nt} & {lvl:.2f} & {fmt(r['precision_mean'],r['precision_std'])} & {fmt(r['recall_mean'],r['recall_std'])} & {fmt(r['f1_mean'],r['f1_std'])} & {fmt(r['coeff_l2_mean'],r['coeff_l2_std'])} \\\\"
        lines.append(row)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# =============================================================================
# Optional: a compact demo plot (noise-free) to visualize sampling and coeff traces
# (kept here, but uses the SAME hyperparameters; safe to comment out if not needed)
# =============================================================================
def run_single_demo_and_plots(foldername: str, num_of_samples: int = 900,
                              poly_order: int = POLY_ORDER, diff_order: int = DIFF_ORDER,
                              deim_n_d: int = DEIM_N_D, deim_num_basis: int = DEIM_NUM_BASIS, deim_tol: float = DEIM_TOL):
    create_or_reset_directory(foldername)

    ds = Dataset(
        lambda: create_data_KDV(noise_type="gaussian", noise_level=0.00, seed=0,
                                deim_n_d=deim_n_d, deim_num_basis=deim_num_basis, deim_tol=deim_tol),
        preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": num_of_samples},
        device=device,
    )
    train_loader, test_loader = get_train_test_loader(ds, train_test_split=1.00)

    model, sparsity_scheduler, optimizer, _ = make_model(poly_order, diff_order)

    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        sparsity_scheduler,
        log_dir=foldername,
        exp_ID="KDV_Demo",
        write_iterations=WRITE_ITERS,
        max_iterations=25,   # quick visual demo only
        delta=TRAIN_DELTA,
        patience=TRAIN_PATIENCE,
    )

    history = load_tensorboard(foldername)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for history_key in history.keys():
        parts = history_key.split("_")
        if parts[0] == "estimator" and parts[3] == "0" and len(parts) > 5:
            idx = parts[5]
            y = history[history_key].loc[100:]
            if idx == str(IDX_U_UX):
                ax.semilogx(y, label=r"$u\,u_x$", linewidth=2.0)
            elif idx == str(IDX_UXXX):
                ax.semilogx(y, label=r"$u_{xxx}$", linewidth=2.0)
            else:
                ax.semilogx(y, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Coefficients")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(foldername, "KDV_coeffs_iterations_demo.png"), dpi=300)
    plt.show()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Output/experiment folders
    out_dir = "./data/deepymod/KDV_noise_2/"  # placed under KDV to keep context together
    create_or_reset_directory(out_dir)

    # (A) Optional: quick demo (no noise) with coefficient traces (uses same hyperparams)
    # Comment out if not needed.
    # run_single_demo_and_plots(out_dir, num_of_samples=900)

    # (B) Noise robustness grid
    noise_types  = ["gaussian", "laplace"]
    noise_levels = [0.01]      # 1%, 5%, 10%
    seeds        = list(range(1))          # 5 realizations per level
    num_samples  = 900                     # match non-noise plots/sample budget

    df = run_noise_grid(
        noise_types=noise_types,
        noise_levels=noise_levels,
        seeds=seeds,
        num_samples=num_samples,
        poly_order=POLY_ORDER,
        diff_order=DIFF_ORDER,
        out_dir=out_dir,
        deim_n_d=DEIM_N_D,
        deim_num_basis=DEIM_NUM_BASIS,
        deim_tol=DEIM_TOL,
        max_iterations=TRAIN_MAX_ITERS,
        patience=TRAIN_PATIENCE,
        delta=TRAIN_DELTA
    )

    # Summaries (stability across realizations)
    summary = summarize_results(df)
    print("\n=== KdV: Noise Robustness Summary (mean ± std over seeds) ===")
    print(summary.to_string(index=False))

    # Save summary CSV
    summary_csv = os.path.join(out_dir, "kdv_noise_robustness_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Print LaTeX table for the paper
    print("\n\n=== LaTeX table (paste into manuscript) ===\n")
    print(latex_table(summary))
