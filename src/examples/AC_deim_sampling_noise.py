#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:13:03 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allen–Cahn equation discovery with GN-SINDy: Noise robustness study

- Loads AC.mat (fields: x, tt, uu)
- Q-DEIM sampling on the clean field
- Adds Gaussian / Laplace noise at levels 1%, 5%, 10% of signal std
- Trains GN-SINDy and reports precision/recall/F1 (support recovery) and coeff L2 error
- Saves per-run and summary CSVs; prints a LaTeX table

Author: forootani (adapted from Burgers noise study)
"""

import os, sys, shutil, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, List
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
# Allen–Cahn ground-truth mapping (adjust indices here if needed!)
# PDE: u_t = u_xx + u - u^3
# =============================================================================
# Default guesses based on your earlier logs: u -> 4, u^3 -> 12.
# u_xx depends on term ordering; start with 8 and adjust if necessary.
IDX_UXX = 8
IDX_U   = 4
IDX_U3  = 12

TRUE_COEFFS: Dict[int, float] = {
    IDX_UXX: +1.0,
    IDX_U:   +1.0,
    IDX_U3:  -1.0,
}

# Threshold for support detection from estimated coefficients
SUPPORT_THR = 1e-3

# =============================================================================
# Data loading (Allen–Cahn) + Q-DEIM sampling + optional noise
# =============================================================================
def _load_ac_mat(ac_path: str):
    """
    Expects variables:
      x   : (Nx, ) spatial grid
      tt  : (Nt, ) temporal grid
      uu  : (Nt, Nx) or (Nx, Nt); we convert to shape (Nx, Nt)
    """
    data = loadmat(ac_path)
    x = np.asarray(data["x"]).squeeze()          # (Nx,)
    t = np.asarray(data["tt"]).squeeze()         # (Nt,)
    U = np.real(np.asarray(data["uu"]))          # could be (Nt, Nx) or (Nx, Nt)

    # unify to U(x_i, t_j) with shape (Nx, Nt)
    if U.shape[0] == t.size and U.shape[1] == x.size:
        U = U.T  # make shape (Nx, Nt)
    assert U.shape == (x.size, t.size), "AC.mat: unexpected 'uu' shape"
    return x, t, U

def create_data_AC(noise_type: str = "gaussian", noise_level: float = 0.0, seed: int = None,
                   deim_n_d: int = 3, deim_num_basis: int = 1, deim_tol: float = 1e-7):
    """
    Returns (coords, data) sampled by Q-DEIM from the Allen–Cahn field
    and then corrupted with optional noise.

    noise_level is relative to signal std (i.e., sigma = noise_level * std(data)).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ac_path = os.path.join(root_dir, "src", "data", "AC.mat")
    x, t, U = _load_ac_mat(ac_path)

    # Q-DEIM sampling
    deim_instance = DEIM(U, deim_n_d, t, x, tolerance=deim_tol, num_basis=deim_num_basis)
    S_s, T_s, U_s = deim_instance.execute()  # S_s ~ x-samples, T_s ~ t-samples, U_s values

    coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1)).float()   # (N, 2) with (t, x)
    data   = torch.from_numpy(U_s.reshape(-1, 1)).float()              # (N, 1)

    # Add noise to observations (not to sampling indices)
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
# Model / training configuration
# =============================================================================
def make_model(poly_order=3, diff_order=3) -> Tuple[DeepMoD, TrainTestPeriodic, torch.optim.Optimizer, Library1D]:
    # 2D inputs (t,x) -> 1D output u
    network = NN(2, [64, 64, 64, 64], 1)
    library = Library1D(poly_order, diff_order)
    constraint = STRidgeCons()
    estimator  = STRidge()
    model = DeepMoD(network, library, estimator, constraint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)
    sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
    return model, sparsity_scheduler, optimizer, library

def make_dataset_callable(noise_type, noise_level, seed,
                          number_of_samples=200, normalize_coords=False, normalize_data=False, split=1.0, dev=device,
                          deim_n_d=3, deim_num_basis=1, deim_tol=1e-7):
    """
    Returns train/test dataloaders created from a lambda wrapping `create_data_AC` with chosen noise configuration.
    """
    ds = Dataset(
        lambda: create_data_AC(noise_type=noise_type, noise_level=noise_level, seed=seed,
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
    # model.estimator_coeffs() returns list per output; take first (single-output PDE)
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
                          deim_n_d: int = 3,
                          deim_num_basis: int = 1,
                          deim_tol: float = 1e-7,
                          max_iterations: int = 25000,
                          patience: int = 200,
                          delta: float = 1e-4) -> Dict:
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

    # Train
    exp_id = f"AC_noise_{noise_type}_lvl_{noise_level:.3f}_seed_{seed}"
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        sparsity_scheduler,
        log_dir=log_dir,
        exp_ID=exp_id,
        write_iterations=50,
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
        "equation": "Allen-Cahn",
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
    deim_n_d: int = 3,
    deim_num_basis: int = 1,
    deim_tol: float = 1e-7,
    max_iterations: int = 25000,
    patience: int = 200,
    delta: float = 1e-4
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for nt in noise_types:
        for nl in noise_levels:
            for sd in seeds:
                print(f"=== AC | {nt} noise, level={nl}, seed={sd} ===")
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
    csv_path = os.path.join(out_dir, "ac_noise_robustness_results.csv")
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
    lines.append("\\caption{Noise robustness of GN-SINDy on Allen--Cahn: precision/recall/F1 for support recovery and coefficient $\\ell_2$ error across multiple noise realizations. Mean$\\pm$std over seeds.}")
    lines.append("\\label{tab:ac_noise_robustness}")
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
# =============================================================================
def run_single_demo_and_plots(foldername: str, num_of_samples: int = 200,
                              poly_order: int = 3, diff_order: int = 2,
                              deim_n_d: int = 3, deim_num_basis: int = 1, deim_tol: float = 1e-7):
    create_or_reset_directory(foldername)

    # Build dataset and loaders (demo without added noise)
    ds = Dataset(
        lambda: create_data_AC(noise_type="gaussian", noise_level=0.00, seed=0,
                               deim_n_d=deim_n_d, deim_num_basis=deim_num_basis, deim_tol=deim_tol),
        preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": num_of_samples},
        device=device,
    )
    train_loader, test_loader = get_train_test_loader(ds, train_test_split=1.00)

    model, sparsity_scheduler, optimizer, _ = make_model(poly_order, diff_order)

    # Train demo
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        sparsity_scheduler,
        log_dir=foldername,
        exp_ID="AC_Demo",
        write_iterations=25,
        max_iterations=25000,
        delta=1e-4,
        patience=200,
    )

    # Plot estimator history: highlight u, u^3, u_xx (indices above)
    history = load_tensorboard(foldername)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    for history_key in history.keys():
        parts = history_key.split("_")
        if parts[0] == "estimator" and parts[3] == "0" and len(parts) > 5:
            idx = parts[5]
            y = history[history_key].loc[100:]
            if idx == str(IDX_U):
                ax.semilogx(y, label=r"$u$", linewidth=2.0)
            elif idx == str(IDX_U3):
                ax.semilogx(y, label=r"$u^3$", linewidth=2.0)
            elif idx == str(IDX_UXX):
                ax.semilogx(y, label=r"$u_{xx}$", linewidth=2.0)
            else:
                ax.semilogx(y, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Coefficients")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(foldername, "AC_coeffs_iterations_demo.png"), dpi=300)
    plt.show()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # -----------------------------
    # Output/experiment folders
    # -----------------------------
    out_dir = "./data/deepymod/allen_cahn_noise/"
    create_or_reset_directory(out_dir)

    # -----------------------------
    # (A) Optional: quick demo (no noise) with coefficient traces
    # -----------------------------
    run_single_demo_and_plots(out_dir, num_of_samples=300, poly_order=3, diff_order=3,
                              deim_n_d=3, deim_num_basis=1, deim_tol=1e-7)

    # -----------------------------
    # (B) Noise robustness grid
    # -----------------------------
    noise_types  = ["gaussian", "laplace"]      # will be labeled as Gaussian / Laplace in CSV
    noise_levels = [0.01, 0.05, 0.10]           # 1%, 5%, 10%
    seeds        = list(range(5))               # 5 realizations per level
    num_samples  = 500                           # Q-DEIM budget per run
    poly_order, diff_order = 3, 3

    df = run_noise_grid(
        noise_types=noise_types,
        noise_levels=noise_levels,
        seeds=seeds,
        num_samples=num_samples,
        poly_order=poly_order,
        diff_order=diff_order,
        out_dir=out_dir,
        deim_n_d=3,
        deim_num_basis=1,
        deim_tol=1e-7,
        max_iterations=25000,
        patience=200,
        delta=1e-4
    )

    # Summaries (stability across realizations)
    summary = summarize_results(df)
    print("\n=== Allen–Cahn: Noise Robustness Summary (mean ± std over seeds) ===")
    print(summary.to_string(index=False))

    # Save summary CSV
    summary_csv = os.path.join(out_dir, "ac_noise_robustness_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Print LaTeX table for the paper
    print("\n\n=== LaTeX table (paste into manuscript) ===\n")
    print(latex_table(summary))
