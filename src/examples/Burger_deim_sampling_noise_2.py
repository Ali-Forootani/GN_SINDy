#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 15:08:57 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:20:52 2025

Author: forootan (extended for noise robustness study)

Discovering Burgers' equation with GNSINDy + Noise Robustness

- Adds controlled Gaussian / Laplace noise at levels 1%, 5%, 10%
- Multiple seeds per level to assess stability
- Reports precision/recall (support recovery), coefficient L2 error, and variability
"""

import os, sys, shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, List

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
from GNSINDy.src.deepymod.data.burgers import burgers_delta_org
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
# Problem settings (Burgers)
# =============================================================================
# Library assumptions used in your plotting code:
IDX_UXX = 2      # index of u_xx in estimator history
IDX_UUX = 4      # index of u*u_x in estimator history

# Burgers true coefficients (u_t + u u_x = v u_xx)
VISCOSITY = 0.1
TRUE_COEFFS = {IDX_UXX: VISCOSITY, IDX_UUX: -1.0}   # other terms are 0

# Threshold for support detection from estimated coefficients
SUPPORT_THR = 1e-3

# =============================================================================
# Data generation with optional noise (Gaussian or Laplace)
# =============================================================================
def create_data(noise_type: str = "gaussian", noise_level: float = 0.1, seed: int = None):
    """
    Returns (coords, data) sampled by Q-DEIM from the analytic Burgers solution
    and then corrupted with optional noise.

    noise_level is relative to signal std (i.e., sigma = noise_level * std(data)).
    """
    if seed is not None:
        torch.manual_seed(seed)

    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v = VISCOSITY
    A = 1.0

    _, Exact = burgers_delta_org(x_o, t_o, v, A)

    # DEIM sampling on the clean field
    deim_instance = DEIM(Exact, 5, t_o, x_o, tolerance=1e-3, num_basis=1)
    S_s, T_s, U_s = deim_instance.execute()

    coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1)).float()
    data = torch.from_numpy(U_s.reshape(-1, 1)).float()

    # Add noise to observations (not to sampling indices)
    if noise_level > 0:
        sigma = noise_level * data.std()
        if noise_type.lower() == "gaussian":
            noise = sigma * torch.randn_like(data)
        elif noise_type.lower() == "laplace":
            noise = torch.distributions.Laplace(0.0, sigma).sample(data.shape)
        else:
            raise ValueError("noise_type must be 'gaussian' or 'laplace'")
        data = data + noise

    return coords, data

# =============================================================================
# Model / training configuration
# =============================================================================
def make_model(poly_order=2, diff_order=2) -> Tuple[DeepMoD, TrainTestPeriodic, torch.optim.Optimizer, Library1D]:
    network = NN(2, [64, 64, 64, 64], 1)
    library = Library1D(poly_order, diff_order)
    constraint = STRidgeCons()
    estimator = STRidge()

    model = DeepMoD(network, library, estimator, constraint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)
    sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
    return model, sparsity_scheduler, optimizer, library

def make_dataset_callable(noise_type, noise_level, seed, number_of_samples=100, normalize_coords=False, normalize_data=False, split=1.0, dev=device):
    """
    Returns train/test dataloaders created from a lambda wrapping `create_data` with chosen noise configuration.
    """
    ds = Dataset(
        lambda: create_data(noise_type=noise_type, noise_level=noise_level, seed=seed),
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
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1

def make_true_support(n_terms: int, true_coeffs_map: Dict[int, float], thr: float) -> np.ndarray:
    s = np.zeros(n_terms, dtype=int)
    for k, v in true_coeffs_map.items():
        if abs(v) > thr:
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
    )

    # Train
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        sparsity_scheduler,
        log_dir=log_dir,
        exp_ID=f"noise_{noise_type}_lvl_{noise_level:.3f}_seed_{seed}",
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
        "noise_type": noise_type,
        "noise_level": noise_level,
        "seed": seed,
        "num_samples": num_samples,
        "poly_order": poly_order,
        "diff_order": diff_order,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "coeff_l2": l2err,
        "coeffs": c_est.tolist()
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
    max_iterations: int = 25000,
    patience: int = 200,
    delta: float = 1e-4
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for nt in noise_types:
        for nl in noise_levels:
            for sd in seeds:
                print(f"=== Running {nt} noise, level={nl}, seed={sd} ===")
                res = run_single_experiment(
                    noise_type=nt,
                    noise_level=nl,
                    seed=sd,
                    num_samples=num_samples,
                    poly_order=poly_order,
                    diff_order=diff_order,
                    log_dir=out_dir,
                    max_iterations=max_iterations,
                    patience=patience,
                    delta=delta,
                )
                all_rows.append(res)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "noise_robustness_results.csv")
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
    lines.append("\\caption{Noise robustness of GN-SINDy on Burgers: precision/recall/F1 for support recovery and coefficient $\\ell_2$ error across multiple noise realizations. Mean$\\pm$std over seeds.}")
    lines.append("\\label{tab:noise_robustness}")
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
# Your original single-run visualization (kept)
# =============================================================================
def run_single_demo_and_plots(foldername: str, num_of_samples: int = 1000,
                              poly_order: int = 2, diff_order: int = 2):
    create_or_reset_directory(foldername)

    # Build dataset and loaders (demo without added noise)
    ds = Dataset(
        lambda: create_data(noise_type="gaussian", noise_level=0.00, seed=0),
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
        exp_ID="Test",
        write_iterations=25,
        max_iterations=25000,
        delta=1e-4,
        patience=200,
    )

    # Plot estimator history (your style)
    history = load_tensorboard(foldername)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    line_width = 2

    for history_key in history.keys():
        parts = history_key.split("_")
        if parts[0] == "estimator" and parts[3] == "0" and len(parts) > 4:
            if parts[5] != "2" and parts[5] != "4":
                axs[0].semilogx(history[history_key].loc[100:], linestyle="--", linewidth=line_width)
            elif parts[5] == "2":
                axs[0].semilogx(history[history_key].loc[100:], label=r"$u_{xx}$", linewidth=line_width + 1)
            elif parts[5] == "4":
                axs[0].semilogx(history[history_key].loc[100:], label=r"$uu_x$", linewidth=line_width + 1)

    axs[0].set_ylim([-2.5, 2.5])
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Coefficients")
    axs[0].grid(True)

    # Q-DEIM sampled points
    coords_demo = ds.get_coords().cpu()
    data_demo = ds.get_data().cpu()
    im = axs[1].scatter(coords_demo[:, 0], coords_demo[:, 1], c=data_demo[:, 0], marker="o",
                        label=r"Greedy samples: \texttt{Q-DEIM}", s=20)
    axs[1].set_xlabel(r'$t$')
    axs[1].set_ylabel(r'$x$', labelpad=0)
    axs[1].set_ylim([-8, 8])

    # Ground-truth field (dense)
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    coords_org, Exact = burgers_delta_org(x_o, t_o, VISCOSITY, 1.0)
    im = axs[2].scatter(coords_org[:, 0], coords_org[:, 1], c=Exact[:, :], marker="x", s=10)
    axs[2].set_xlabel(r'$t$')
    axs[2].set_ylabel(r'$x$', labelpad=0)
    axs[2].set_ylim([-8, 8])
    fig.colorbar(mappable=im)

    fig.legend(loc="center", ncol=4, bbox_to_anchor=(0.51, 1), bbox_transform=fig.transFigure,
               fontsize=20, frameon=True)

    plt.savefig(os.path.join(foldername,
        f"Burgers_coefficients_DEIM_sampling{num_of_samples}_poly_order_{poly_order}_diff_order_{diff_order}.png"),
        bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(foldername,
        f"Burgers_coefficients_DEIM_sampling{num_of_samples}_poly_order_{poly_order}_diff_order_{diff_order}.pdf"),
        bbox_inches='tight', dpi=600)
    plt.show()

    # Compact single-axes plot for coefficients vs iterations
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    line_width = 1
    for history_key in history.keys():
        parts = history_key.split("_")
        if parts[0] == "estimator" and parts[3] == "0" and len(parts) > 4:
            if parts[5] != "2" and parts[5] != "4":
                ax.semilogx(history[history_key].loc[100:], linestyle="--", linewidth=line_width)
            elif parts[5] == "2":
                ax.semilogx(history[history_key].loc[100:], label=r"$u_{xx}$", linewidth=line_width + 1)
            elif parts[5] == "4":
                ax.semilogx(history[history_key].loc[100:], label=r"$uu_x$", linewidth=line_width + 1)

    ax.set_ylim([-2.5, 2.5])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Coefficients")
    ax.grid(True)

    fig.legend(loc="center", ncol=4, bbox_to_anchor=(0.51, 1), bbox_transform=fig.transFigure,
               fontsize=10, frameon=True)

    plt.savefig(os.path.join(foldername,
        f"Burgers_coeff_iterations_DEIM_{num_of_samples}_poly_order_{poly_order}_diff_order_{diff_order}.png"),
        bbox_inches='tight', dpi=600)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # -----------------------------
    # Output/experiment folders
    # -----------------------------
    foldername = "./data/deepymod/burgers_noise/"
    create_or_reset_directory(foldername)

    # -----------------------------
    # (A) Your original demo run and plots (noise-free)
    #     Comment out if you only want the robustness grid.
    # -----------------------------
    run_single_demo_and_plots(foldername, num_of_samples=1000, poly_order=2, diff_order=2)

    # -----------------------------
    # (B) Noise robustness grid answering the reviewer
    # -----------------------------
    noise_types = ["gaussian", "laplace"]          # Gaussian and non-Gaussian
    noise_levels = [0.01, 0.05, 0.10]              # 1%, 5%, 10%
    seeds = list(range(5))                         # 5 realizations per level
    num_samples = 1000                              # same sampling budget
    poly_order, diff_order = 2, 2

    df = run_noise_grid(
        noise_types=noise_types,
        noise_levels=noise_levels,
        seeds=seeds,
        num_samples=num_samples,
        poly_order=poly_order,
        diff_order=diff_order,
        out_dir=foldername,
        max_iterations=25000,
        patience=200,
        delta=1e-4
    )

    # Summaries (stability across realizations)
    summary = summarize_results(df)
    print("\n=== Noise Robustness Summary (mean ± std over seeds) ===")
    print(summary.to_string(index=False))

    # Save summary CSV
    summary_csv = os.path.join(foldername, "noise_robustness_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Print LaTeX table for the paper
    print("\n\n=== LaTeX table (paste into manuscript) ===\n")
    print(latex_table(summary))
