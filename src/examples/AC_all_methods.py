#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:35:21 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, csv, time, random
from functools import partial
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat


import sys
import os

# ---------------- Device & determinism ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
np.random.seed(42)
torch.manual_seed(50)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir
root_dir = setting_directory(2)




# ---------------- Root dir (adjust if needed) ----------------
# If you already have root_dir in your session, keep it; else define here:
try:
    root_dir
except NameError:
    root_dir = os.getcwd()  # change to your repo root if desired

# ---------------- DeepMoD / GNSINDy ----------------
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import STRidge
from GNSINDy.src.deepymod.model.constraint import STRidgeCons
from GNSINDy.src.deepymod.training.sparsity_scheduler import TrainTestPeriodic
from GNSINDy.src.deepymod.training import train
from GNSINDy.src.deepymod.utils.utilities import create_or_reset_directory

# Samplers
from GNSINDy.src.deepymod.data.DEIM_class import DEIM, RandDEIM, LDEIM, CURSelector

# ============================================================
# Config: Library & Truth (Allen–Cahn: u_t = nu*u_xx + u - u^3)
# ============================================================
POLY_ORDER = 3
DIFF_ORDER = 3
NU_AC = 0.01  # used for L2 error target

# Matches your LaTeX dictionary order exactly:
ORDER = [
    "1", "u_x", "u_xx", "u_xxx",
    "u", "u*u_x", "u*u_xx", "u*u_xxx",
    "u^2", "u^2*u_x", "u^2*u_xx", "u^2*u_xxx",
    "u^3", "u^3*u_x", "u^3*u_xx", "u^3*u_xxx"
]
TRUE_SUPPORT = {"u_xx", "u", "u^3"}

# ============================================================
# Utilities
# ============================================================
def _norm_name(name: str) -> str:
    n = name.strip().upper()
    if n in {"Q-DEIM", "QDEIM", "DEIM"}: return "DEIM"
    if n in {"RAND-DEIM", "RANDDEIM"}:   return "RANDDEIM"
    if n in {"L-DEIM", "LDEIM"}:         return "LDEIM"
    if n in {"CUR", "CURSELECTOR"}:      return "CURSELECTOR"
    return n

def _build_sampler(sampler_name: str, X: np.ndarray, n_d: int,
                   t_o_np: np.ndarray, x_o_np: np.ndarray, kwargs: Dict[str, Any]):
    s = _norm_name(sampler_name)
    if s == "DEIM":
        return DEIM(X, n_d, t_o_np, x_o_np,
                    tolerance=kwargs.get("tolerance", 1e-7),
                    num_basis=kwargs.get("num_basis", 1))
    if s == "RANDDEIM":
        return RandDEIM(X, n_d, t_o_np, x_o_np,
                        tolerance=kwargs.get("tolerance", 1e-7),
                        num_basis=kwargs.get("num_basis", 8),
                        oversample=kwargs.get("oversample", 12),
                        n_power=kwargs.get("n_power", 1),
                        rng=kwargs.get("rng", None))
    if s == "LDEIM":
        return LDEIM(X, n_d, t_o_np, x_o_np,
                     tolerance=kwargs.get("tolerance", 1e-7),
                     num_basis=kwargs.get("num_basis", 8),
                     n_patches=kwargs.get("n_patches", 4),
                     overlap=kwargs.get("overlap", 2),
                     k_per_patch=kwargs.get("k_per_patch", 3))
    if s == "CURSELECTOR":
        return CURSelector(X, n_d, t_o_np, x_o_np,
                           tolerance=kwargs.get("tolerance", 1e-7),
                           num_basis=kwargs.get("num_basis", 8),
                           c_cols=kwargs.get("c_cols", None),
                           r_rows=kwargs.get("r_rows", None),
                           deterministic=kwargs.get("deterministic", True),
                           rng=kwargs.get("rng", None))
    raise ValueError(f"Unknown sampler '{sampler_name}'.")

def _support_from_coeffs(coeff_vec: np.ndarray, order: List[str], tol: float = 1e-8) -> set:
    idxs = [i for i, c in enumerate(coeff_vec) if abs(float(c)) > tol]
    return set(order[i] for i in idxs)

def _precision_recall_f1(pred: set, truth: set) -> Tuple[float, float, float]:
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def _coeff_l2_err(coeff_vec: np.ndarray, order: List[str], nu_ac: float = NU_AC) -> float:
    true = np.zeros(len(order))
    if "u" in order:      true[order.index("u")] = 1.0
    if "u^3" in order:    true[order.index("u^3")] = -1.0
    if "u_xx" in order:   true[order.index("u_xx")] = nu_ac
    coeff = np.array([float(c) for c in coeff_vec])
    return float(np.linalg.norm(coeff - true, 2))

def _expand_kwargs_cols(kw: Dict[str, Any]) -> Dict[str, Any]:
    # Flatten known keys to have stable CSV columns
    return {
        "kw_n_d": kw.get("n_d", ""),
        "kw_num_basis": kw.get("num_basis", ""),
        "kw_tolerance": kw.get("tolerance", ""),
        "kw_oversample": kw.get("oversample", ""),
        "kw_n_power": kw.get("n_power", ""),
        "kw_n_patches": kw.get("n_patches", ""),
        "kw_overlap": kw.get("overlap", ""),
        "kw_k_per_patch": kw.get("k_per_patch", ""),
        "kw_c_cols": kw.get("c_cols", ""),
        "kw_r_rows": kw.get("r_rows", ""),
        "kw_deterministic": kw.get("deterministic", ""),
    }

# ============================================================
# Data loader from AC.mat + sampling
# ============================================================
def create_data(sampler: str = "DEIM", sampler_kwargs: Dict[str, Any] = None):
    """
    Load AC.mat and return (coords, data) with coords[:,0]=t, coords[:,1]=x.
    Sampler ∈ {DEIM, RandDEIM, LDEIM, CURSelector}.
    """
    if sampler_kwargs is None:
        sampler_kwargs = {}

    mat = loadmat(os.path.join(root_dir, "src", "data", "AC.mat"))
    t_o = np.asarray(mat["tt"]).flatten()[0:201]  # (Nt,)
    x_o = np.asarray(mat["x"]).flatten()         # (Nx,)
    Exact = np.real(mat["uu"])                   # (Nx,Nt) or (Nt,Nx)

    # Ensure X has shape (Nx, Nt) = (len(x_o), len(t_o))
    if Exact.shape == (x_o.size, t_o.size):
        X = Exact.copy()              # already (Nx, Nt) -> OK
    elif Exact.shape == (t_o.size, x_o.size):
        X = Exact.T.copy()            # transpose to (Nx, Nt)
    else:
        raise ValueError(
            f"Unexpected uu shape {Exact.shape}; expected "
            f"({x_o.size}, {t_o.size}) or ({t_o.size}, {x_o.size})."
    )

    n_d = int(sampler_kwargs.get("n_d", 3))
    
    
    
    sampler_obj = _build_sampler(sampler, X, n_d, t_o, x_o, sampler_kwargs)
    S_s, T_s, U_s = sampler_obj.execute()  # (x_samples, t_samples, u_samples)

    T_s = np.ravel(T_s); S_s = np.ravel(S_s); U_s = np.ravel(U_s)
    coords = torch.from_numpy(np.column_stack([T_s, S_s])).float()  # [t, x]
    data   = torch.from_numpy(U_s.reshape(-1, 1)).float()           # [u]
    return coords, data

# ============================================================
# Optional: visualize sampling patterns
# ============================================================
def plot_samples_only_ac(methods=None, sampler_kwargs=None, color_by_u=True):
    if methods is None: methods = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CURSelector"]
    if sampler_kwargs is None: sampler_kwargs = {}

    # Load full field once (for color)
    mat = loadmat(os.path.join(root_dir, "src", "data", "AC.mat"))
    t_o = np.asarray(mat["tt"]).flatten()[0:201]
    x_o = np.asarray(mat["x"]).flatten()
    Exact = np.real(mat["uu"])
    
        
        
        # Ensure X has shape (Nx, Nt) = (len(x_o), len(t_o))
    if Exact.shape == (x_o.size, t_o.size):
        X = Exact.copy()              # already (Nx, Nt) -> OK
    elif Exact.shape == (t_o.size, x_o.size):
        X = Exact.T.copy()            # transpose to (Nx, Nt)
    else:
        raise ValueError(
            f"Unexpected uu shape {Exact.shape}; expected "
            f"({x_o.size}, {t_o.size}) or ({t_o.size}, {x_o.size})."
    )

        
    t_o_np, x_o_np = t_o, x_o

    tmin, tmax = t_o.min(), t_o.max()
    xmin, xmax = x_o.min(), x_o.max()

    fig, axes = plt.subplots(2,2, figsize=(10,7), sharex=True, sharey=True, layout="constrained")
    axes = axes.ravel()
    titles = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CURSelector"]
    last_scatter = None

    for ax, name, title in zip(axes, methods, titles):
        sname = _norm_name(name)
        n_d = int(sampler_kwargs.get("n_d", 3))
        sampler_obj = _build_sampler(sname, X, n_d, t_o_np, x_o_np, sampler_kwargs)
        S_s, T_s, U_s = sampler_obj.execute()
        T_s = np.ravel(T_s); S_s = np.ravel(S_s); U_s = np.ravel(U_s)

        if color_by_u:
            last_scatter = ax.scatter(T_s, S_s, c=U_s, s=16, marker="o", edgecolors="none")
        else:
            last_scatter = ax.scatter(T_s, S_s, s=16, marker="o", edgecolors="none")

        ax.set_title(title); ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$")
        ax.set_xlim(tmin, tmax); ax.set_ylim(xmin, xmax)

    if color_by_u and last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes, location="right", shrink=0.85)
    plt.show()

# ============================================================
# Training a single configuration
# ============================================================
def train_one_combo_ac(sampler_name: str, sampler_kwargs: Dict[str, Any], seed: int,
                       max_iter: int = 25000, num_used: int = 5000) -> Dict[str, Any]:
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ds = Dataset(
        partial(create_data, sampler=sampler_name, sampler_kwargs=sampler_kwargs),
        preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": num_used},
        device=device,
    )
    try:
        n_samples_raw = int(ds.get_coords().shape[0])
    except Exception:
        coords_tmp, _ = partial(create_data, sampler=sampler_name, sampler_kwargs=sampler_kwargs)()
        n_samples_raw = int(coords_tmp.shape[0])
    n_samples_used = min(num_used, n_samples_raw)

    train_loader, test_loader = get_train_test_loader(ds, train_test_split=1.00)

    net = NN(2, [64, 64, 64, 64], 1)
    lib = Library1D(poly_order=POLY_ORDER, diff_order=DIFF_ORDER)
    sched = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
    cons  = STRidgeCons()
    est   = STRidge()
    mdl = DeepMoD(net, lib, est, cons).to(device)
    opt = torch.optim.Adam(mdl.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

    run_id = f"{_norm_name(sampler_name)}__{json.dumps(sampler_kwargs, sort_keys=True).replace(' ','') }__seed{seed}"
    run_dir = os.path.join("./data/deepymod/allen_cahn/grid", _norm_name(sampler_name), run_id)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    train(
        mdl, train_loader, test_loader, opt, sched,
        log_dir=run_dir, exp_ID="exp", write_iterations=25,
        max_iterations=max_iter, delta=1e-4, patience=200,
    )
    train_time = time.time() - t0

    coeffs = mdl.estimator_coeffs()[0].detach().cpu().numpy().ravel()
    supp   = _support_from_coeffs(coeffs, ORDER)
    prec, rec, f1 = _precision_recall_f1(supp, TRUE_SUPPORT)
    l2err = _coeff_l2_err(coeffs, ORDER, nu_ac=NU_AC)

    try:
        from GNSINDy.src.deepymod.analysis import load_tensorboard
        hist = load_tensorboard(run_dir)
        final_loss = float(hist["loss_train"][-1])
    except Exception:
        final_loss = 0.0

    meta = {
        "sampler": _norm_name(sampler_name),
        "seed": seed,
        "sampler_kwargs": sampler_kwargs,
        "train_time_s": round(train_time, 3),
        "final_loss": final_loss,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "coeff_l2": round(l2err, 4),
        "n_samples_raw": n_samples_raw,
        "n_samples_used": n_samples_used,
        "coeff_vector": [float(c) for c in coeffs],
        "coeffs_dict": {term: float(coeffs[i]) if i < len(coeffs) else 0.0
                        for i, term in enumerate(ORDER)},
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta

# ============================================================
# Grid definition (aligned; smaller DEIM basis includes 1)
# ============================================================
samplers = ["DEIM", "RandDEIM", "LDEIM", "CURSelector"]
grid = {
    "DEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-7}
        for nd in [3, 5] for nb in [4, 8]
    ],
    "RandDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-7, "oversample": os, "n_power": 1}
        for nd in [3, 5] for nb in [4, 8] for os in [8, 12]
    ],
    "LDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-7,
         "n_patches": 4, "overlap": ov, "k_per_patch": 3}
        for nd in [3, 5] for nb in [4, 8] for ov in [0, 2]
    ],
    "CURSelector": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-7,
         "c_cols": None, "r_rows": None, "deterministic": True}
        for nd in [3, 5] for nb in [4, 8]
    ],
}
seeds = [0, 1]

# ============================================================
# Run grid and save CSVs
# ============================================================
def main():
    results_csv = "./data/deepymod/allen_cahn/grid_results.csv"
    coeffs_csv  = "./data/deepymod/allen_cahn/grid_coeffs.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    # headers for kwargs expansion:
    kw_headers = [
        "kw_n_d","kw_num_basis","kw_tolerance","kw_oversample","kw_n_power",
        "kw_n_patches","kw_overlap","kw_k_per_patch","kw_c_cols","kw_r_rows","kw_deterministic"
    ]

    with open(results_csv, "w", newline="") as f_sum, open(coeffs_csv, "w", newline="") as f_coef:
        sum_writer = csv.writer(f_sum)
        coef_writer = csv.writer(f_coef)

        # summary header
        sum_writer.writerow([
            "sampler","seed","sampler_kwargs",
            "n_samples_raw","n_samples_used",
            "train_time_s","final_loss",
            "precision","recall","f1","coeff_l2",
            *kw_headers,
            "coeff_vector_json"
        ])
        # coefficients header
        coef_writer.writerow(["sampler","seed","n_samples_raw","n_samples_used", *ORDER])

        for s in samplers:
            for kw in grid[s]:
                for sd in seeds:
                    print(f"\n>>> [AC] Running {s} | seed={sd} | {kw}")
                    meta = train_one_combo_ac(s, kw, seed=sd, max_iter=25000, num_used=5000)

                    # expand kwargs
                    kw_exp = _expand_kwargs_cols(meta["sampler_kwargs"])
                    kw_vals = [kw_exp[h] for h in kw_headers]

                    # write summary
                    sum_writer.writerow([
                        meta["sampler"], meta["seed"], json.dumps(meta["sampler_kwargs"], sort_keys=True),
                        meta["n_samples_raw"], meta["n_samples_used"],
                        meta["train_time_s"], meta["final_loss"],
                        meta["precision"], meta["recall"], meta["f1"], meta["coeff_l2"],
                        *kw_vals,
                        json.dumps(meta["coeff_vector"])
                    ])

                    # write named coefficients
                    coef_row = [meta["sampler"], meta["seed"], meta["n_samples_raw"], meta["n_samples_used"]]
                    coef_row += [meta["coeffs_dict"].get(term, 0.0) for term in ORDER]
                    coef_writer.writerow(coef_row)

    print(f"\nDone (Allen–Cahn).")
    print(f"Summary:     {results_csv}")
    print(f"Coefficients:{coeffs_csv}")

if __name__ == "__main__":
    main()

# --------------- Optional: to quickly preview sampling ---------------
# plot_samples_only_ac(
#     methods=["Q-DEIM","Rand-DEIM","L-DEIM","CURSelector"],
#     sampler_kwargs={"n_d": 3, "num_basis": 4, "n_patches": 4, "overlap": 2, "k_per_patch": 3},
#     color_by_u=True
# )
