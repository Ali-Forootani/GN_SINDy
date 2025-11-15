#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 09:28:46 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid experiment: KdV equation discovery with GNSINDy/DeepMoD + DEIM-family samplers.

- Loads kdv.mat (fields: t, x, usol)
- Runs a sampler grid over {DEIM, RandDEIM, LDEIM, CURSelector}
- Trains DeepMoD and logs metrics (precision/recall/F1; coeff L2 vs true KdV)
- Saves:
    ./data/deepymod/kdv/grid_results.csv
    ./data/deepymod/kdv/grid_coeffs.csv
"""

import os, json, csv, time, random
from functools import partial
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------- Device & determinism ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
np.random.seed(42)
torch.manual_seed(50)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Root dir helper ----------------
def setting_directory(depth: int):
    import sys, os
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

try:
    root_dir
except NameError:
    root_dir = setting_directory(2)

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

# Samplers
from GNSINDy.src.deepymod.data.DEIM_class import DEIM, RandDEIM, LDEIM, CURSelector

# ============================================================
# Config: Library & Truth (KdV: u_t = - u_xxx + 6 u u_x)
# ============================================================
POLY_ORDER = 2
DIFF_ORDER = 3

# Fixed ordering (same as your AC script, superset of needed terms)
ORDER = [
    "1", "u_x", "u_xx", "u_xxx",
    "u", "u*u_x", "u*u_xx", "u*u_xxx",
    "u^2", "u^2*u_x", "u^2*u_xx", "u^2*u_xxx",
]
TRUE_SUPPORT = {"u_xxx", "u*u_x"}
TRUE_COEFFS = {"u_xxx": -1.0, "u*u_x": 6.0}  # u_t = -u_xxx + 6 u u_x

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
                    num_basis=kwargs.get("num_basis", 4))
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

def _coeff_l2_err(coeff_vec: np.ndarray, order: List[str], true_coeffs: Dict[str, float]) -> float:
    true = np.zeros(len(order), dtype=float)
    for k, v in true_coeffs.items():
        if k in order: true[order.index(k)] = float(v)
    coeff = np.array([float(c) for c in coeff_vec], dtype=float)
    # pad if estimator produced fewer terms than ORDER
    if coeff.size < true.size:
        coeff = np.pad(coeff, (0, true.size - coeff.size))
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
# Data loader from kdv.mat + sampling
# ============================================================
def create_data_kdv(sampler: str = "DEIM", sampler_kwargs: Dict[str, Any] = None):
    """
    Load kdv.mat and return (coords, data) with coords[:,0]=t, coords[:,1]=x.
    Sampler ∈ {DEIM, RandDEIM, LDEIM, CURSelector}.
    """
    if sampler_kwargs is None:
        sampler_kwargs = {}

    mat = loadmat(os.path.join(root_dir, "src", "data", "kdv.mat"))
    t_o = np.asarray(mat["t"]).flatten()[0:201]   # (Nt,)
    x_o = np.asarray(mat["x"]).flatten()          # (Nx,)
    Exact = np.real(mat["usol"])                  # (Nx,Nt) or (Nt,Nx)

    # Ensure X has shape (Nx, Nt)
    if Exact.shape == (x_o.size, t_o.size):
        X = Exact.copy()
    elif Exact.shape == (t_o.size, x_o.size):
        X = Exact.T.copy()
    else:
        raise ValueError(
            f"Unexpected usol shape {Exact.shape}; expected "
            f"({x_o.size}, {t_o.size}) or ({t_o.size}, {x_o.size})."
        )

    # n_d for KdV: default 2 (t,x). You can extend to [3] if you wish.
    n_d = int(sampler_kwargs.get("n_d", 2))
    sampler_obj = _build_sampler(_norm_name(sampler), X, n_d, t_o, x_o, sampler_kwargs)
    S_s, T_s, U_s = sampler_obj.execute()  # (x_samples, t_samples, u_samples)

    T_s = np.ravel(T_s); S_s = np.ravel(S_s); U_s = np.ravel(U_s)
    coords = torch.from_numpy(np.column_stack([T_s, S_s])).float()  # [t, x]
    data   = torch.from_numpy(U_s.reshape(-1, 1)).float()           # [u]
    return coords, data

# ============================================================
# Train one configuration
# ============================================================
def train_one_combo_kdv(sampler_name: str, sampler_kwargs: Dict[str, Any], seed: int,
                        max_iter: int = 25000, num_used: int = 5000) -> Dict[str, Any]:
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ds = Dataset(
        partial(create_data_kdv, sampler=sampler_name, sampler_kwargs=sampler_kwargs),
        preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": num_used},
        device=device,
    )

    try:
        n_samples_raw = int(ds.get_coords().shape[0])
    except Exception:
        coords_tmp, _ = partial(create_data_kdv, sampler=sampler_name, sampler_kwargs=sampler_kwargs)()
        n_samples_raw = int(coords_tmp.shape[0])
    n_samples_used = min(num_used, n_samples_raw)

    train_loader, test_loader = get_train_test_loader(ds, train_test_split=1.00)

    net = NN(2, [32, 32, 32, 32], 1)
    lib = Library1D(poly_order=POLY_ORDER, diff_order=DIFF_ORDER)
    sched = TrainTestPeriodic(periodicity=50, patience=1000, delta=1e-5)
    cons  = STRidgeCons()
    est   = STRidge()
    mdl = DeepMoD(net, lib, est, cons).to(device)
    opt = torch.optim.Adam(mdl.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

    run_id = f"{_norm_name(sampler_name)}__{json.dumps(sampler_kwargs, sort_keys=True).replace(' ','') }__seed{seed}"
    run_dir = os.path.join("./data/deepymod/kdv/grid", _norm_name(sampler_name), run_id)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    train(
        mdl, train_loader, test_loader, opt, sched,
        log_dir=run_dir, exp_ID="exp", write_iterations=250,
        max_iterations=max_iter, delta=1e-4, patience=200,
    )
    train_time = time.time() - t0

    coeffs = mdl.estimator_coeffs()[0].detach().cpu().numpy().ravel()
    supp   = _support_from_coeffs(coeffs, ORDER)
    prec, rec, f1 = _precision_recall_f1(supp, TRUE_SUPPORT)
    l2err = _coeff_l2_err(coeffs, ORDER, TRUE_COEFFS)

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
# Grid (mirrors AC structure; nd defaults 2 for KdV)
# ============================================================
samplers = ["DEIM", "RandDEIM", "LDEIM", "CURSelector"]

# If you prefer exact parity with AC, you can set nd_list = [3,5].
nd_list = [2, 3]
grid = {
    "DEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5}
        for nd in nd_list for nb in [1, 2]
    ],
    "RandDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5, "oversample": os, "n_power": 1}
        for nd in nd_list for nb in [8, 10] for os in [8, 12]
    ],
    "LDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5,
         "n_patches": 4, "overlap": ov, "k_per_patch": 3}
        for nd in nd_list for nb in [8, 10] for ov in [0, 2]
    ],
    "CURSelector": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5,
         "c_cols": None, "r_rows": None, "deterministic": True}
        for nd in nd_list for nb in [8, 10]
    ],
}
seeds = [0, 1]

# ============================================================
# Run grid and save CSVs
# ============================================================
def main():
    results_csv = "./data/deepymod/kdv/grid_results.csv"
    coeffs_csv  = "./data/deepymod/kdv/grid_coeffs.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

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
                    print(f"\n>>> [KdV] Running {s} | seed={sd} | {kw}")
                    meta = train_one_combo_kdv(s, kw, seed=sd, max_iter=2, num_used=5000)

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

    print(f"\nDone (KdV).")
    print(f"Summary:     {results_csv}")
    print(f"Coefficients:{coeffs_csv}")

if __name__ == "__main__":
    main()
