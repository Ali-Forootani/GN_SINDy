#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 19:18:19 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allen–Cahn sampling demo (DEIM / RandDEIM / LDEIM / CURSelector)
- Mirrors the structure and plotting style used for Burgers.
- Shows ONLY the selected (t,x) samples as scatter, colored by u.
"""

import os
import sys
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from typing import Dict, Any, List


# -----------------------------------------------------------------------------
# Project paths (same helper you used elsewhere)
# -----------------------------------------------------------------------------
cwd = os.getcwd()
sys.path.append(cwd)

def setting_directory(depth: int):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for _ in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(2)

# -----------------------------------------------------------------------------
# DeePyMoD / GNSINDy imports
# -----------------------------------------------------------------------------
from GNSINDy.src.deepymod.data.DEIM_class import DEIM, RandDEIM, LDEIM, CURSelector
from GNSINDy.src.deepymod.utils import plot_config_file

# -----------------------------------------------------------------------------
# Repro & device
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device:", device)

np.random.seed(42)
torch.manual_seed(50)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Sampler name normalization and factory (same API as your Burgers version)
# -----------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    n = name.strip().upper()
    if n in {"Q-DEIM", "QDEIM", "DEIM"}: return "DEIM"
    if n in {"RAND-DEIM", "RANDDEIM"}:   return "RANDDEIM"
    if n in {"L-DEIM", "LDEIM"}:         return "LDEIM"
    if n in {"CUR", "CUR-SELECTOR", "CURSELECTOR"}: return "CURSELECTOR"
    return n

def _build_sampler(
    sampler_name: str,
    X: np.ndarray,
    n_d: int,
    t_o_np: np.ndarray,
    x_o_np: np.ndarray,
    kwargs: Dict[str, Any],
):
    s = _norm_name(sampler_name)
    if s == "DEIM":
        return DEIM(
            X, n_d, t_o_np, x_o_np,
            tolerance=kwargs.get("tolerance", 1e-5),
            num_basis=kwargs.get("num_basis", 5),
        )
    if s == "RANDDEIM":
        return RandDEIM(
            X, n_d, t_o_np, x_o_np,
            tolerance=kwargs.get("tolerance", 1e-5),
            num_basis=kwargs.get("num_basis", 8),
            oversample=kwargs.get("oversample", 12),
            n_power=kwargs.get("n_power", 1),
            rng=kwargs.get("rng", None),
        )
    if s == "LDEIM":
        return LDEIM(
            X, n_d, t_o_np, x_o_np,
            tolerance=kwargs.get("tolerance", 1e-5),
            num_basis=kwargs.get("num_basis", 8),
            n_patches=kwargs.get("n_patches", 4),
            overlap=kwargs.get("overlap", 2),
            k_per_patch=kwargs.get("k_per_patch", 3),
        )
    if s == "CURSELECTOR":
        return CURSelector(
            X, n_d, t_o_np, x_o_np,
            tolerance=kwargs.get("tolerance", 1e-5),
            num_basis=kwargs.get("num_basis", 8),
            c_cols=kwargs.get("c_cols", None),
            r_rows=kwargs.get("r_rows", None),
            deterministic=kwargs.get("deterministic", True),
            rng=kwargs.get("rng", None),
        )
    raise ValueError(f"Unknown sampler '{sampler_name}'. Use DEIM | RandDEIM | LDEIM | CURSelector.")

# -----------------------------------------------------------------------------
# Load full Allen–Cahn field from AC.mat
# Expects keys: 'tt' (T x 1 or 1 x T), 'x' (X x 1 or 1 x X), 'uu' (X x T)
# -----------------------------------------------------------------------------
def load_ac_field(mat_path: str):
    """
    Returns:
        X         : np.ndarray of shape (nx, nt) with the field values
        t_o_np    : np.ndarray of shape (nt,)
        x_o_np    : np.ndarray of shape (nx,)
    """
    data = sio.loadmat(mat_path)
    # take the first 201 time steps as in your previous script
    t_o = np.asarray(data["tt"]).flatten()[:201]
    x_o = np.asarray(data["x"]).flatten()
    # uu expected shape (nx, nt); slice time dimension to match t_o
    Exact = np.real(np.asarray(data["uu"]))[:, :len(t_o)]

    # Sanity checks
    assert Exact.shape[0] == len(x_o), f"uu rows ({Exact.shape[0]}) != len(x) ({len(x_o)})"
    assert Exact.shape[1] == len(t_o), f"uu cols ({Exact.shape[1]}) != len(t) ({len(t_o)})"

    return Exact, t_o.astype(float), x_o.astype(float)

# -----------------------------------------------------------------------------
# Data creation for DeepMoD (coords [t,x], data [u]) using a sampler
# -----------------------------------------------------------------------------
def create_data_ac(
    mat_path: str,
    sampler: str = "DEIM",
    sampler_kwargs: Dict[str, Any] = None,
):
    if sampler_kwargs is None:
        sampler_kwargs = {}

    X, t_o_np, x_o_np = load_ac_field(mat_path)  # X: (nx, nt)
    n_d = int(sampler_kwargs.get("n_d", 3))

    sampler_obj = _build_sampler(_norm_name(sampler), X, n_d, t_o_np, x_o_np, sampler_kwargs)
    S_s, T_s, U_s = sampler_obj.execute()  # S: x-samples, T: t-samples, U: sampled values

    # Flatten
    T_s = np.ravel(T_s)
    S_s = np.ravel(S_s)
    U_s = np.ravel(U_s)

    # DeepMoD format: coords (N,2) with columns [t, x], data (N,1)
    coords = torch.from_numpy(np.column_stack([T_s, S_s])).float()
    data = torch.from_numpy(U_s.reshape(-1, 1)).float()
    return coords, data

# -----------------------------------------------------------------------------
# Plot ONLY the samples, like your Burgers plot (aligned axes and ticks)
# -----------------------------------------------------------------------------
def _nice_ticks(vmin: float, vmax: float, step: float):
    """Create ticks from nearest multiples of 'step' within [vmin, vmax]."""
    lo = np.ceil(vmin / step) * step
    hi = np.floor(vmax / step) * step
    if hi < lo:
        lo, hi = vmin, vmax
    return np.arange(lo, hi + 1e-9, step)

def plot_samples_only_ac(
    mat_path: str,
    methods: List[str] = None,
    sampler_kwargs: Dict[str, Any] = None,
    color_by_u: bool = True,
    savepath: str = None,
    # Optional tick spacing (change if you want exact integers like Burgers)
    t_step: float = None,
    x_step: float = None,
):
    if methods is None:
        methods = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR"]
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # Load the full field once (for axis limits and samplers)
    X_full, t_o_np, x_o_np = load_ac_field(mat_path)
    tmin, tmax = float(np.min(t_o_np)), float(np.max(t_o_np))
    xmin, xmax = float(np.min(x_o_np)), float(np.max(x_o_np))
    
    
    # Build ticks (either automatic nice ticks or user-specified spacing)
    if t_step is None:
        # heuristic: split into ~6 ticks
        t_step = max((tmax - tmin) / 2.0, 1e-6)
    if x_step is None:
        x_step = max((xmax - xmin) / 4.0, 1e-6)
        
    
        
    xticks = _nice_ticks(tmin-0, tmax+0, t_step)
    yticks = _nice_ticks(xmin-0, xmax+0, x_step)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    axes = axes.ravel()
    titles = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR-Selector"]

    last_scatter = None
    for ax, name, title in zip(axes, methods, titles):
        coords, data = create_data_ac(mat_path, sampler=name, sampler_kwargs=sampler_kwargs)
        # coords: (N,2) with [t,x]; data: (N,1)
        T_s = coords[:, 0].cpu().numpy().ravel()
        S_s = coords[:, 1].cpu().numpy().ravel()
        U_s = data[:, 0].cpu().numpy().ravel()

        if color_by_u:
            last_scatter = ax.scatter(T_s, S_s, c=U_s, s=18, marker="o", edgecolors="none")
        else:
            last_scatter = ax.scatter(T_s, S_s, s=18, marker="o", edgecolors="none")

        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    if color_by_u and last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes, location="right", shrink=0.85)

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=600, bbox_inches="tight")
        print(f"Saved figure to: {savepath}")

    plt.show()

# -----------------------------------------------------------------------------
# Example usage (mirrors your Burgers call)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Path to AC.mat (adjust if yours is elsewhere)
    mat_path = os.path.join(root_dir, "src", "data", "AC.mat")

    save_dir = os.path.join(root_dir, "src", "examples", "data", "deepymod", "allen_cahn", "grid")
    savepath = os.path.join(save_dir, "allen_cahn_sampling_comparison.png")

    # Use the SAME default sampler grid you used for Burgers:
    plot_samples_only_ac(
        mat_path=mat_path,
        methods=["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR"],
        sampler_kwargs={
            "n_d": 5,           # DOFs per time slice (tunable)
            "num_basis": 4,     # POD basis count
            "tolerance": 1e-7,
            # RandDEIM extras:
            "oversample": 12, "n_power": 1,
            # LDEIM extras:
            "n_patches": 4, "overlap": 2, "k_per_patch": 3,
            # CUR extras:
            "deterministic": True
        },
        color_by_u=True,
        savepath=savepath,
        # If you want exact ticks like Burgers, set explicit steps here:
        # e.g., t_step=2.0, x_step=5.0
        # Otherwise we compute "nice" steps from the data range.
        # t_step=2.0,
        # x_step=5.0,
    )
