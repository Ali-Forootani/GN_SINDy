#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 08:03:41 2025

@author: forootan
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 08:48:12 2023

@author: forootani


Discovering Burgers' equation with GNSINDy

aliforootani@ieee.org
forootani@mpi-magdeburg.mpg.de 

"""


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


import matplotlib.pyplot as plt

# General imports
import numpy as np
import torch

# DeePyMoD imports
from GNSINDy.src.deepymod import DeepMoD
from GNSINDy.src.deepymod.data import Dataset, get_train_test_loader
from GNSINDy.src.deepymod.data.samples import Subsample_random
from GNSINDy.src.deepymod.data.burgers import burgers_delta, burgers_delta_org
from GNSINDy.src.deepymod.data.burgers import burgers_delta
from GNSINDy.src.deepymod.model.constraint import LeastSquares, Ridge, STRidgeCons
from GNSINDy.src.deepymod.model.func_approx import NN
from GNSINDy.src.deepymod.model.library import Library1D
from GNSINDy.src.deepymod.model.sparse_estimators import Threshold, STRidge
from GNSINDy.src.deepymod.training import train
#from deepymod.training.training_2 import train
from GNSINDy.src.deepymod.training.sparsity_scheduler import Periodic, TrainTest, TrainTestPeriodic
#from deepymod.data.data_set_preparation import DatasetPDE, pde_data_loader
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from GNSINDy.src.deepymod.data.DEIM_class import DEIM, RandDEIM, LDEIM, CURSelector
import shutil

from GNSINDy.src.deepymod.utils.utilities import create_or_reset_directory

from GNSINDy.src.deepymod.utils import plot_config_file



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(50)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#########################
#########################
#########################

# Making dataset



###########################################



# --- add at the top with the other imports ---
from functools import partial
from typing import Dict, Any






###########################################################
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# ---------- Sampler name normalization ----------
def _norm_name(name: str) -> str:
    n = name.strip().upper()
    if n in {"Q-DEIM","QDEIM","DEIM"}: return "DEIM"
    if n in {"RAND-DEIM","RANDDEIM"}:  return "RANDDEIM"
    if n in {"L-DEIM","LDEIM"}:        return "LDEIM"
    if n in {"CUR","CUR-SELECTOR"}:     return "CUR-SELECTOR"
    return n

# ---------- Sampler factory ----------
def _build_sampler(sampler_name: str, X: np.ndarray, n_d: int,
                   t_o_np: np.ndarray, x_o_np: np.ndarray, kwargs: Dict[str, Any]):
    s = _norm_name(sampler_name)
    if s == "DEIM":
        return DEIM(X, n_d, t_o_np, x_o_np,
                    tolerance=kwargs.get("tolerance", 1e-3),
                    num_basis=kwargs.get("num_basis", 1))
    if s == "RANDDEIM":
        return RandDEIM(X, n_d, t_o_np, x_o_np,
                        tolerance=kwargs.get("tolerance", 1e-5),
                        num_basis=kwargs.get("num_basis", 20),
                        oversample=kwargs.get("oversample", 10),
                        n_power=kwargs.get("n_power", 1),
                        rng=kwargs.get("rng", None))
    if s == "LDEIM":
        return LDEIM(X, n_d, t_o_np, x_o_np,
                     tolerance=kwargs.get("tolerance", 1e-5),
                     num_basis=kwargs.get("num_basis", 5),
                     n_patches=kwargs.get("n_patches", 4),
                     overlap=kwargs.get("overlap", 0),
                     k_per_patch=kwargs.get("k_per_patch", None))
    if s == "CUR-SELECTOR":
        return CURSelector(X, n_d, t_o_np, x_o_np,
                           tolerance=kwargs.get("tolerance", 1e-5),
                           num_basis=kwargs.get("num_basis", 5),
                           c_cols=kwargs.get("c_cols", None),
                           r_rows=kwargs.get("r_rows", None),
                           deterministic=kwargs.get("deterministic", True),
                           rng=kwargs.get("rng", None))
    raise ValueError(f"Unknown sampler '{sampler_name}'. Use DEIM | RandDEIM | LDEIM | CUR-Selector.")







# ---------- Data creation using a sampler ----------




def create_data(sampler: str = "LDEIM", sampler_kwargs: Dict[str, Any] = None):
    """Return ONLY (coords, data) for DeepMoD.Dataset."""
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # reference grid (used internally by samplers)
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0

    # full field (for sampling); numpy for samplers
    _, Exact_torch = burgers_delta_org(x_o, t_o, v, A)
    X = Exact_torch.detach().cpu().numpy()
    t_o_np = t_o.detach().cpu().numpy()
    x_o_np = x_o.detach().cpu().numpy()

    n_d = int(sampler_kwargs.get("n_d", 5))
    sampler_obj = _build_sampler(sampler, X, n_d, t_o_np, x_o_np, sampler_kwargs)
    S_s, T_s, U_s = sampler_obj.execute()  # NOTE: returns S (x), T (t), U values

    # ensure 1D arrays
    T_s = np.ravel(T_s)
    S_s = np.ravel(S_s)
    U_s = np.ravel(U_s)

    # DeepMoD expects: coords (N,2) with columns [t, x], data (N,1)
    coords = torch.from_numpy(np.column_stack([T_s, S_s])).float()   # (N,2)
    data   = torch.from_numpy(U_s.reshape(-1, 1)).float()            # (N,1)
    return coords, data



# ---------- ONLY SAMPLES (aligned axes) ----------



def plot_samples_only(methods=None, sampler_kwargs=None, color_by_u=True):
    import numpy as np
    import matplotlib.pyplot as plt

    if methods is None:
        methods = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR-Selector"]
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # fixed axis limits
    tmin, tmax = 0.5, 10.0
    xmin, xmax = -8.0, 8.0

    # exact ticks you requested
    xticks = np.arange(0, 10 + 1e-9, 2)     # 0,2,4,6,8,10
    yticks = np.arange(-5, 6 + 1e-9, 5)   # ..., -5,0,5, ...

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    axes = axes.ravel()
    titles = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR-Selector"]

    # prepare the same field once
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0
    _, Exact_torch = burgers_delta_org(x_o, t_o, v, A)
    X = Exact_torch.detach().cpu().numpy()
    t_o_np = t_o.detach().cpu().numpy()
    x_o_np = x_o.detach().cpu().numpy()

    last_scatter = None
    for ax, name, title in zip(axes, methods, titles):
        sname = _norm_name(name)
        n_d = int(sampler_kwargs.get("n_d", 5))
        sampler_obj = _build_sampler(sname, X, n_d, t_o_np, x_o_np, sampler_kwargs)
        S_s, T_s, U_s = sampler_obj.execute()

        T_s = np.ravel(T_s)
        S_s = np.ravel(S_s)
        U_s = np.ravel(U_s)

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
    plt.show()



# Simply modify the end of plot_samples_only
def plot_samples_only(methods=None, sampler_kwargs=None, color_by_u=True, savepath=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if methods is None:
        methods = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR-Selector"]
    if sampler_kwargs is None:
        sampler_kwargs = {}

    tmin, tmax = 0.5, 10.0
    xmin, xmax = -8.0, 8.0

    xticks = np.arange(0, 10 + 1e-9, 2)
    yticks = np.arange(-5, 6 + 1e-9, 5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    axes = axes.ravel()
    titles = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CUR-Selector"]

    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0
    _, Exact_torch = burgers_delta_org(x_o, t_o, v, A)
    X = Exact_torch.detach().cpu().numpy()
    t_o_np = t_o.detach().cpu().numpy()
    x_o_np = x_o.detach().cpu().numpy()

    last_scatter = None
    for ax, name, title in zip(axes, methods, titles):
        sname = _norm_name(name)
        sampler_obj = _build_sampler(sname, X, sampler_kwargs.get("n_d",5), t_o_np, x_o_np, sampler_kwargs)
        S_s, T_s, U_s = sampler_obj.execute()

        T_s = np.ravel(T_s)
        S_s = np.ravel(S_s)
        U_s = np.ravel(U_s)

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

    # ---- SAVE THE FIGURE ----
    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")
        print(f"Saved figure to: {savepath}")

    plt.show()


import os

save_dir = "/home/forootan/Documents/MPI/GNSINDy/src/examples/data/deepymod/burgers/grid"
os.makedirs(save_dir, exist_ok=True)   # ensures dir exists

savepath = os.path.join(save_dir, "burgers_sampling_comparison.png")



plot_samples_only(
    methods=["Q-DEIM","Rand-DEIM","L-DEIM","CUR-Selector"],
    sampler_kwargs={"n_d": 5, "num_basis": 8, "n_patches": 4, "overlap": 2, "k_per_patch": 3},
    color_by_u=True,
    savepath = savepath 
)


