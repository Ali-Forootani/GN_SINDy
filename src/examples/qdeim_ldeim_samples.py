#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:42:58 2025

@author: forootan
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













# --- helpers to build matched-sample creators (no class edits needed) ---

def create_data_qdeim(n_d=5, k=10):
    """Q-DEIM (your DEIM class uses pivoted-QR). Returns (coords, data)."""
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0
    _, Exact = burgers_delta_org(x_o, t_o, v, A)  # Exact: shape (nx, nt) numpy or torch

    # Force exactly k samples per axis per chunk: total n_d * k^2 points
    deim = DEIM(
        X=Exact, n_d=n_d, t_o=t_o.numpy(), x_o=x_o.numpy(),
        tolerance=1.0,      # makes k == num_basis in your DEIM implementation
        num_basis=k
    )
    S_s, T_s, U_s = deim.execute()
    coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1))  # (t, x)
    data   = torch.from_numpy(U_s.reshape(-1, 1))
    return coords, data

def create_data_ldeim(n_d=5, k=10):
    """Localized DEIM configured to yield the same count n_d * k^2."""
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0
    _, Exact = burgers_delta_org(x_o, t_o, v, A)

    ldeim = LDEIM(
        X=Exact, n_d=n_d, t_o=t_o.numpy(), x_o=x_o.numpy(),
        tolerance=-1.0,     # chooser returns large k; later capped to num_basis
        num_basis=k,
        n_patches=1,        # global patch -> exactly k indices on each axis
        overlap=0,
        k_per_patch=None
    )
    
    
    ldeim = LDEIM(
    X=Exact,
    n_d=n_d,
    t_o=t_o.numpy(),
    x_o=x_o.numpy(),
    tolerance=-1.0,
    num_basis=k,
    n_patches=4,     # <-- key change
    overlap=2,       # small overlap prevents sharp discontinuities
    k_per_patch=3    # per-patch DEIM rank (optional, controls density)
)

    
    
    S_s, T_s, U_s = ldeim.execute()
    coords = torch.from_numpy(np.stack((T_s, S_s), axis=-1))
    data   = torch.from_numpy(U_s.reshape(-1, 1))
    return coords, data





# Choose matching parameters (total samples = n_d * k**2)
n_d = 5
k   = 5

# Q-DEIM (pivoted-QR DEIM)
dataset_q = Dataset(
    lambda: create_data_qdeim(n_d=n_d, k=k),
    preprocess_kwargs=dict(noise_level=0.0, normalize_coords=False, normalize_data=False),
    subsampler=None,   # IMPORTANT: do NOT random-sub-sample; we want the selected points
    device=device
)

# L-DEIM
n_d = 5   # time chunking unchanged
k   = 5  # basis / per-chunk size unchanged

dataset_l = Dataset(
    lambda: create_data_ldeim(n_d=n_d, k=k),  # use updated ldeim params
    preprocess_kwargs=dict(noise_level=0.0, normalize_coords=False, normalize_data=False),
    subsampler=None,
    device=device
)

# Pull coords/data to CPU for plotting
coords_q = dataset_q.get_coords().cpu()
data_q   = dataset_q.get_data().cpu()
coords_l = dataset_l.get_coords().cpu()
data_l   = dataset_l.get_data().cpu()

print("Q-DEIM samples:", data_q.shape[0])
print("L-DEIM samples:", data_l.shape[0])  # both should be n_d * k**2



vmin = float(torch.min(torch.cat([data_q, data_l])))
vmax = float(torch.max(torch.cat([data_q, data_l])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc1 = ax1.scatter(coords_q[:, 0], coords_q[:, 1], c=data_q[:, 0], marker="x", s=25, vmin=vmin, vmax=vmax)
ax1.set_title("Q-DEIM samples")
ax1.set_xlabel("t"); ax1.set_ylabel("x")

sc2 = ax2.scatter(coords_l[:, 0], coords_l[:, 1], c=data_l[:, 0], marker="o", s=25, vmin=vmin, vmax=vmax)
ax2.set_title("L-DEIM samples")
ax2.set_xlabel("t")

cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.9)
cbar.set_label("u(x,t)")

plt.tight_layout()
plt.show()


