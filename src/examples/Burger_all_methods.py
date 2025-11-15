#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:00:58 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:34:34 2025

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:50:11 2025

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
    if n in {"CUR","CURSELECTOR"}:     return "CURSELECTOR"
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
    if s == "CURSELECTOR":
        return CURSelector(X, n_d, t_o_np, x_o_np,
                           tolerance=kwargs.get("tolerance", 1e-5),
                           num_basis=kwargs.get("num_basis", 5),
                           c_cols=kwargs.get("c_cols", None),
                           r_rows=kwargs.get("r_rows", None),
                           deterministic=kwargs.get("deterministic", True),
                           rng=kwargs.get("rng", None))
    raise ValueError(f"Unknown sampler '{sampler_name}'. Use DEIM | RandDEIM | LDEIM | CURSelector.")







# ---------- Data creation using a sampler ----------

"""
def create_data(sampler: str = "LDEIM", sampler_kwargs: Dict[str, Any] = None):
    if sampler_kwargs is None: sampler_kwargs = {}

    # Reference grid for axis limits
    x_o = torch.linspace(-8, 8, 100)
    t_o = torch.linspace(0.5, 10.0, 100)
    v, A = 0.1, 1.0
    _, Exact_torch = burgers_delta_org(x_o, t_o, v, A)  # full field (only to feed samplers)
    X = Exact_torch.detach().cpu().numpy()
    t_o_np = t_o.detach().cpu().numpy()
    x_o_np = x_o.detach().cpu().numpy()

    n_d = int(sampler_kwargs.get("n_d", 5))
    sampler_obj = _build_sampler(sampler, X, n_d, t_o_np, x_o_np, sampler_kwargs)
    S_s, T_s, U_s = sampler_obj.execute()

    # ensure 1D arrays of equal length
    T_s = np.ravel(T_s)
    S_s = np.ravel(S_s)
    U_s = np.ravel(U_s)

    return (t_o_np.min(), t_o_np.max(), x_o_np.min(), x_o_np.max()), T_s, S_s, U_s
"""



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
        methods = ["Q-DEIM", "Rand-DEIM", "L-DEIM", "CURSelector"]
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
    titles = ["Q-DEIM (reference field)", "Rand-DEIM", "L-DEIM", "CURSelector"]

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





plot_samples_only(
    methods=["Q-DEIM","Rand-DEIM","L-DEIM","CURSelector"],
    sampler_kwargs={"n_d": 5, "num_basis": 10, "n_patches": 4, "overlap": 2, "k_per_patch": 3},
    color_by_u=True
)





########################################################







poly_order = 2
diff_order = 2

n_combinations = (poly_order+1)*(diff_order+1) 
n_features = 1









from functools import partial

SAMPLER_NAME  = "LDEIM"  # "DEIM" | "RandDEIM" | "LDEIM" | "CURSelector"
SAMPLER_KWARGS = {
    "n_d": 5, "tolerance": 1e-5, "num_basis": 5,
    # LDEIM extras:
    "n_patches": 4, "overlap": 2, "k_per_patch": 3,
    # RandDEIM extras:
    "oversample": 10, "n_power": 1, "rng": None,
    # CUR extras:
    "c_cols": None, "r_rows": None, "deterministic": True, "rng": None
}

num_of_samples = 1000
dataset = Dataset(
    partial(create_data, sampler=SAMPLER_NAME, sampler_kwargs=SAMPLER_KWARGS),
    preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
    subsampler=Subsample_random,
    subsampler_kwargs={"number_of_samples": num_of_samples},
    device=device,
)
train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=1.00)

network = NN(2, [64, 64, 64, 64], 1)
library = Library1D(poly_order=2, diff_order=2)
sparsity_scheduler = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
constraint = STRidgeCons()
estimator = STRidge()
model = DeepMoD(network, library, estimator, constraint).to(device)

optimizer = torch.optim.Adam(model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

out_dir = os.path.join("./data/deepymod/burgers/", f"{_norm_name(SAMPLER_NAME)}")
create_or_reset_directory(out_dir)



train(
    model, train_dataloader, test_dataloader, optimizer, sparsity_scheduler,
    log_dir=out_dir, exp_ID=f"{_norm_name(SAMPLER_NAME)}_exp",
    write_iterations=25, max_iterations=1, delta=1e-4, patience=200,
)





##################################################




import json, time, itertools, csv, random

# Term order used by Library1D (poly=2, diff=2) in your setup
ORDER = ["1","u_x","u_xx","u","u*u_x","u*u_xx","u^2","u^2*u_x","u^2*u_xx"]
TRUE_SUPPORT = {"u*u_x", "u_xx"}  # Burgers
# sign of coefficients: u_t = -u*u_x + v * u_xx (v=0.1) — we evaluate support + L2 anyway

def _support_from_coeffs(coeff_vec, order=ORDER, tol=1e-8):
    # coeff_vec expected shape (n_terms,)
    idxs = [i for i, c in enumerate(coeff_vec) if abs(float(c)) > tol]
    return set(order[i] for i in idxs)

def _precision_recall_f1(pred: set, truth: set):
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1

def _coeff_l2_err(coeff_vec, order=ORDER, v=0.1):
    # Build true vector
    true = np.zeros(len(order))
    true[order.index("u*u_x")] = -1.0
    true[order.index("u_xx")]  = v
    coeff = np.array([float(c) for c in coeff_vec])
    return float(np.linalg.norm(coeff - true, 2))

def train_one_combo(sampler_name, sampler_kwargs, seed, max_iter=15000):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    ds = Dataset(
        partial(create_data, sampler=sampler_name, sampler_kwargs=sampler_kwargs),
        preprocess_kwargs={"noise_level": 0.00, "normalize_coords": False, "normalize_data": False},
        subsampler=Subsample_random,
        subsampler_kwargs={"number_of_samples": 5000},
        device=device,
    )
    train_loader, test_loader = get_train_test_loader(ds, train_test_split=1.00)

    net = NN(2, [64, 64, 64, 64], 1)
    lib = Library1D(poly_order=2, diff_order=2)
    sched = TrainTestPeriodic(periodicity=100, patience=500, delta=1e-5)
    cons  = STRidgeCons()
    est   = STRidge()
    mdl = DeepMoD(net, lib, est, cons).to(device)
    opt = torch.optim.Adam(mdl.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=1e-3)

    run_id = f"{_norm_name(sampler_name)}__{json.dumps(sampler_kwargs, sort_keys=True).replace(' ','') }__seed{seed}"
    run_dir = os.path.join("./data/deepymod/burgers/grid", _norm_name(sampler_name), run_id)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    train(
        mdl, train_loader, test_loader, opt, sched,
        log_dir=run_dir, exp_ID="exp", write_iterations=25,
        max_iterations=max_iter, delta=1e-4, patience=200,
    )
    train_time = time.time() - t0

    # Collect metrics
    coeffs = mdl.estimator_coeffs()[0].detach().cpu().numpy().ravel()  # (n_terms,)
    supp   = _support_from_coeffs(coeffs)
    prec, rec, f1 = _precision_recall_f1(supp, TRUE_SUPPORT)
    l2err = _coeff_l2_err(coeffs)

    # Try to read final loss from TensorBoard (optional); otherwise 0.0
    try:
        from GNSINDy.src.deepymod.analysis import load_tensorboard
        hist = load_tensorboard(run_dir)
        final_loss = float(hist["loss_train"][-1])
    except Exception:
        final_loss = 0.0

    # Save metadata
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
        "coeff_vector": [float(c) for c in coeffs],
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta

# ---------- Define the grid ----------
samplers = ["DEIM", "RandDEIM", "LDEIM", "CURSelector"]
grid = {
    "DEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5}
        for nd in [3, 5] for nb in [5, 8]
    ],
    "RandDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5, "oversample": os, "n_power": 1}
        for nd in [3, 5] for nb in [5, 8] for os in [8, 12]
    ],
    "LDEIM": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5, "n_patches": np_, "overlap": ov, "k_per_patch": kp}
        for nd in [3, 5] for nb in [5, 8] for np_ in [4] for ov in [0, 2] for kp in [3]
    ],
    "CURSelector": [
        {"n_d": nd, "num_basis": nb, "tolerance": 1e-5, "c_cols": None, "r_rows": None, "deterministic": True}
        for nd in [3, 5] for nb in [5, 8]
    ],
}
seeds = [0, 1]  # adjust as needed

# ---------- Run the grid ----------
results_csv = "./data/deepymod/burgers/grid_results.csv"
os.makedirs(os.path.dirname(results_csv), exist_ok=True)
with open(results_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sampler","seed","sampler_kwargs","train_time_s","final_loss","precision","recall","f1","coeff_l2"])

    for s in samplers:
        for kw in grid[s]:
            for sd in seeds:
                print(f"\n>>> Running {s} | seed={sd} | {kw}")
                meta = train_one_combo(s, kw, seed=sd, max_iter=1)
                writer.writerow([
                    meta["sampler"], meta["seed"], json.dumps(meta["sampler_kwargs"], sort_keys=True),
                    meta["train_time_s"], meta["final_loss"], meta["precision"], meta["recall"], meta["f1"], meta["coeff_l2"]
                ])

print(f"\nGrid search complete. CSV at: {results_csv}")












