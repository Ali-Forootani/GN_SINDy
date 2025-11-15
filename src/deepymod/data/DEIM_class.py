#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 08:25:47 2023

@author: forootani
"""


import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


root_dir = setting_directory(1)


from pathlib import Path
import torch
from scipy import linalg
import torch.nn as nn
import torch.nn.init as init

"""
from Functions.modules import Siren
from Functions.utils import loss_func_AC
from Functions.utils import leastsquares_fit
from Functions.utils import equation_residual_AC
from Functions.library import library_deriv
from Functions import plot_config_file
"""


from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time
import numpy as np
from scipy.linalg import svd, qr
import itertools

############################################


# cwd = os.getcwd()
# sys.path.append(cwd + '/my_directory')
# sys.path.append(cwd)


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


"""
####### Some examples of how to load the data and split it


data = scipy.io.loadmat(root_dir + "/data/AC.mat")
t_o = data["tt"].flatten()[0:201, None]
x_o = data["x"].flatten()[:, None]
Exact = np.real(data["uu"])



data = scipy.io.loadmat(root_dir + '/data/kdv.mat')
t_o = data["t"].flatten()[:, None]
x_o = data["x"].flatten()[:, None]
Exact = np.real(data["usol"][:,:])
"""


###################################################
###################################################
###################################################


class DEIM:
    def __init__(self, X, n_d, t_o, x_o, tolerance = 1e-5,num_basis=20):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.x_o = x_o
        self.tolerance = tolerance
        # self.dec_rate = dec_rate

    def deim(self, X, i):
        U, Sigma, Vt = svd(X, full_matrices=False)

        # Step 2: Select the basis functions
        # k = (self.num_basis - self.dec_rate * 2)  # Number of basis functions to retain

        k = self.num_basis

        precision = 1 - np.sum(Sigma[:k]) / (np.sum(Sigma))
        # print(precision)
        while precision >= self.tolerance:
            k = k + 1
            precision = 1 - np.sum(Sigma[:k]) / np.sum(Sigma)
        #print(k)

        # Step 3: Compute the SVD-based approximation
        Uk = U[:, :k]
        Sigma_k = np.diag(Sigma[:k])
        Vk_t = Vt[:k, :]

        X_k = Uk @ Sigma_k @ Vk_t

        left = Uk @ np.sqrt(Sigma_k)
        right = np.sqrt(Sigma_k) @ Vk_t

        q_x, r_x, p_x = qr(Uk.T, mode="economic", pivoting=True)
        i_x = p_x[:k]

        q_t, r_t, p_t = qr(Vk_t, mode="economic", pivoting=True)
        i_t = p_t[:k]
        
        
        return i_t, i_x

    def execute(self):
        n_k = self.X.shape[1]
        n_s = int(n_k / self.n_d)

        for i in range(self.n_d):
            i_tf, i_xf = self.deim(self.X[:, i * n_s : (i + 1) * n_s], i)
            i_tf = i_tf + i * n_s
            self.i_t.append([i_tf])
            self.i_x.append([i_xf])

            space_o, T_o = np.meshgrid(self.x_o, self.t_o, indexing="ij")

            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o)

            #########################

            t, space = np.meshgrid(i_tf, i_xf, indexing="ij")

            self.u_selected.append(self.X[space, t])

            self.t_sampled.append(T_o[space, t])
            self.x_sampled.append(space_o[space, t])

            X_star = np.hstack((t.flatten()[:, None], space.flatten()[:, None]))
            
            
            # plt.scatter(X_star[:,0], X_star[:,1])
            #plt.scatter(X_star[:, 0], X_star[:, 1], c=self.X[space, t])
            # plt.ylim([-50,600])

            ############################

            self.S_star.append(self.x_sampled[i].flatten())
            self.T_star.append(self.t_sampled[i].flatten())

            self.U_star.append(self.u_selected[i].flatten())

        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)

        self.coords = np.hstack((S_s, T_s))

        return S_s, T_s, U_s


"""
If you want to make use of this Class locally just use the following syntaxes
"""

#deim_instance = DEIM(Exact[:,0:200], 1, t_o, x_o, num_basis = 1)
#S_s, T_s, U_s_1 = deim_instance.execute()



# deim_instance = DEIM(Exact[:,:], 5, t_o, x_o, num_basis = 2)
# S_s, T_s, U_s_2 = deim_instance.execute()


# fig, ax = plt.subplots()
# im = ax.scatter(T_s, S_s, c=U_s_2)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# fig.colorbar(mappable=im)
# plt.show()


# print(U_s_2)


# S_star = deim_instance.S_star
# T_star = deim_instance.T_star
# U_star = deim_instance.U_star
# coords = deim_instance.coords

# print(U_s)
# print(T_s)
# print(S_s)



import numpy as np
from scipy.linalg import svd, qr

# ---------- helpers ----------

def choose_k_by_energy(Sigma, tol, k_min=1):
    """Increase k until captured energy >= 1 - tol."""
    k = max(k_min, 1)
    total = np.sum(Sigma)
    if total <= 0:
        return k
    captured = np.sum(Sigma[:k])
    while (1 - captured / total) > tol and k < len(Sigma):
        k += 1
        captured = np.sum(Sigma[:k])
    return k

def rsvd(X, k, n_oversample=10, n_power=1, rng=None):
    """
    Randomized SVD: returns U(:, :k), s[:k], Vt[:k, :].
    """
    rng = np.random.default_rng(None if rng is None else rng)
    m, n = X.shape
    l = min(k + n_oversample, min(m, n))
    Omega = rng.standard_normal(size=(n, l))
    Y = X @ Omega
    for _ in range(n_power):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = Q.T @ X
    Ub, s, Vt = svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k, :]

# ---------- 1) Localized DEIM (L-DEIM) ----------

class LDEIM:
    """
    Localized DEIM over spatial patches.
    Same constructor + execute() signature as your DEIM.
    Extra args:
      n_patches: number of spatial patches
      overlap:   number of spatial indices to overlap between neighboring patches
      k_per_patch: basis per patch (if None, split num_basis approximately)
    """
    def __init__(self, X, n_d, t_o, x_o, tolerance=1e-5, num_basis=20,
                 n_patches=4, overlap=0, k_per_patch=None):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.x_o = x_o
        self.tolerance = tolerance
        self.n_patches = max(1, int(n_patches))
        self.overlap = max(0, int(overlap))
        self.k_per_patch = k_per_patch

    def _patch_bounds(self, n_space):
        """Build overlapping patch index ranges that cover [0, n_space)."""
        p = self.n_patches
        base = n_space // p
        rem = n_space % p
        starts = []
        s = 0
        for i in range(p):
            size = base + (1 if i < rem else 0)
            starts.append(s)
            s += size
        bounds = []
        for i, st in enumerate(starts):
            ed = starts[i + 1] if i + 1 < len(starts) else n_space
            a = max(0, st - (self.overlap if i > 0 else 0))
            b = min(n_space, ed + (self.overlap if i + 1 < len(starts) else 0))
            bounds.append((a, b))
        return bounds

    def _patch_deim_indices(self, X_block, k_target):
        # SVD of block
        U, Sigma, Vt = svd(X_block, full_matrices=False)
        k = choose_k_by_energy(Sigma, self.tolerance, k_min=min(1, k_target))
        k = min(k, k_target, U.shape[1], Vt.shape[0])
        Uk = U[:, :k]
        Vk_t = Vt[:k, :]
        # DEIM via pivoted QR
        _, _, p_x = qr(Uk.T, mode="economic", pivoting=True)
        i_x = p_x[:k]
        _, _, p_t = qr(Vk_t, mode="economic", pivoting=True)
        i_t = p_t[:k]
        return i_t, i_x, k

    def execute(self):
        n_space, n_k = self.X.shape
        n_s = int(n_k / self.n_d)
        space_o, T_o_full = np.meshgrid(self.x_o, self.t_o, indexing="ij")

        for d in range(self.n_d):
            X_chunk = self.X[:, d * n_s:(d + 1) * n_s]

            # Determine per-patch budget
            if self.k_per_patch is not None:
                kpp = self.k_per_patch
            else:
                kpp = max(1, int(np.ceil(self.num_basis / self.n_patches)))

            # Build patch-wise selections and merge
            bounds = self._patch_bounds(n_space)
            sel_x = []
            sel_t = []
            for (a, b) in bounds:
                i_t_loc, i_x_loc, k_loc = self._patch_deim_indices(X_chunk[a:b, :], kpp)
                sel_x.append((a + i_x_loc).astype(int))
                sel_t.append(i_t_loc.astype(int))

            # Merge, keep unique and cap to budget
            i_x_all = np.unique(np.concatenate(sel_x)) if sel_x else np.array([], dtype=int)
            i_t_all = np.unique(np.concatenate(sel_t)) if sel_t else np.array([], dtype=int)

            # Safety: ensure we don't exceed total dims or budget
            if len(i_x_all) > self.num_basis:
                i_x_all = i_x_all[:self.num_basis]
            if len(i_t_all) > self.num_basis:
                i_t_all = i_t_all[:self.num_basis]

            # If one side is shorter, pad by running global DEIM to fill
            need = max(0, len(i_x_all) - len(i_t_all))
            if need > 0:
                # add more time samples via global pivoting on V
                Uc, Sc, Vtc = svd(X_chunk, full_matrices=False)
                kc = min(choose_k_by_energy(Sc, self.tolerance), self.num_basis, Vtc.shape[0])
                _, _, p_tg = qr(Vtc[:kc, :], mode="economic", pivoting=True)
                extra = [idx for idx in p_tg if idx not in i_t_all][:need]
                i_t_all = np.asarray(list(i_t_all) + extra, dtype=int)

            need = max(0, len(i_t_all) - len(i_x_all))
            if need > 0:
                Uc, Sc, Vtc = svd(X_chunk, full_matrices=False)
                kc = min(choose_k_by_energy(Sc, self.tolerance), self.num_basis, Uc.shape[1])
                _, _, p_xg = qr(Uc[:, :kc].T, mode="economic", pivoting=True)
                extra = [idx for idx in p_xg if idx not in i_x_all][:need]
                i_x_all = np.asarray(list(i_x_all) + extra, dtype=int)

            # Final guards
            k_final = min(len(i_x_all), len(i_t_all))
            i_xf = i_x_all[:k_final]
            i_tf = i_t_all[:k_final] + d * n_s

            self.i_t.append([i_tf])
            self.i_x.append([i_xf])

            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o_full)

            t_idx, x_idx = np.meshgrid(i_tf, i_xf, indexing="ij")
            self.u_selected.append(self.X[x_idx, t_idx])

            self.t_sampled.append(T_o_full[x_idx, t_idx])
            self.x_sampled.append(space_o[x_idx, t_idx])

            self.S_star.append(self.x_sampled[d].flatten())
            self.T_star.append(self.t_sampled[d].flatten())
            self.U_star.append(self.u_selected[d].flatten())

        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)
        self.coords = np.hstack((S_s, T_s))
        return S_s, T_s, U_s

# ---------- 2) Randomized DEIM ----------

class RandDEIM:
    """
    DEIM with randomized SVD backbone.
    Same constructor + execute() signature as your DEIM.
    Extra args:
      oversample: RSVD oversampling
      n_power:    number of power iterations
      rng:        seed or Generator for reproducibility
    """
    def __init__(self, X, n_d, t_o, x_o, tolerance=1e-5, num_basis=20,
                 oversample=10, n_power=1, rng=None):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.x_o = x_o
        self.tolerance = tolerance
        self.oversample = oversample
        self.n_power = n_power
        self.rng = rng

    def _deim_indices(self, X):
        # estimate k by a quick pass using standard SVD on a thin sketch if needed
        # but we can choose k = num_basis, then refine upward if tol not met
        # Do one RSVD with k=num_basis + oversample, then trim using tol
        k_try = min(self.num_basis + self.oversample, min(X.shape))
        U, s, Vt = rsvd(X, k_try, n_oversample=0, n_power=self.n_power, rng=self.rng)
        # s is length k_try in descending order
        k = choose_k_by_energy(s, self.tolerance, k_min=min(1, self.num_basis))
        k = min(k, self.num_basis, U.shape[1], Vt.shape[0])
        Uk = U[:, :k]
        Vk_t = Vt[:k, :]
        # pivoted QR for DEIM indices
        _, _, p_x = qr(Uk.T, mode="economic", pivoting=True)
        i_x = p_x[:k]
        _, _, p_t = qr(Vk_t, mode="economic", pivoting=True)
        i_t = p_t[:k]
        return i_t, i_x

    def execute(self):
        n_space, n_k = self.X.shape
        n_s = int(n_k / self.n_d)
        space_o, T_o_full = np.meshgrid(self.x_o, self.t_o, indexing="ij")

        for d in range(self.n_d):
            X_chunk = self.X[:, d * n_s:(d + 1) * n_s]
            i_tf_rel, i_xf = self._deim_indices(X_chunk)
            i_tf = i_tf_rel + d * n_s

            self.i_t.append([i_tf])
            self.i_x.append([i_xf])

            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o_full)

            t_idx, x_idx = np.meshgrid(i_tf, i_xf, indexing="ij")
            self.u_selected.append(self.X[x_idx, t_idx])

            self.t_sampled.append(T_o_full[x_idx, t_idx])
            self.x_sampled.append(space_o[x_idx, t_idx])

            self.S_star.append(self.x_sampled[d].flatten())
            self.T_star.append(self.t_sampled[d].flatten())
            self.U_star.append(self.u_selected[d].flatten())

        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)
        self.coords = np.hstack((S_s, T_s))
        return S_s, T_s, U_s

# ---------- 3) CUR via leverage scores ----------

class CURSelector:
    """
    Leverage-score CUR sampler.
    Same constructor + execute() signature as your DEIM.
    Extra args:
      c_cols: number of columns (time indices) to select (default: num_basis)
      r_rows: number of rows   (space indices) to select (default: num_basis)
      deterministic: pick top probabilities instead of sampling
      rng: for stochastic selection
    Notes:
      - Uses top-k SVD to compute leverage scores for rows (U_k) and columns (V_k).
      - Indices are chosen without replacement.
    """
    def __init__(self, X, n_d, t_o, x_o, tolerance=1e-5, num_basis=20,
                 c_cols=None, r_rows=None, deterministic=True, rng=None):
        self.X = X
        self.n_d = n_d
        self.num_basis = num_basis
        self.i_t = []
        self.i_x = []
        self.u_selected = []
        self.t_sampled = []
        self.x_sampled = []
        self.X_sampled = []
        self.T_sampled = []
        self.S_star = []
        self.T_star = []
        self.U_star = []
        self.coords = None
        self.t_o = t_o
        self.x_o = x_o
        self.tolerance = tolerance
        self.c_cols = c_cols  # time
        self.r_rows = r_rows  # space
        self.deterministic = deterministic
        self.rng = np.random.default_rng(None if rng is None else rng)

    def _leverage_select(self, probs, k, deterministic=True):
        """Select k indices by leverage probs (top-k or sample w/out replacement)."""
        probs = np.maximum(probs, 1e-16)
        probs = probs / probs.sum()
        n = len(probs)
        if deterministic:
            return np.argsort(probs)[::-1][:k]
        else:
            # sample without replacement proportional to probs
            return self.rng.choice(n, size=min(k, n), replace=False, p=probs)

    def _cur_indices(self, X, k_target):
        # SVD
        U, s, Vt = svd(X, full_matrices=False)
        k = choose_k_by_energy(s, self.tolerance, k_min=min(1, k_target))
        k = min(k, k_target, U.shape[1], Vt.shape[0])
        Uk = U[:, :k]
        Vk = Vt[:k, :].T  # shape (n_time, k)

        # row (space) leverage scores: row-wise squared norms of Uk
        row_lev = np.sum(Uk**2, axis=1)
        # col (time) leverage scores: row-wise squared norms of Vk (i.e., col leverage of X)
        col_lev = np.sum(Vk**2, axis=1)

        r_rows = self.r_rows if self.r_rows is not None else self.num_basis
        c_cols = self.c_cols if self.c_cols is not None else self.num_basis

        i_x = self._leverage_select(row_lev, r_rows, deterministic=self.deterministic)
        i_t_rel = self._leverage_select(col_lev, c_cols, deterministic=self.deterministic)

        # To keep parity with DEIM (square sampling), trim to same length
        k_final = min(len(i_x), len(i_t_rel))
        return i_t_rel[:k_final], i_x[:k_final]

    def execute(self):
        n_space, n_k = self.X.shape
        n_s = int(n_k / self.n_d)
        space_o, T_o_full = np.meshgrid(self.x_o, self.t_o, indexing="ij")

        for d in range(self.n_d):
            X_chunk = self.X[:, d * n_s:(d + 1) * n_s]
            i_tf_rel, i_xf = self._cur_indices(X_chunk, k_target=self.num_basis)
            i_tf = i_tf_rel + d * n_s

            self.i_t.append([i_tf])
            self.i_x.append([i_xf])

            self.X_sampled.append(space_o)
            self.T_sampled.append(T_o_full)

            t_idx, x_idx = np.meshgrid(i_tf, i_xf, indexing="ij")
            self.u_selected.append(self.X[x_idx, t_idx])

            self.t_sampled.append(T_o_full[x_idx, t_idx])
            self.x_sampled.append(space_o[x_idx, t_idx])

            self.S_star.append(self.x_sampled[d].flatten())
            self.T_star.append(self.t_sampled[d].flatten())
            self.U_star.append(self.u_selected[d].flatten())

        S_s = np.concatenate(self.S_star, axis=0).reshape(-1, 1)
        T_s = np.concatenate(self.T_star, axis=0).reshape(-1, 1)
        U_s = np.concatenate(self.U_star, axis=0).reshape(-1, 1)
        self.coords = np.hstack((S_s, T_s))
        return S_s, T_s, U_s



