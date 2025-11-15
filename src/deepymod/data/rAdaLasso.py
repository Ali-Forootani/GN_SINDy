#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:00:05 2025

@author: forootan
"""

import numpy as np
from numpy.linalg import inv
from typing import Tuple, Optional, Dict
from sklearn.linear_model import Ridge, Lasso
from sklearn.utils import check_random_state

# ----------------------------- utilities ------------------------------------
def _central_grad_x(u: np.ndarray, dx: float, order: int = 1) -> np.ndarray:
    """Central differences in x; u shape (nx, nt)."""
    g = u.copy()
    for _ in range(order):
        # second-order central differences; Neumann at boundaries
        gx = np.empty_like(g)
        gx[1:-1, :] = (g[2:, :] - g[:-2, :])/(2.0*dx)
        gx[0,  :]   = (g[1,  :] - g[0,   :])/dx
        gx[-1, :]   = (g[-1, :] - g[-2,  :])/dx
        g = gx
    return g

def _central_grad_t(u: np.ndarray, dt: float, order: int = 1) -> np.ndarray:
    """Central differences in t; u shape (nx, nt)."""
    g = u.copy()
    for _ in range(order):
        gt = np.empty_like(g)
        gt[:, 1:-1] = (g[:, 2:] - g[:, :-2])/(2.0*dt)
        gt[:, 0]    = (g[:, 1]  - g[:, 0])/dt
        gt[:, -1]   = (g[:, -1] - g[:, -2])/dt
        g = gt
    return g

def _build_library_1d(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    poly_order: int = 2,
    diff_order: int = 3
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Construct a standard 1D library: {1, u, u^2, ..., u^P} x {∂_x^k u, k=0..D}
    plus mixed products u^p * ∂_x^k u (like DeepMoD/Library1D style).
    Returns (Theta, ut, columns) with shapes flattened to (n_samples, p).
    """
    nx, nt = u.shape
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    # derivatives in x up to diff_order
    derivs = {0: u}
    for k in range(1, diff_order+1):
        derivs[k] = _central_grad_x(derivs[k-1], dx, order=1)

    ut = _central_grad_t(u, dt, order=1)

    # basis columns
    cols = {}
    # pure derivatives
    for k in range(0, diff_order+1):
        cols[f"u_x^{k}"] = derivs[k]

    # pure polynomials u^p
    for p in range(1, poly_order+1):
        cols[f"u^{p}"] = u**p

    # mixed: u^p * u_x^k (exclude trivial (p=0,k=0) since captured by '1' and 'u')
    for p in range(1, poly_order+1):
        for k in range(1, diff_order+1):
            cols[f"u^{p}*u_x^{k}"] = (u**p) * derivs[k]

    # constant column
    cols["1"] = np.ones_like(u)

    names = list(cols.keys())
    Theta = np.stack([cols[name] for name in names], axis=-1).reshape(nx*nt, -1)
    y = ut.reshape(nx*nt)
    return Theta, y, cols

def _randomized_adaptive_lasso_stability(
    Theta: np.ndarray,
    y: np.ndarray,
    B: int = 40,
    pi_thr: float = 0.9,
    EV_max: int = 3,
    lambda_path_len: int = 30,
    ridge_alpha: float = 1e-6,
    beta_a: float = 1.0, beta_b: float = 2.0,
    subsample_frac: float = 0.5,
    random_state: Optional[int] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    rAdaLasso with stability selection and error control (Sec. 2.4, App. B).
    Returns:
      support_mask: boolean (p,) stable support
      sel_probs:    (p,) max stability π_k across admissible λ ∈ Λ*
    """
    rng = check_random_state(random_state)
    n, p = Theta.shape

    # Step 1: Adaptive weights via Ridge (Sec. 2.2)
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(Theta, y)
    coef0 = ridge.coef_
    w = 1.0 / (np.abs(coef0) + 1e-12)**2  # γ=2
    # Reweight design: Θ̃_i = Θ_i / w_i
    Theta_tilde = Theta / w

    # Step 2: build λ-path on Θ̃ (common across subsamples)
    # A rough λ_max where Lasso makes all-zero (heuristic via  ||Θ^T y||_∞ / n)
    lam_max = np.linalg.norm(Theta_tilde.T @ y, ord=np.inf) / max(1, n)
    lam_min = lam_max * 1e-3
    lambdas = np.exp(np.linspace(np.log(lam_max), np.log(lam_min), lambda_path_len))

    # storage: selection counts Π̂_k(λ)
    sel_counts = np.zeros((lambda_path_len, p), dtype=float)
    q_lambda = np.zeros(lambda_path_len, dtype=float)  # avg selected per λ

    half = int(np.ceil(subsample_frac * n))

    for b in range(B):
        # Subsample without replacement
        idx = rng.choice(n, size=half, replace=False)
        Xb = Theta_tilde[idx]
        yb = y[idx]

        # Randomize penalty per feature: Wi ~ Beta(1,2)  (Sec. 2.4)
        Wi = rng.beta(beta_a, beta_b, size=p) + 1e-12  # avoid zero

        for li, lam in enumerate(lambdas):
            # Equivalent to Lasso on Xb * diag(1/sqrt(Wi)) with α scaled
            # Implement as feature-wise scaling of X, and backscale coef.
            X_scaled = Xb / Wi[np.newaxis, :]
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=5000)
            lasso.fit(X_scaled, yb)
            coef = lasso.coef_ / Wi  # undo scaling
            selected = np.abs(coef) > 1e-12
            sel_counts[li] += selected.astype(float)
            q_lambda[li] += selected.sum()

    # convert counts → probabilities Π̂_k(λ)
    Pi = sel_counts / float(B)
    q_lambda = q_lambda / float(B)

    # Step 3: Error control — restrict Λ to Λ* (App. B, Eq. (14))
    # Use upper bound E[V] ≤ q_Λ^2 / ((2 π_thr - 1) p)
    admissible = []
    for li in range(lambda_path_len):
        EV_upper = (q_lambda[li]**2) / (max((2.0*pi_thr - 1.0), 1e-6) * p)
        if EV_upper <= EV_max:
            admissible.append(li)

    if len(admissible) == 0:
        # fallback: take the single λ with smallest EV_upper
        EVs = (q_lambda**2) / (max((2.0*pi_thr - 1.0), 1e-6) * p)
        admissible = [int(np.argmin(EVs))]

    Pi_adm = Pi[admissible]          # (L*, p)
    max_probs = Pi_adm.max(axis=0)   # Π̂_k^* = max_λ∈Λ* Π̂_k(λ)
    support_mask = max_probs >= pi_thr
    return support_mask, max_probs

def _ridge_leverage_scores(Theta_S: np.ndarray, alpha: float = 1e-8) -> np.ndarray:
    """
    Ridge leverage scores h_i = diag(Θ_S (Θ_S^T Θ_S + α I)^(-1) Θ_S^T)
    Used to sample informative rows once the support S is fixed.
    """
    # (n x |S|)
    G = Theta_S.T @ Theta_S
    H = Theta_S @ inv(G + alpha * np.eye(G.shape[0])) @ Theta_S.T
    return np.clip(np.diag(H), 0.0, None)  # (n,)

# ----------------------- rAdaLasso sampler class -----------------------------
class RadaLassoSampler:
    """
    Randomized-Adaptive Lasso sampler:
      1) Build library Θ(u, ux, uxx, ...; u^p, u^p * u_x^k) from snapshot X
      2) rAdaLasso + stability selection → stable support S  (Sec. 2.4, App. B)
      3) Row sampling by ridge leverage on Θ[:, S]
      4) Return S_s, T_s, U_s (like your DEIM), to feed your DNN later
    References: rAdaLasso, stability selection, and EV control as in the paper. :contentReference[oaicite:2]{index=2}
    """
    def __init__(
        self,
        X: np.ndarray,             # snapshot matrix u(x_i, t_j), shape (nx, nt)
        t_o: np.ndarray,           # shape (nt,)
        x_o: np.ndarray,           # shape (nx,)
        num_samples: int = 1000,
        poly_order: int = 2,
        diff_order: int = 3,
        # rAdaLasso / stability params
        B: int = 40,
        pi_thr: float = 0.9,
        EV_max: int = 3,
        lambda_path_len: int = 30,
        ridge_alpha_adapt: float = 1e-6,
        ridge_alpha_leverage: float = 1e-8,
        random_state: Optional[int] = 0,
    ):
        self.X = np.asarray(X, dtype=float)
        self.t_o = np.asarray(t_o, dtype=float)
        self.x_o = np.asarray(x_o, dtype=float)
        self.num_samples = int(num_samples)
        self.poly_order = int(poly_order)
        self.diff_order = int(diff_order)
        self.B = int(B)
        self.pi_thr = float(pi_thr)
        self.EV_max = int(EV_max)
        self.lambda_path_len = int(lambda_path_len)
        self.ridge_alpha_adapt = float(ridge_alpha_adapt)
        self.ridge_alpha_leverage = float(ridge_alpha_leverage)
        self.random_state = random_state

        self._nx, self._nt = self.X.shape
        assert self.t_o.shape[0] == self._nt and self.x_o.shape[0] == self._nx, \
            "Shapes of X, x_o, t_o must align."

        # outputs
        self.S_star = None
        self.T_star = None
        self.U_star = None

    def execute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          S_s: (m,1) sampled x's
          T_s: (m,1) sampled t's
          U_s: (m,1) sampled u values
        """
        # 1) Build Θ and ut (finite-diff; only for sampler; DNN will recompute via AD later)
        Theta, ut, _ = _build_library_1d(
            self.X, self.x_o, self.t_o,
            poly_order=self.poly_order, diff_order=self.diff_order
        )
        n, p = Theta.shape

        # 2) rAdaLasso stability selection with EV control -> stable support S
        support, probs = _randomized_adaptive_lasso_stability(
            Theta, ut,
            B=self.B, pi_thr=self.pi_thr, EV_max=self.EV_max,
            lambda_path_len=self.lambda_path_len,
            ridge_alpha=self.ridge_alpha_adapt,
            random_state=self.random_state
        )

        # Fallback if nothing passes the threshold: take top-K by prob (very rare)
        if not np.any(support):
            k = max(1, min(5, p//4))
            topk = np.argsort(probs)[-k:]
            support[topk] = True

        Theta_S = Theta[:, support]  # (n, |S|)

        # 3) Row scores via ridge leverage; sample without replacement ∝ leverage
        lev = _ridge_leverage_scores(Theta_S, alpha=self.ridge_alpha_leverage)
        lev = np.clip(lev, 1e-12, None)
        p_row = lev / lev.sum()
        rng = check_random_state(self.random_state)
        m = min(self.num_samples, n)
        idx_flat = rng.choice(n, size=m, replace=False, p=p_row)

        # map flat indices back to (ix, it)
        ix = idx_flat // self._nt
        it = idx_flat %  self._nt

        # 4) Pack like DEIM: concatenate across temporal blocks
        S_s = self.x_o[ix].reshape(-1, 1)
        T_s = self.t_o[it].reshape(-1, 1)
        U_s = self.X[ix, it].reshape(-1, 1)

        self.S_star, self.T_star, self.U_star = S_s, T_s, U_s
        return S_s, T_s, U_s
