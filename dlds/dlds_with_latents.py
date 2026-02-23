# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:37:03 2026

@author: noga mudrik
"""

# -*- coding: utf-8 -*-
"""
dLDS with Latents (Indirect Observation)
========================================

dLDS model for the indirect observation case (D ≠ I).

Observation:  y_t = D @ x_t + ε_t
Dynamics:     x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1} + ν_t

Author: Noga Mudrik

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│  dLDS_with_latents                                              │
├─────────────────────────────────────────────────────────────────┤
│  Learns: F (dynamics), c (coefficients), D (observation), x     │
├─────────────────────────────────────────────────────────────────┤
│  METHODS:                                                       │
│    fit(y)              → Main training loop                     │
│    reconstruct(y)      → Reconstruct observations               │
│    get_latents(y)      → Extract latent states                  │
└─────────────────────────────────────────────────────────────────┘

FIT() FLOW:
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: y (k x T) observations                                   │
└──────────────────┬───────────────────────────────────────────────┘
                   ▼
         ┌─────────────────┐
         │  TRAINING LOOP  │
         └────────┬────────┘
                  ▼
    ┌─────────────────────────────┐
    │  1. _update_x(y, D, F, c)   │  Infer latents from observations
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  2. _update_c(x, F)         │  Solve for coefficients
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  3. _update_d(y, x)         │  Update observation matrix
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  4. _update_f(x, F, c)      │  Gradient descent on F
    └──────────────┬──────────────┘
                   ▼
         ┌────────────────┐
         │  Converged?    │──No──→ Loop back to 1
         └───────┬────────┘
                 │ Yes
                 ▼
    ┌─────────────────────────────┐
    │  Final _update_c, _update_x │  Polish estimates
    └──────────────┬──────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: F, c, D, x                                              │
└──────────────────────────────────────────────────────────────────┘

TODO - INFER C AND X TOGETHER


"""
print('reloadjj')
import numpy as np
from scipy import linalg
import warnings
from typing import List, Optional, Tuple

from .config import dLDS_config
from .results import dLDS_latent_result
from .solvers import solve_lasso_style, solve_coefficients_single_time
from .dlds_utils import init_mat, norm_mat, validate_data, init_distant_f, center_data

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class dLDS_with_latents:
    """
    dLDS model with latent states (D ≠ I).
    
    y_t = D @ x_t
    x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1}
    """
    
    def __init__(self, config=None, **kwargs):
        assert config is None or isinstance(config, (dLDS_config, dict)), "config must be dLDS_config, dict, or None, got %s" % type(config).__name__
        if config is None:
            self.config = dLDS_config(**kwargs)
        elif isinstance(config, dLDS_config):
            if kwargs:
                config_dict = config.to_dict()
                config_dict.update(kwargs)
                self.config = dLDS_config(**config_dict)
            else:
                self.config = config
        elif isinstance(config, dict):
            config.update(kwargs)
            self.config = dLDS_config(**config)
        else:
            raise TypeError("config must be dLDS_config, dict, or None. Got %s" % type(config).__name__)
        
        self.F_ = None
        self.D_ = None
        self.x_ = None
        self.dynamic_coefficients_ = None
        self.result_ = None
        self._is_fitted = False
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_d(self, obs_dim, latent_dim, init_type='random'):
        """
        Initialize observation matrix D.
        
        Parameters
        ----------
        obs_dim : int
            Observation dimension (k).
        latent_dim : int
            Latent dimension (p).
        init_type : str
            'random', 'pca', or np.ndarray for custom init.
        """
        assert obs_dim > 0 and latent_dim > 0, "obs_dim and latent_dim must be positive, got obs_dim=%d, latent_dim=%d" % (obs_dim, latent_dim)
        assert obs_dim >= latent_dim, "obs_dim (%d) must be >= latent_dim (%d)" % (obs_dim, latent_dim)
        assert isinstance(init_type, (str, np.ndarray)), "init_type must be str or np.ndarray, got %s" % type(init_type).__name__
        
        if isinstance(init_type, np.ndarray):
            D = init_type.copy()
            assert D.shape == (obs_dim, latent_dim), "D shape must be (%d, %d), got %s" % (obs_dim, latent_dim, D.shape)
            assert not np.isnan(D).any(), "init_type D contains NaN values"
            assert not np.isinf(D).any(), "init_type D contains Inf values"
        elif init_type == 'random':
            np.random.seed(self.config.seed)
            D = np.random.randn(obs_dim, latent_dim)
            assert D.shape == (obs_dim, latent_dim), "Random D shape must be (%d, %d), got %s" % (obs_dim, latent_dim, D.shape)
        elif init_type == 'pca':
            # Will be set later with data
            D = None
        else:
            raise ValueError("init_type must be 'random', 'pca', or np.ndarray. Got %s" % init_type)
        
        return D
    
    def _init_d_pca(self, y_data):
        """Initialize D using PCA on observations."""
        assert isinstance(y_data, (list, np.ndarray)), "y_data must be list or np.ndarray, got %s" % type(y_data).__name__
        if isinstance(y_data, list):
            assert len(y_data) > 0, "y_data list must be non-empty"
            y_concat = np.hstack(y_data)
        else:
            y_concat = y_data
        
        assert y_concat.ndim == 2, "y_concat must be 2D, got shape %s" % (y_concat.shape,)
        assert self.config.latent_dim, 'self.config.latent_dim cannot be none?'
        assert y_concat.shape[0] >= self.config.latent_dim, "y_concat rows (%d) must be >= latent_dim (%d)" % (y_concat.shape[0], self.config.latent_dim)
        assert y_concat.shape[1] > 0, "y_concat must have at least 1 time point, got %d" % y_concat.shape[1]
        
        # PCA: D = top p eigenvectors of y @ y.T
        cov = y_concat @ y_concat.T
        assert cov.shape == (y_concat.shape[0], y_concat.shape[0]), "Covariance shape mismatch: expected (%d, %d), got %s" % (y_concat.shape[0], y_concat.shape[0], cov.shape)
        eigenvalues, eigenvectors = linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        D = eigenvectors[:, idx[:self.config.latent_dim]]
        assert D.shape == (y_concat.shape[0], self.config.latent_dim), "PCA D shape must be (%d, %d), got %s" % (y_concat.shape[0], self.config.latent_dim, D.shape)
        assert not np.isnan(D).any(), "PCA D contains NaN values"
        return D
    
    def _init_x(self, y_data, D):
        """Initialize latent states x from observations y using pseudoinverse of D."""
        assert D is not None, "D must not be None"
        assert D.ndim == 2, "D must be 2D, got shape %s" % (D.shape,)
        assert isinstance(y_data, (list, np.ndarray)), "y_data must be list or np.ndarray, got %s" % type(y_data).__name__
        D_pinv = linalg.pinv(D)
        assert D_pinv.shape == (D.shape[1], D.shape[0]), "D_pinv shape must be (%d, %d), got %s" % (D.shape[1], D.shape[0], D_pinv.shape)
        
        if isinstance(y_data, list):
            assert len(y_data) > 0, "y_data list must be non-empty"
            x_list = [D_pinv @ y for y in y_data]
            assert all(x.shape[0] == D.shape[1] for x in x_list), "All x must have shape[0] == latent_dim (%d)" % D.shape[1]
        else:
            x_list = [D_pinv @ y_data]
            assert x_list[0].shape[0] == D.shape[1], "x must have shape[0] == latent_dim (%d), got %d" % (D.shape[1], x_list[0].shape[0])
        
        return x_list
    
    def _init_f(self, latent_dim):
        """Initialize dynamics operators F."""
        assert latent_dim > 0, "latent_dim must be positive, got %d" % latent_dim
        num_subdyns = self.config.num_subdyns
        assert num_subdyns > 0, "num_subdyns must be positive, got %d" % num_subdyns
        
        if self.config.init_distant_f and num_subdyns > 1:
            F, achieved_corr = init_distant_f(
                latent_dim=latent_dim,
                num_subdyns=num_subdyns,
                max_corr=self.config.max_corr,
                max_retries=self.config.max_init_retries,
                normalize=self.config.normalize_eig,
                seed=self.config.seed
            )
            assert len(F) == num_subdyns, "init_distant_f returned %d matrices, expected %d" % (len(F), num_subdyns)
        else:
            F = [init_mat((latent_dim, latent_dim), r_seed=self.config.seed + i, normalize=self.config.normalize_eig)
                 for i in range(num_subdyns)]
        
        assert len(F) == num_subdyns, "F must have %d matrices, got %d" % (num_subdyns, len(F))
        assert all(f.shape == (latent_dim, latent_dim) for f in F), "All F matrices must be (%d, %d)" % (latent_dim, latent_dim)
        assert all(not np.isnan(f).any() for f in F), "F contains NaN values"
        return F
    
    def _init_c(self, trial_lengths):
        """Initialize coefficients."""
        assert isinstance(trial_lengths, list), "trial_lengths must be list, got %s" % type(trial_lengths).__name__
        assert len(trial_lengths) > 0, "trial_lengths must be non-empty"
        assert all(T > 1 for T in trial_lengths), "All trial lengths must be > 1, got %s" % trial_lengths
        num_subdyns = self.config.num_subdyns
        assert num_subdyns > 0, "num_subdyns must be positive, got %d" % num_subdyns
        coefficients = []
        for i, T in enumerate(trial_lengths):
            c = init_mat((num_subdyns, T - 1), r_seed=self.config.seed + 1000 + i)
            assert c.shape == (num_subdyns, T - 1), "c shape must be (%d, %d), got %s" % (num_subdyns, T - 1, c.shape)
            coefficients.append(c)
        assert len(coefficients) == len(trial_lengths), "coefficients length (%d) must match trial_lengths (%d)" % (len(coefficients), len(trial_lengths))
        return coefficients
    
    # =========================================================================
    # Updates
    # =========================================================================
    
    def _update_x(self, y_list, D, F, coefficients, x_prev):
        """
        Update latent states x given observations y, D, F, c.
        
        Solves jointly:
            [D                    ] @ x_all = [y_all]
            [sqrt(λ) * A_dynamics ]           [0    ]
        
        Where A_dynamics encodes: x_{t+1} = F_t @ x_t
        TODO: INFER C AND X TOGETHER (vstack x and c, solve jointly)
        """

        assert isinstance(y_list, list) and len(y_list) > 0, "y_list must be non-empty list"
        assert D is not None and D.ndim == 2, "D must be 2D array"
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list"
        assert isinstance(coefficients, list) and len(coefficients) == len(y_list), f"coefficients must be list with len {len(y_list)}, got {type(coefficients).__name__} with len {len(coefficients) if hasattr(coefficients, '__len__') else 'N/A'}"
        
        lambda_dyn = self.config.lambda_dyn
        obs_dim, latent_dim = D.shape
        num_subdyns = len(F)
        
        x_list = []
        for trial_idx, y in enumerate(y_list):
            T = y.shape[1]
            c_trial = coefficients[trial_idx]
            
            # Build observation block: D @ x_t = y_t for all t
            # A_obs is block diagonal with D repeated T times
            # Shape: (obs_dim * T, latent_dim * T)
            A_obs = np.zeros((obs_dim * T, latent_dim * T))
            b_obs = np.zeros(obs_dim * T)
            
            for t in range(T):
                row_start = t * obs_dim
                col_start = t * latent_dim
                A_obs[row_start:row_start + obs_dim, col_start:col_start + latent_dim] = D
                b_obs[row_start:row_start + obs_dim] = y[:, t]
            
            # Build dynamics block: x_{t+1} - F_t @ x_t = 0
            # Shape: (latent_dim * (T-1), latent_dim * T)
            A_dyn = np.zeros((latent_dim * (T - 1), latent_dim * T))
            b_dyn = np.zeros(latent_dim * (T - 1))
            
            for t in range(T - 1):
                # Combined F at time t
                F_t = sum(c_trial[m, t] * F[m] for m in range(num_subdyns))
                
                row_start = t * latent_dim
                col_t = t * latent_dim
                col_t1 = (t + 1) * latent_dim
                
                # -F_t @ x_t + I @ x_{t+1} = 0
                A_dyn[row_start:row_start + latent_dim, col_t:col_t + latent_dim] = -F_t
                A_dyn[row_start:row_start + latent_dim, col_t1:col_t1 + latent_dim] = np.eye(latent_dim)
            
            # Stack with weighting
            sqrt_lambda = np.sqrt(lambda_dyn)
            A_full = np.vstack([A_obs, sqrt_lambda * A_dyn])
            b_full = np.concatenate([b_obs, sqrt_lambda * b_dyn])
            
            # Solve least squares
            #x_flat, _, _, _ = linalg.lstsq(A_full, b_full, rcond=None)
            x_flat = solve_lasso_style(A_full, b_full, l1=0, l2=self.config.l2_reg, solver='inv')
            
            # Reshape to (latent_dim, T)
            x = x_flat.reshape((T, latent_dim)).T
            
            assert x.shape == (latent_dim, T), "x shape must be (%d, %d), got %s" % (latent_dim, T, x.shape)
            assert not np.isnan(x).any(), "x contains NaN for trial %d" % trial_idx
            assert not np.isinf(x).any(), "x contains Inf for trial %d" % trial_idx
            
            x_list.append(x)
        
        return x_list

    
    def _update_c(self, x_list, F, coefficients_prev=None, solver_params = {}):
        """Update coefficients c given latent states x and F."""
        assert isinstance(x_list, list) and len(x_list) > 0, "x_list must be non-empty list, got %s with len %d" % (type(x_list).__name__, len(x_list) if isinstance(x_list, list) else 0)
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list, got %s with len %d" % (type(F).__name__, len(F) if isinstance(F, list) else 0)
        num_subdyns = len(F)
        coefficients = []
        
        for trial_idx, x in enumerate(x_list):
            assert x.ndim == 2, "x must be 2D for trial %d, got shape %s" % (trial_idx, x.shape)
            n_dims, T = x.shape
            assert T > 1, "Trial %d must have T > 1, got %d" % (trial_idx, T)
            assert n_dims == F[0].shape[0], "x dims (%d) must match F dims (%d) for trial %d" % (n_dims, F[0].shape[0], trial_idx)
            c_trial = np.zeros((num_subdyns, T - 1))
            
            for t in range(T - 1):
                x_t = x[:, t]
                x_next = x[:, t + 1]
                assert x_t.shape == (n_dims,), "x_t shape must be (%d,), got %s at t=%d" % (n_dims, x_t.shape, t)
                assert x_next.shape == (n_dims,), "x_next shape must be (%d,), got %s at t=%d" % (n_dims, x_next.shape, t)
                f_x_stacked = np.column_stack([f @ x_t for f in F])
                assert f_x_stacked.shape == (n_dims, num_subdyns), "f_x_stacked shape must be (%d, %d), got %s at t=%d" % (n_dims, num_subdyns, f_x_stacked.shape, t)
                
                c_prev = c_trial[:, t - 1] if (self.config.smooth_term > 0 and t > 0) else None
                
                c_t = solve_coefficients_single_time(
                    f_x_stacked=f_x_stacked,
                    x_next=x_next,
                    l1=self.config.reg_term,
                    solver=self.config.solver,
                    solver_params= {**{'num_iters': self.config.solver_max_iters}, **solver_params},
                    random_state=self.config.seed + t,
                    smooth_term=self.config.smooth_term,
                    c_prev=c_prev
                )
                assert c_t.shape == (num_subdyns,), "c_t shape must be (%d,), got %s at t=%d" % (num_subdyns, c_t.shape, t)
                c_trial[:, t] = c_t
            
            assert c_trial.shape == (num_subdyns, T - 1), "c_trial shape must be (%d, %d), got %s for trial %d" % (num_subdyns, T - 1, c_trial.shape, trial_idx)
            assert not np.isnan(c_trial).any(), "c_trial contains NaN for trial %d" % trial_idx
            coefficients.append(c_trial)
        
        assert len(coefficients) == len(x_list), "coefficients length (%d) must match x_list length (%d)" % (len(coefficients), len(x_list))
        return coefficients
    
    def _update_d(self, y_list, x_list, normalize_cols=False):
        """
        Update observation matrix D given y and x.
        
        Solves: min_D ||Y - D @ X||^2
        Solution: D = Y @ X^T @ (X @ X^T)^{-1}
        """
        assert isinstance(y_list, list) and len(y_list) > 0, "y_list must be non-empty list, got %s with len %d" % (type(y_list).__name__, len(y_list) if isinstance(y_list, list) else 0)
        assert isinstance(x_list, list) and len(x_list) == len(y_list), "x_list length (%d) must match y_list length (%d)" % (len(x_list), len(y_list))
        # Stack all trials
        Y = np.hstack(y_list)
        X = np.hstack(x_list)
        assert Y.ndim == 2 and X.ndim == 2, "Y and X must be 2D, got Y.shape=%s, X.shape=%s" % (Y.shape, X.shape)
        assert Y.shape[1] == X.shape[1], "Y and X must have same time dimension, got Y.shape[1]=%d, X.shape[1]=%d" % (Y.shape[1], X.shape[1])
        
        # Least squares: D = Y @ X.T @ inv(X @ X.T)
        XXT = X @ X.T
        assert XXT.shape == (X.shape[0], X.shape[0]), "XXT shape must be (%d, %d), got %s" % (X.shape[0], X.shape[0], XXT.shape)
        XXT_inv = linalg.pinv(XXT)
        D = Y @ X.T @ XXT_inv
        assert D.shape == (Y.shape[0], X.shape[0]), "D shape must be (%d, %d), got %s" % (Y.shape[0], X.shape[0], D.shape)
        assert not np.isnan(D).any(), "D contains NaN values"
        
        if normalize_cols:
            col_norms = np.linalg.norm(D, axis=0, keepdims=True)
            col_norms[col_norms < 1e-10] = 1.0
            D = D / col_norms
            assert not np.isnan(D).any(), "D contains NaN after normalization"
        
        return D
    
    def _update_f(self, x_list, F, coefficients, step_f):
        """Update dynamics operators F via gradient descent."""
        assert isinstance(x_list, list) and len(x_list) > 0, "x_list must be non-empty list"
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list"
        assert isinstance(coefficients, list) and len(coefficients) == len(x_list), "coefficients length (%d) must match x_list length (%d)" % (len(coefficients), len(x_list))
        assert step_f > 0, "step_f must be positive, got %.6f" % step_f
        gradients = self._compute_gradient(x_list, F, coefficients)
        assert len(gradients) == len(F), "gradients length (%d) must match F length (%d)" % (len(gradients), len(F))
        
        F_new = []
        for m, (f, grad) in enumerate(zip(F, gradients)):
            assert f.shape == grad.shape, "f and grad shapes must match for m=%d, got f.shape=%s, grad.shape=%s" % (m, f.shape, grad.shape)
            f_updated = f + step_f * grad
            nan_mask = np.isnan(f_updated)
            if nan_mask.any():
                np.random.seed(self.config.seed + m + 999)
                f_updated[nan_mask] = np.random.randn(np.sum(nan_mask))
            assert not np.isnan(f_updated).any(), "f_updated contains NaN after fixing for m=%d" % m
            F_new.append(f_updated)
        
        if self.config.normalize_eig:
            F_new = [norm_mat(f, type_norm='evals', to_norm=True) for f in F_new]
            assert all(not np.isnan(f).any() for f in F_new), "F_new contains NaN after normalization"
        
        assert len(F_new) == len(F), "F_new length (%d) must match F length (%d)" % (len(F_new), len(F))
        return F_new
    
    def _compute_gradient(self, x_list, F, coefficients):
        """Compute gradient of dynamics loss w.r.t. F."""
        assert isinstance(x_list, list) and len(x_list) > 0, "x_list must be non-empty list"
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list"
        assert isinstance(coefficients, list) and len(coefficients) == len(x_list), "coefficients length must match x_list length"
        num_subdyns = len(F)
        latent_dim = F[0].shape[0]
        assert all(f.shape == (latent_dim, latent_dim) for f in F), "All F matrices must be square with dim=%d" % latent_dim
        gradients = [np.zeros((latent_dim, latent_dim)) for _ in range(num_subdyns)]
        grad_counts = [0 for _ in range(num_subdyns)]
        
        for trial_idx, x in enumerate(x_list):
            c_trial = coefficients[trial_idx]
            n_dims, T = x.shape
            assert n_dims == latent_dim, "x dims (%d) must match latent_dim (%d) for trial %d" % (n_dims, latent_dim, trial_idx)
            assert c_trial.shape == (num_subdyns, T - 1), "c_trial shape must be (%d, %d) for trial %d" % (num_subdyns, T - 1, trial_idx)
            
            for t in range(T - 1):
                x_t = x[:, t]
                x_next = x[:, t + 1]
                
                x_pred = np.zeros(n_dims)
                for m, f in enumerate(F):
                    x_pred += c_trial[m, t] * (f @ x_t)
                
                residual = x_next - x_pred
                
                for m in range(num_subdyns):
                    grad_m = 2.0 * c_trial[m, t] * np.outer(residual, x_t)
                    assert grad_m.shape == (latent_dim, latent_dim), "grad_m shape must be (%d, %d), got %s" % (latent_dim, latent_dim, grad_m.shape)
                    gradients[m] += grad_m
                    grad_counts[m] += 1
        
        for m in range(num_subdyns):
            if grad_counts[m] > 0:
                gradients[m] /= grad_counts[m]
            assert not np.isnan(gradients[m]).any(), "gradients[%d] contains NaN" % m
        
        assert len(gradients) == num_subdyns, "gradients length must be %d, got %d" % (num_subdyns, len(gradients))
        return gradients
    
    # =========================================================================
    # Error computation
    # =========================================================================
    
    def _compute_error(self, y_list, x_list, D, F, coefficients, return_per = False):
        """Compute total error: observation + dynamics."""
        assert isinstance(y_list, list) and isinstance(x_list, list), "y_list and x_list must be lists"
        assert len(y_list) == len(x_list), "y_list and x_list must have same length, got %d vs %d" % (len(y_list), len(x_list))
        assert D is not None, "D must not be None"
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list"
        
        obs_error = 0.0
        dyn_error = 0.0
        obs_count = 0
        dyn_count = 0
        
        for trial_idx, (y, x) in enumerate(zip(y_list, x_list)):
            c_trial = coefficients[trial_idx]
            T = y.shape[1]
            assert x.shape[1] == T, "x and y must have same T for trial %d, got x.T=%d, y.T=%d" % (trial_idx, x.shape[1], T)
            
            for t in range(T):
                # Observation error
                y_pred = D @ x[:, t]
                obs_error += np.sum((y[:, t] - y_pred) ** 2)
                obs_count += 1
                
                # Dynamics error
                if t > 0:
                    x_pred = np.dstack([c_trial[m, t - 1] * (f @ x[:, t - 1]) for m, f in enumerate(F)]).sum(axis=2)
                    dyn_error += np.sum((x[:, t] - x_pred) ** 2)
                    dyn_count += 1
        
        assert obs_count > 0 and dyn_count > 0, "counts must be positive, got obs_count=%d, dyn_count=%d" % (obs_count, dyn_count)
        
        obs_error_normalized = obs_error / obs_count
        dyn_error_normalized = dyn_error / dyn_count
        if not return_per:
            return (1 - self.config.w_dyn) * obs_error_normalized + self.config.w_dyn * dyn_error_normalized
        else:
            return (1 - self.config.w_dyn) * obs_error_normalized + self.config.w_dyn * dyn_error_normalized,obs_error_normalized,dyn_error_normalized
    
    def _check_stuck(self, error_history):
        """Check if training is stuck."""
        assert isinstance(error_history, list), "error_history must be list, got %s" % type(error_history).__name__
        thresh = self.config.num_no_change_thresh
        if len(error_history) < thresh + 1:
            return False
        recent = error_history[-thresh:]
        error_change = np.abs(recent[-1] - recent[0])
        relative_change = error_change / (np.abs(recent[0]) + 1e-10)
        if relative_change < 1e-6:
            return True
        if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
            return True
        return False
    
    def _perturb_f(self, F):
        """Add noise to F when stuck."""
        assert isinstance(F, list) and len(F) > 0, "F must be non-empty list"
        sigma = self.config.sigma_perturbation
        assert sigma > 0, "sigma_perturbation must be positive, got %.6f" % sigma
        F_perturbed = []
        for m, f in enumerate(F):
            np.random.seed(self.config.seed + m + 5000)
            noise = np.random.randn(*f.shape) * sigma
            assert noise.shape == f.shape, "noise shape must match f shape for m=%d" % m
            F_perturbed.append(f + noise)
        assert len(F_perturbed) == len(F), "F_perturbed length must match F length"
        return F_perturbed
    
    # =========================================================================
    # Fit
    # =========================================================================
    
    def fit(self, y, latent_dim=None, init_D='random', normalize_D_cols=False):
        """
        Fit dLDS model to observations.
        
        Parameters
        ----------
        y : np.ndarray or List[np.ndarray]
            Observations, shape (obs_dim, T) or list of arrays.
        latent_dim : int, optional
            Latent dimension. If None, uses config.latent_dim.
        init_D : str or np.ndarray
            'random', 'pca', or custom D matrix.
        normalize_D_cols : bool
            If True, normalize D columns after each update.
        
        Returns
        -------
        result : dLDS_latent_result
        """
        # === Validate input ===
        self.obs_error_history_ = []
        self.dyn_error_history_ = []


        assert y is not None, "y must not be None"
        y_list, single_trial = validate_data(y)
        assert len(y_list) > 0, "y_list must be non-empty"
        n_trials = len(y_list)
        trial_lengths = [d.shape[1] for d in y_list]
        assert all(T > 1 for T in trial_lengths), "All trials must have T > 1, got %s" % trial_lengths
        obs_dim = y_list[0].shape[0]
        assert obs_dim > 0, "obs_dim must be positive, got %d" % obs_dim
        
        if latent_dim is None:
            latent_dim = self.config.latent_dim if self.config.latent_dim else obs_dim
        
        assert latent_dim > 0 and latent_dim <= obs_dim, "latent_dim must be in (0, %d], got %d" % (obs_dim, latent_dim)
        assert all(d.shape[0] == obs_dim for d in y_list), "All trials must have same obs_dim (%d)" % obs_dim
        assert all(d.ndim == 2 for d in y_list), "All trials must be 2D"
        
        if self.config.verbose >= 1:
            print("=" * 60)
            print("dLDS_with_latents Fitting")
            print("=" * 60)
            print("Number of trials: %d" % n_trials)
            print("Trial lengths: %s" % trial_lengths)
            print("Observation dimension: %d" % obs_dim)
            print("Latent dimension: %d" % latent_dim)
            print("Number of dynamics operators (M): %d" % self.config.num_subdyns)
            print("-" * 60)
        
        # === Initialize ===
        if init_D == 'pca':
            D = self._init_d_pca(y_list)
        else:
            D = self._init_d(obs_dim, latent_dim, init_D)
        
        assert D is not None and D.shape == (obs_dim, latent_dim), "D must be (%d, %d), got %s" % (obs_dim, latent_dim, D.shape if D is not None else None)
        x_list = self._init_x(y_list, D)
        assert len(x_list) == n_trials, "x_list length must be %d, got %d" % (n_trials, len(x_list))
        F = self._init_f(latent_dim)
        assert len(F) == self.config.num_subdyns, "F length must be %d, got %d" % (self.config.num_subdyns, len(F))
        coefficients = self._init_c(trial_lengths)
        assert len(coefficients) == n_trials, "coefficients length must be %d, got %d" % (n_trials, len(coefficients))
        
        step_f = self.config.step_f
        assert step_f > 0, "step_f must be positive, got %.6f" % step_f
        error_history = []
        
        # === Progress bar ===
        if HAS_TQDM and self.config.verbose >= 1:
            pbar = tqdm(range(self.config.max_iter), desc="Fitting", ncols=80)
        else:
            pbar = range(self.config.max_iter)
        
        # === Main loop ===
        converged = False
        for iteration in pbar:
# 1. Update x
            x_list = self._update_x(y_list, D, F, coefficients, x_list)
            assert len(x_list) == n_trials, "x_list length must remain %d after update" % n_trials
            
            # 2. Update c
            solver_params = self.config.solver_params
            coefficients = self._update_c(x_list, F, coefficients, solver_params = solver_params)
            assert len(coefficients) == n_trials, "coefficients length must remain %d after update" % n_trials
            
            # 3. Update D
            D = self._update_d(y_list, x_list, normalize_cols=normalize_D_cols)
            assert D.shape == (obs_dim, latent_dim), "D shape must remain (%d, %d)" % (obs_dim, latent_dim)
            
            # Compute error BEFORE F update (with current x, c, D)
            #error_old = self._compute_error(y_list, x_list, D, F, coefficients)
            error, obs_err, dyn_err = self._compute_error(y_list, x_list, D, F, coefficients, return_per=True)
            error_history.append(error)
            self.obs_error_history_.append(obs_err)
            self.dyn_error_history_.append(dyn_err)

            
            # 4. Update F with backtracking
            F_old = [f.copy() for f in F]
            
            #error_old = error_history[-1] if error_history else np.inf
            
            for bt in range(self.config.max_backtrack):
                F_new = self._update_f(x_list, F_old, coefficients, step_f)
                #error_new = self._compute_error(y_list, x_list, D, F_new, coefficients)
                error_new = self._compute_error(y_list, x_list, D, F_new, coefficients, return_per=False)
                
                #if error_new <= error_old * 1.1:
                if error_new <= error * 1.1:
                    F = F_new
                    break
                else:
                    step_f = step_f * self.config.backtrack_factor
                    if step_f < self.config.min_step_f:
                        break
            
            #if error_new > error_old * 1.1:
            if error_new > error * 1.1:
                F = F_old
                
                        

            # Compute error
            error = self._compute_error(y_list, x_list, D, F, coefficients)
            assert not np.isnan(error) and not np.isinf(error), "error is invalid: %.6f at iteration %d" % (error, iteration)
            error_history.append(error)
            
            
            # Check for error increase
            if len(error_history) >= 10:
                min_recent = min(error_history[-10:])
                if error > min_recent * 1.5:
                    if not hasattr(self, '_perturb_count'):
                        self._perturb_count = 0
                    self._perturb_count += 1
                    if self._perturb_count > 5:
                        raise ValueError("Error keeps increasing after 5 perturbations. Stopping.")
                    if self.config.verbose >= 1:
                        print("Error increased: %.2f vs min %.2f, perturbing F (%d/5)" % (error, min_recent, self._perturb_count))
                    F = self._perturb_f(F)
                    step_f = self.config.step_f
                    
        
            
            if HAS_TQDM and self.config.verbose >= 1:
                pbar.set_postfix({'error': '%.6f' % error})
            
            # Check convergence
            if error < self.config.max_error:
                converged = True
                if self.config.verbose >= 1:
                    print("\nConverged at iteration %d" % iteration)
                break
            
            # Check stuck
            if self._check_stuck(error_history):
                if self.config.verbose >= 2:
                    print("\nStuck at iteration %d, perturbing F" % iteration)
                F = self._perturb_f(F)
            
            # Decay step_f
            step_f = max(step_f * self.config.gd_decay, self.config.min_step_f)
        
        # === Final updates ===
        coefficients = self._update_c(x_list, F, coefficients)
        x_list = self._update_x(y_list, D, F, coefficients, x_list)
        
        final_error = error_history[-1] if error_history else np.inf
        assert len(error_history) > 0, "error_history must be non-empty after fitting"
        
        if self.config.verbose >= 1:
            print("-" * 60)
            print("Final error: %.6f" % final_error)
            print("=" * 60)
        
        # === Store ===
        self.F_ = F
        self.D_ = D
        self.x_ = x_list if not single_trial else x_list[0]
        self.dynamic_coefficients_ = coefficients
        self._is_fitted = True
        
        # Build result
        result = dLDS_latent_result(
            F=F,
            D=D,
            x=x_list,
            dynamic_coefficients=coefficients,
            error_history=error_history,
            n_iterations=len(error_history),
            converged=converged,
            final_error=final_error,
            n_trials=n_trials,
            trial_lengths=trial_lengths,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            config=self.config
        )
        
        self.result_ = result
        return result
    
    def reconstruct(self, y=None, return_error=False):
        """Reconstruct observations using fitted model."""
        assert self._is_fitted, "Model must be fitted first, call .fit() before .reconstruct()"
        
        if y is None:
            raise ValueError("y is required")
        
        y_list, single_trial = validate_data(y)
        assert len(y_list) > 0, "y_list must be non-empty"
        
        # Get latents
        x_list = self._update_x(y_list, self.D_, self.F_, self.dynamic_coefficients_, None)
        assert len(x_list) == len(y_list), "x_list length must match y_list length"
        
        # Reconstruct
        y_hat_list = [self.D_ @ x for x in x_list]
        assert len(y_hat_list) == len(y_list), "y_hat_list length must match y_list length"
        
        if single_trial:
            if return_error:
                err = np.sum((y_list[0] - y_hat_list[0]) ** 2, axis=0)
                return y_hat_list[0], err
            return y_hat_list[0]
        else:
            if return_error:
                errors = [np.sum((y - y_hat) ** 2, axis=0) for y, y_hat in zip(y_list, y_hat_list)]
                return y_hat_list, errors
            return y_hat_list
    
    def get_latents(self, y):
        """Extract latent states from observations."""
        assert self._is_fitted, "Model must be fitted first, call .fit() before .get_latents()"
        
        y_list, single_trial = validate_data(y)
        assert len(y_list) > 0, "y_list must be non-empty"
        x_list = self._update_x(y_list, self.D_, self.F_, self.dynamic_coefficients_, None)
        assert len(x_list) == len(y_list), "x_list length must match y_list length"
        
        return x_list[0] if single_trial else x_list