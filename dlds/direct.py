# -*- coding: utf-8 -*-
"""
Direct dLDS Model
=================

dLDS model for the direct observation case (D = I).

In this case, we observe the latent states directly:
    x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1}

Author: Noga Mudrik 

Direct_dLDS Architecture
========================

CLASS STRUCTURE:
┌─────────────────────────────────────────────────────────────────┐
│  Direct_dLDS                                                    │
├─────────────────────────────────────────────────────────────────┤
│  Config: dLDS_config (num_subdyns, step_f, reg_term, ...)       │
│  Output: dLDS_result (F, dynamic_coefficients, error_history)   │
├─────────────────────────────────────────────────────────────────┤
│  METHODS:                                                       │
│    fit(data)           → Main training loop                     │
│    reconstruct(data)   → Reconstruct trajectory                 │
│    predict(x0, n_steps)→ Predict future                         │
└─────────────────────────────────────────────────────────────────┘

FIT() FLOW:
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: data (p x T) or list of (p x T_i)                        │
└──────────────────┬───────────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  INIT:  F = _init_f()        → M random matrices (p x p)         │
│         c = _init_c()        → M x (T-1) per trial               │
└──────────────────┬───────────────────────────────────────────────┘
                   ▼
         ┌─────────────────┐
         │  TRAINING LOOP  │
         └────────┬────────┘
                  ▼
    ┌─────────────────────────────┐
    │  1. _update_c(data, F)      │  Solve: x_{t+1} = (Σ c_m f_m) x_t
    │     └─ solve_lasso_style()  │  for c (sparse, per time point)
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  2. _update_f(data, F, c)   │  Gradient descent on F:
    │     └─ _compute_gradient()  │  f_m += η * Σ_t c_mt (x_{t+1} - x̂_{t+1}) x_t^T
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  3. _normalize_f(F)         │  Normalize by spectral radius
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  4. _compute_error()        │  MSE: ||x_{t+1} - x̂_{t+1}||²
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  5. _check_stuck()          │  If error flat → _perturb_f()
    └──────────────┬──────────────┘
                   ▼
    ┌─────────────────────────────┐
    │  6. Decay step_f            │  step_f *= gd_decay
    └──────────────┬──────────────┘
                   ▼
         ┌────────────────┐
         │  Converged?    │──No──→ Loop back to 1
         └───────┬────────┘
                 │ Yes
                 ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT: dLDS_result(F, dynamic_coefficients, error_history)     │
└──────────────────────────────────────────────────────────────────┘

MODEL EQUATION:
    x_{t+1} = F_t @ x_t = (Σ_m c_{m,t} * f_m) @ x_t

    f_m  : learned dynamics operators (M matrices, each p x p)
    c_t  : sparse, time-varying coefficients (M x 1 per time)
    F_t  : combined dynamics at time t (p x p)
"""

import numpy as np
from typing import List, Union, Optional
import warnings

#from .base import Base_dLDS
# from config import dLDS_config
# from results import dLDS_result
# from solvers import solve_lasso_style, solve_coefficients_single_time
# from dlds_utils import (
#     init_mat, norm_mat, validate_data, 
#     init_distant_f, center_data, check_f_correlation
# )

from .config import dLDS_config
from .results import dLDS_result
from .dlds_utils import validate_data, center_data
from .viz import *
# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
# -*- coding: utf-8 -*-


import numpy as np
from scipy import linalg
import warnings
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

#from .config import dLDS_config
#from .results import dLDS_result
from .solvers import solve_lasso_style, solve_coefficients_single_time
from .dlds_utils import (
    init_mat, norm_mat, validate_data, 
    init_distant_f, center_data, check_f_correlation
)


class Base_dLDS(ABC):
    """
    Base class for dLDS models.
    
    Contains shared logic for:
    - Data validation
    - Initialization of F and c
    - Coefficient update (solving for c given F and data)
    - Dynamics operator update (gradient descent on F)
    - Reconstruction
    - Perturbation when stuck
    
    Subclasses implement fit() and reconstruct() for specific cases:
    - Direct_dLDS: D = I (observe latent states directly)
    - Latent_dLDS: D ≠ I (observe through observation matrix)
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize dLDS model.
        
        Parameters
        ----------
        config : dLDS_config, optional
            Configuration object. If None, uses default config.
        **kwargs
            If config is None, these are used to create config.
            If config is provided, kwargs override config values.
            
        Examples
        --------
        >>> model = Direct_dLDS(dLDS_config(num_subdyns=3))
        >>> model = Direct_dLDS(num_subdyns=3, step_f=10)
        >>> model = Direct_dLDS(config, step_f=20)  # override step_f
        """
        # === Create or update config ===
        if config is None:
            self.config = dLDS_config(**kwargs)
        elif isinstance(config, dLDS_config):
            if kwargs:
                # Override config with kwargs
                config_dict = config.to_dict()
                config_dict.update(kwargs)
                self.config = dLDS_config(**config_dict)
            else:
                self.config = config
        elif isinstance(config, dict):
            config.update(kwargs)
            self.config = dLDS_config(**config)
        else:
            raise TypeError(
                "config must be dLDS_config, dict, or None. Got %s" % type(config).__name__
            )
        
       
        # === Initialize state ===
        self.F_ = None  # Learned operators
        self.dynamic_coefficients_ = None  # Learned coefficients
        self.result_ = None  # Full result object
        self._is_fitted = False
        
        
    # =========================================================================
    # Abstract methods (implemented in subclasses)
    # =========================================================================
    
    @abstractmethod
    def fit(self, data):
        """
        Fit the model to data.
        
        Parameters
        ----------
        data : np.ndarray or List[np.ndarray]
            Training data.
            
        Returns
        -------
        dLDS_result
            Fitting results.
        """
        pass
    
    @abstractmethod
    def reconstruct(self, data, return_error=False):
        """
        Reconstruct data using the fitted model.
        
        Parameters
        ----------
        data : np.ndarray or List[np.ndarray]
            Data to reconstruct.
        return_error : bool
            If True, also return reconstruction error.
            
        Returns
        -------
        reconstruction : np.ndarray or List[np.ndarray]
            Reconstructed trajectories.
        error : np.ndarray or List[np.ndarray], optional
            Reconstruction error (if return_error=True).
        """
        pass
    
    # =========================================================================
    # Data validation
    # =========================================================================
    
    def _validate_data(self, data):
        """
        Validate input data and convert to list format.
        
        Returns
        -------
        data_list : List[np.ndarray]
            List of 2D arrays.
        single_trial : bool
            True if input was single array.
        """
        return validate_data(data)
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_f(self, latent_dim):
        """
        Initialize dynamics operators.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of latent space.
            
        Returns
        -------
        F : List[np.ndarray]
            List of M operators, each (latent_dim, latent_dim).
        """
        num_subdyns = self.config.num_subdyns
        
        if self.config.init_distant_f and num_subdyns > 1:
            F, achieved_corr = init_distant_f(
                latent_dim=latent_dim,
                num_subdyns=num_subdyns,
                max_corr=self.config.max_corr,
                max_retries=self.config.max_init_retries,
                normalize=self.config.normalize_eig,
                seed=self.config.seed
            )
            if self.config.verbose >= 2:
                print("Initialized F with max correlation: %.4f" % achieved_corr)
        else:
            F = [
                init_mat(
                    (latent_dim, latent_dim),
                    r_seed=self.config.seed + i,
                    normalize=self.config.normalize_eig
                )
                for i in range(num_subdyns)
            ]
        
        # Validate F
        for i, f in enumerate(F):
            assert f.shape == (latent_dim, latent_dim), (
                "F[%d] has shape %s but expected (%d, %d)" % (
                    i, f.shape, latent_dim, latent_dim
                )
            )
        
        return F
    
    def _init_c(self, trial_lengths):
        """
        Initialize coefficients for each trial.
        
        Parameters
        ----------
        trial_lengths : List[int]
            Number of time points in each trial.
            
        Returns
        -------
        coefficients : List[np.ndarray]
            List of coefficient matrices, each (M, T-1).
        """
        num_subdyns = self.config.num_subdyns
        
        coefficients = []
        for i, T in enumerate(trial_lengths):
            c = init_mat(
                (num_subdyns, T - 1),
                r_seed=self.config.seed + 1000 + i
            )
            coefficients.append(c)
        
        # Validate
        for i, (c, T) in enumerate(zip(coefficients, trial_lengths)):
            assert c.shape == (num_subdyns, T - 1), (
                "coefficients[%d] has shape %s but expected (%d, %d)" % (
                    i, c.shape, num_subdyns, T - 1
                )
            )
        
        return coefficients
    
    # =========================================================================
    # Coefficient update
    # =========================================================================
    
    def _update_c(self, data_list, F, coefficients_prev=None):
        """
        Update coefficients for all trials.
        
        Solves: x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1} for c
        
        Parameters
        ----------
        data_list : List[np.ndarray]
            List of trials, each (n_dims, T).
        F : List[np.ndarray]
            List of M operators, each (n_dims, n_dims).
        coefficients_prev : List[np.ndarray], optional
            Previous coefficients (for smoothness term).
            
        Returns
        -------
        coefficients : List[np.ndarray]
            Updated coefficients, list of (M, T-1) arrays.
        """
        solver_params = {
            'l1': self.config.reg_term,
            'l2': self.config.l2_reg,
            'solver': self.config.solver,
            'num_iters': self.config.solver_max_iters,
            'smooth_term': self.config.smooth_term
        }
        
        num_subdyns = len(F)
        coefficients = []
        
        for trial_idx, trial_data in enumerate(data_list):
            n_dims, T = trial_data.shape
            c_trial = np.zeros((num_subdyns, T - 1))
            
            # Get previous coefficients for smoothness
            c_prev_trial = None
            if coefficients_prev is not None and len(coefficients_prev) > trial_idx:
                c_prev_trial = coefficients_prev[trial_idx]
            
            for t in range(T - 1):
                # Build stacked f_m @ x_t matrix
                x_t = trial_data[:, t]
                x_next = trial_data[:, t + 1]
                
                # f_x_stacked: (n_dims, M)
                f_x_stacked = np.column_stack([f @ x_t for f in F])
                
                # Get previous c for smoothness
                c_prev = None
                if self.config.smooth_term > 0 and t > 0:
                    c_prev = c_trial[:, t - 1]
                elif self.config.smooth_term > 0 and c_prev_trial is not None and t == 0:
                    # First time point: use last c from previous trial? 
                    # Or just skip smoothness for t=0
                    c_prev = None
                
                # Solve for c
                c_t = solve_coefficients_single_time(
                    f_x_stacked=f_x_stacked,
                    x_next=x_next,
                    l1=self.config.reg_term,
                    solver=self.config.solver,
                    solver_params=solver_params,
                    random_state=self.config.seed + t,
                    smooth_term=self.config.smooth_term,
                    c_prev=c_prev
                )
                
                c_trial[:, t] = c_t
            
            coefficients.append(c_trial)
        
        # Validate
        for i, c in enumerate(coefficients):
            assert c.shape[0] == num_subdyns, (
                "coefficients[%d] has %d rows but there are %d operators" % (
                    i, c.shape[0], num_subdyns
                )
            )
        
        return coefficients
    
    # =========================================================================
    # Dynamics operator update
    # =========================================================================
    
    def _compute_gradient(self, data_list, F, coefficients):
        """
        Compute gradient of loss w.r.t. each dynamics operator.
        
        Loss: L = Σ_trials Σ_t ||x_{t+1} - x̂_{t+1}||^2
        where x̂_{t+1} = (Σ_m c_{m,t} * f_m) @ x_t
        
        Gradient w.r.t. f_m:
            ∂L/∂f_m = -2 * Σ_t c_{m,t} * (x_{t+1} - x̂_{t+1}) @ x_t^T
        
        Parameters
        ----------
        data_list : List[np.ndarray]
            List of trials.
        F : List[np.ndarray]
            Current operators.
        coefficients : List[np.ndarray]
            Current coefficients.
            
        Returns
        -------
        gradients : List[np.ndarray]
            Gradient for each operator, same shape as F.
        """
        num_subdyns = len(F)
        latent_dim = F[0].shape[0]
        
        # Initialize gradients
        gradients = [np.zeros((latent_dim, latent_dim)) for _ in range(num_subdyns)]
        grad_counts = [0 for _ in range(num_subdyns)]
        
        for trial_idx, trial_data in enumerate(data_list):
            c_trial = coefficients[trial_idx]
            n_dims, T = trial_data.shape
            
            for t in range(T - 1):
                x_t = trial_data[:, t]
                x_next = trial_data[:, t + 1]
                
                # Compute prediction
                x_pred = np.zeros(n_dims)
                for m, f in enumerate(F):
                    x_pred += c_trial[m, t] * (f @ x_t)
                
                # Error (residual)
                residual = x_next - x_pred  # Shape: (n_dims,)
                
                # Gradient for each f_m:
                # ∂L/∂f_m = -2 * c_{m,t} * residual @ x_t^T
                # But we want gradient DESCENT, so we use the negative:
                # update = +2 * c_{m,t} * residual @ x_t^T
                
                for m in range(num_subdyns):
                    # Factor of 2 comes from derivative of squared error: d/df ||e||^2 = 2 * e * de/df
                    grad_m = 2.0 * c_trial[m, t] * np.outer(residual, x_t)
                    gradients[m] += grad_m
                    grad_counts[m] += 1
        
        # Average gradients (or take median based on config)
        if self.config.action_along_time == 'mean':
            for m in range(num_subdyns):
                if grad_counts[m] > 0:
                    gradients[m] /= grad_counts[m]
        # For 'median', we would need to store all gradients per time
        # For now, just use mean for simplicity
        
        return gradients
    
    def _update_f(self, data_list, F, coefficients, step_f):
        """
        Update dynamics operators via gradient descent.
        
        Update rule (gradient DESCENT to minimize loss):
            f_m := f_m + η * gradient_m
        
        where gradient_m = mean over t of: c_{m,t} * (x_{t+1} - x̂_{t+1}) @ x_t^T
        
        The sign is PLUS because the gradient of the loss w.r.t. f_m is negative
        of this quantity, so descent means adding it.
        
        Parameters
        ----------
        data_list : List[np.ndarray]
            List of trials.
        F : List[np.ndarray]
            Current operators.
        coefficients : List[np.ndarray]
            Current coefficients.
        step_f : float
            Learning rate.
            
        Returns
        -------
        F_new : List[np.ndarray]
            Updated operators.
        """
        gradients = self._compute_gradient(data_list, F, coefficients)
        
        F_new = []
        for m, (f, grad) in enumerate(zip(F, gradients)):
            # Gradient descent: f := f + η * grad
            # (The grad already has the correct sign for descent)
            f_updated = f + step_f * grad
            
            # Handle NaN values
            nan_mask = np.isnan(f_updated)
            if nan_mask.any():
                warnings.warn(
                    "NaN values detected in F[%d] after update. "
                    "Replacing with random values." % m
                )
                np.random.seed(self.config.seed + m + 999)
                f_updated[nan_mask] = np.random.randn(np.sum(nan_mask))
            
            F_new.append(f_updated)
        
        # Normalize if requested
        if self.config.normalize_eig:
            F_new = self._normalize_f(F_new)
        
        return F_new
    
    def _verify_gradient_direction(self, F_old, F_new, data_list, coefficients):
        """
        Assert that the gradient update decreases (or maintains) the loss.
        
        This is a sanity check to catch sign errors in the gradient.
        """
        error_old = self._compute_error(data_list, F_old, coefficients)
        error_new = self._compute_error(data_list, F_new, coefficients)
        
        # Allow small increases due to numerical issues
        max_allowed_increase = 0.1 * error_old + 1e-8
        
        if error_new > error_old + max_allowed_increase:
            warnings.warn(
                "Gradient update INCREASED error from %.6f to %.6f (increase: %.6f). "
                "This may indicate a bug in gradient computation or step_f is too large. "
                "Try reducing step_f." % (error_old, error_new, error_new - error_old)
            )
            return False
        return True
    
    # =========================================================================
    # Normalization and perturbation
    # =========================================================================
    
    def _normalize_f(self, F):
        """Normalize each operator by its spectral radius."""
        return [norm_mat(f, type_norm='evals', to_norm=True) for f in F]
    
    def _perturb_f(self, F):
        """Add small random noise to F when stuck."""
        sigma = self.config.sigma_perturbation
        
    def _perturb_f(self, F, iteration=0):
        sigma = self.config.sigma_perturbation
        F_perturbed = []
        for m, f in enumerate(F):
            np.random.seed(self.config.seed + m + iteration * 100)
            noise = np.random.randn(*f.shape) * sigma
            F_perturbed.append(f + noise)
        return F_perturbed
        
    # =========================================================================
    # Error computation
    # =========================================================================
    
    def _compute_error(self, data_list, F, coefficients):
        """
        Compute mean squared reconstruction error.
        
        Returns
        -------
        error : float
            Mean squared error across all trials and time points.
        """
        total_error = 0.0
        total_count = 0
        
        for trial_idx, trial_data in enumerate(data_list):
            c_trial = coefficients[trial_idx]
            n_dims, T = trial_data.shape
            
            for t in range(T - 1):
                x_t = trial_data[:, t]
                x_next = trial_data[:, t + 1]
                
                # Prediction
                x_pred = np.zeros(n_dims)
                for m, f in enumerate(F):
                    x_pred += c_trial[m, t] * (f @ x_t)
                
                # Squared error
                error_t = np.sum((x_next - x_pred) ** 2)
                total_error += error_t
                total_count += 1
        
        if total_count == 0:
            return np.inf
        
        return total_error / total_count
    
    def _check_stuck(self, error_history):
        """
        Check if training is stuck.
        
        Returns True if:
        - Error hasn't changed significantly for num_no_change_thresh iterations
        - Error is increasing
        
        Parameters
        ----------
        error_history : List[float]
            Error at each iteration.
            
        Returns
        -------
        is_stuck : bool
        """
        #relative_change_param = self.relative_change_param
        
        
        relative_change_thres = self.config.relative_change_param# .get('relative_change_thres', 1e-6) # this is the sensitivty of pertubations
        thresh_num_iter_for_perturbations = self.config.num_no_change_thresh
        
        if len(error_history) < thresh_num_iter_for_perturbations + 1:
            return False
        
        recent = error_history[-thresh_num_iter_for_perturbations:]
        
        # Check if no improvement
        error_change = np.abs(recent[-1] - recent[0])
        relative_change = error_change / (np.abs(recent[0]) + 1e-10)
        
        if relative_change < relative_change_thres:
            return True
        
        # Check if error is increasing
        if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
            return True
        
        return False
    
    # =========================================================================
    # Reconstruction
    # =========================================================================
    
    def _create_reconstruction(self, data, F, coefficients):
        """
        Create one-step-ahead reconstruction for a single trial.
        
        Parameters
        ----------
        data : np.ndarray
            Single trial, shape (n_dims, T).
        F : List[np.ndarray]
            Dynamics operators.
        coefficients : np.ndarray
            Coefficients for this trial, shape (M, T-1).
            
        Returns
        -------
        reconstruction : np.ndarray
            Reconstructed trajectory, shape (n_dims, T).
        """
        n_dims, T = data.shape
        
        # Validate
        assert coefficients.shape == (len(F), T - 1), (
            "coefficients shape %s doesn't match (num_subdyns=%d, T-1=%d)" % ( coefficients.shape, len(F), T - 1  )        )
        
        reconstruction = np.zeros((n_dims, T))
        reconstruction[:, 0] = data[:, 0]  # First point is the same
        
        for t in range(T - 1):
            x_t = data[:, t]
            x_pred = np.zeros(n_dims)
            
            for m, f in enumerate(F):
                x_pred += coefficients[m, t] * (f @ x_t)
            
            reconstruction[:, t + 1] = x_pred
        
        return reconstruction
    
    # =========================================================================
    # Save / Load (methods implemented but calls commented)
    # =========================================================================
    
    def save(self, path):
        """
        Save model to file.
        
        Parameters
        ----------
        path : str
            Path to save file (e.g., 'model.pkl').
        """
        import pickle
        
        assert self._is_fitted, (
            "Model must be fitted before saving. Call fit() first."
        )
        
        save_dict = {
            'config': self.config,
            'F': self.F_,
            'dynamic_coefficients': self.dynamic_coefficients_,
            'result': self.result_
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # print("Model saved to %s" % path)
    
    @classmethod
    def load(cls, path):
        """
        Load model from file.
        
        Parameters
        ----------
        path : str
            Path to saved file.
            
        Returns
        -------
        model : Base_dLDS subclass
            Loaded model.
        """
        import pickle
        
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        model = cls(config=save_dict['config'])
        model.F_ = save_dict['F']
        model.dynamic_coefficients_ = save_dict['dynamic_coefficients']
        model.result_ = save_dict['result']
        model._is_fitted = True
        
        # print("Model loaded from %s" % path)
        return model


class Direct_dLDS(Base_dLDS):
    """
    Direct dLDS model (D = I).
    
    Observes latent states directly:
        x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1}
    
    Parameters
    ----------
    config : dLDS_config, optional
        Configuration object.
    **kwargs
        Configuration parameters (see dLDS_config for options).
        
    Attributes
    ----------
    config : dLDS_config
        Model configuration.
    F_ : List[np.ndarray]
        Learned dynamics operators (after fitting).
    dynamic_coefficients_ : List[np.ndarray]
        Learned coefficients (after fitting).
    result_ : dLDS_result
        Full fitting results (after fitting).
        
    Examples
    --------
    >>> # Simple usage
    >>> model = Direct_dLDS(num_subdyns=3)
    >>> result = model.fit(data)  # data: (n_dims, T)
    >>> 
    >>> # With config
    >>> config = dLDS_config(num_subdyns=3, solver='spgl1', reg_term=0.1)
    >>> model = Direct_dLDS(config)
    >>> result = model.fit(data)
    >>> 
    >>> # Multiple trials
    >>> data = [trial1, trial2, trial3]  # Each (n_dims, T_i)
    >>> result = model.fit(data)
    >>> 
    >>> # Access results
    >>> print(result.F)                     # Learned operators
    >>> print(result.dynamic_coefficients)  # Learned coefficients
    >>> 
    >>> # Reconstruct
    >>> x_hat = model.reconstruct(data)
    """
    
    def fit(self, data):
        """
        Fit the dLDS model to data.
        
        Parameters
        ----------
        data : np.ndarray or List[np.ndarray]
            If np.ndarray: single trial, shape (n_dims, T)
            If list: multiple trials, each (n_dims, T_i)
            Variable trial lengths are allowed.
            
        Returns
        -------
        result : dLDS_result
            Fitting results containing:
            - F: learned dynamics operators
            - dynamic_coefficients: learned coefficients per trial
            - error_history: reconstruction error per iteration
            - converged: whether training converged
            - final_error: final reconstruction error
            
        Notes
        -----
        The model learns shared dynamics operators F across all trials,
        but separate coefficients c for each trial.
        """
        # === Validate input ===
        data_list, single_trial = self._validate_data(data)
        n_trials = len(data_list)
        trial_lengths = [d.shape[1] for d in data_list]
        latent_dim = data_list[0].shape[0]
        
        if self.config.use_relative_error:
            data_var = np.mean([np.var(d) for d in data_list])
            effective_max_error = self.config.max_error_relative * data_var
        else:
            effective_max_error = self.config.max_error
            
            
        # Validate latent_dim consistency
        for i, d in enumerate(data_list):
            assert d.shape[0] == latent_dim, (
                "All trials must have same n_dims. "
                "Trial 0 has %d dims but trial %d has %d dims." % (
                    latent_dim, i, d.shape[0]
                )
            )
        
        if self.config.verbose >= 1:
            print("=" * 60)
            print("Direct_dLDS Fitting")
            print("=" * 60)
            print("Number of trials: %d" % n_trials)
            print("Trial lengths: %s" % trial_lengths)
            print("Latent dimension: %d" % latent_dim)
            print("Number of dynamics operators (M): %d" % self.config.num_subdyns)
            print("Solver: %s" % self.config.solver)
            print("Regularization (τ): %s" % self.config.reg_term)
            print("Max error threshold: %.6f" % effective_max_error)
            print("-" * 60)
            
        
        # === Center data (optional) ===
        data_centers = None
        if self.config.center_data:
            data_list, data_centers = center_data(data_list)
            if self.config.verbose >= 2:
                print("Data centered around zero.")
        
        # === Initialize F ===
        F = self._init_f(latent_dim)
        if self.config.verbose >= 2:
            print("Initialized %d dynamics operators." % len(F))
        
        # === Initialize coefficients ===
        coefficients = self._init_c(trial_lengths)
        if self.config.verbose >= 2:
            print("Initialized coefficients for %d trials." % n_trials)
        
        # === Setup ===
        step_f = self.config.step_f
        reg_term = self.config.reg_term
        error_history = []
        F_evolution = [] if self.config.track_evolution else None
        c_evolution = [] if self.config.track_evolution else None
        
        # === Progress bar ===
        if HAS_TQDM and self.config.verbose >= 1:
            pbar = tqdm(range(self.config.max_iter), desc="Fitting", ncols=80)
        else:
            pbar = range(self.config.max_iter)
            if self.config.verbose >= 1 and not HAS_TQDM:
                print("Note: Install tqdm for progress bar (pip install tqdm)")
        
        # === Main training loop ===
        converged = False
        final_iteration = 0
        
        for iteration in pbar:
            final_iteration = iteration
            
            # --- Update coefficients ---
            coefficients = self._update_c(data_list, F, coefficients)
            
            error = self._compute_error(data_list, F, coefficients)
            error_history.append(error)
            if error < effective_max_error:
                converged = True
                if self.config.verbose >= 1:
                    print("\nConverged at iteration %d with error %.6f" % (iteration, error))
                break
            
            # --- Decay regularization ---
            if self.config.decaying_reg < 1.0:
                # Update internal reg_term for coefficient solver
                # This is a bit hacky; ideally config should be immutable
                pass  # We handle this in _update_c if needed
            
            # --- Update F ---
            # --- Update F with backtracking ---
            F_old = [f.copy() for f in F]
            error_old = error
            
            max_backtrack = self.config.max_backtrack
            backtrack_factor = self.config.backtrack_factor
            for bt in range(max_backtrack):
                F_new = self._update_f(data_list, F_old, coefficients, step_f)
                error_new = self._compute_error(data_list, F_new, coefficients)
                
                if error_new <= error_old:
                    F = F_new
                    break
                else:
                    step_f = step_f * 0.5
                    if self.config.verbose >= 2:
                        print("Backtrack %d: step_f reduced to %.6f" % (bt + 1, step_f))
                        
            if error_new > error_old * 1.1:
                F = self._perturb_f(F_old)
                if self.config.verbose >= 1:
                    print("Warning: F update failed (%.5f vs %.5f), perturbing F" % (error_new, error_old))
        
            # if error_new > error_old*1.1:
            #     F = F_old  # keep old if all backtrack failed
            #     if self.config.verbose >= 1:
            #         raise ValueError("Warning: F update skipped, could not reduce error %.5f vs %.5f"%(error_new , error_old))
                    
                    
                    
            
            # --- Compute error ---
            #error = self._compute_error(data_list, F, coefficients)
            #error_history.append(error)
            
            # --- Update progress bar ---
            if HAS_TQDM and self.config.verbose >= 1:
                pbar.set_postfix({'error': '%.6f' % error, 'step_f': '%.4f' % step_f})
            elif self.config.verbose >= 2 and iteration % 100 == 0:
                print("Iteration %d: error = %.6f, step_f = %.6f" % (iteration, error, step_f))
            
            # --- Track evolution ---
            if self.config.track_evolution:
                F_evolution.append([f.copy() for f in F])
                c_evolution.append([c.copy() for c in coefficients])
            
            # --- Check convergence ---
            # if error < self.config.max_error:
            #     converged = True
            #     if self.config.verbose >= 1:
            #         print("\nConverged at iteration %d with error %.6f" % (iteration, error))
            #     break
            
            # --- Check if stuck ---
            if self._check_stuck(error_history):
                if self.config.verbose >= 2:
                    print("\nStuck at iteration %d, perturbing F" % iteration)
                F = self._perturb_f(F)
            
            # --- Decay learning rate ---
            if step_f > self.config.min_step_f:
                step_f = max(step_f * self.config.gd_decay, self.config.min_step_f)
        
        # === Final messages ===
        final_error = error_history[-1] if error_history else np.inf
        
        if self.config.verbose >= 1:
            print("-" * 60)
            if converged:
                print("Training converged.")
            else:
                print("Training did not converge (reached max_iter=%d)." % self.config.max_iter)
            print("Final error: %.6f" % final_error)
            print("=" * 60)
            print("\nTo reconstruct data, call: x_hat = model.reconstruct(data)")
        
        # === Build result ===
        result = dLDS_result(
            F=F,
            dynamic_coefficients=coefficients,
            error_history=error_history,
            n_iterations=final_iteration + 1,
            converged=converged,
            final_error=final_error,
            n_trials=n_trials,
            trial_lengths=trial_lengths,
            latent_dim=latent_dim,
            config=self.config,
            F_evolution=F_evolution,
            c_evolution=c_evolution,
            data_centers=data_centers
        )
        
        # === Store for later use ===
        self.F_ = F
        self.dynamic_coefficients_ = coefficients
        self.result_ = result
        self._is_fitted = True
        self._data_centers = data_centers
        self._single_trial = single_trial
        
        return result
    
    def reconstruct(self, data=None, return_error=False, to_smooth = True):
        """
        Reconstruct trajectory using learned model.
        
        Parameters
        ----------
        data : np.ndarray or List[np.ndarray], optional
            Data to reconstruct. If None, uses training data structure
            but requires fit() to have been called.
        return_error : bool
            If True, also return per-time-point reconstruction error.
            
        Returns
        -------
        x_hat : np.ndarray or List[np.ndarray]
            Reconstructed trajectory. Same structure as input:
            - If input was single array: returns single array
            - If input was list: returns list
        error : np.ndarray or List[np.ndarray], optional
            Per-time-point squared error (if return_error=True).
        """
        # === Check fitted ===
        assert self._is_fitted, (
            "Model must be fitted before calling reconstruct(). Call fit() first."
        )
        
        # === Validate input ===
        if data is None:
            raise ValueError(
                "data argument is required for reconstruct(). "
                "Pass the same data used in fit() or new data."
            )
        
        data_list, single_trial = self._validate_data(data)
        
        # === Check dimensions match ===
        latent_dim = self.result_.latent_dim
        for i, d in enumerate(data_list):
            assert d.shape[0] == latent_dim, (
                "Data dimension (%d) doesn't match model latent_dim (%d). "
                "Data[%d] has shape %s." % (d.shape[0], latent_dim, i, d.shape)
            )
        
        # === Center data if model was trained with centering ===
        if self._data_centers is not None:
            # Use same centers as training (if same number of trials)
            if len(data_list) == len(self._data_centers):
                data_list = [d - c for d, c in zip(data_list, self._data_centers)]
            else:
                # Different number of trials; center each independently
                data_list, _ = center_data(data_list)
        
        # === Reconstruct ===
        reconstructions = []
        errors = []
        
        for trial_idx, trial_data in enumerate(data_list):
            # Get coefficients (recompute if different data)
            if trial_idx < len(self.dynamic_coefficients_):
                c_trial = self.dynamic_coefficients_[trial_idx]
                # Check if dimensions match
                expected_cols = trial_data.shape[1] - 1
                if c_trial.shape[1] != expected_cols:
                    # Recompute coefficients for this data
                    coefficients_new = self._update_c([trial_data], self.F_)
                    c_trial = coefficients_new[0]
            else:
                # New trial; compute coefficients
                coefficients_new = self._update_c([trial_data], self.F_)
                c_trial = coefficients_new[0]
            
            # Create reconstruction
            x_hat = self._create_reconstruction(trial_data, self.F_, c_trial)
            reconstructions.append(x_hat)
            
            if return_error:
                # Per-time-point squared error
                err = np.sum((trial_data - x_hat) ** 2, axis=0)
                errors.append(err)
        
        # === Un-center if needed ===
        if self._data_centers is not None and len(reconstructions) == len(self._data_centers):
            reconstructions = [r + c for r, c in zip(reconstructions, self._data_centers)]
        
        # === Return in same format as input ===
        if single_trial:
            reconstruction = reconstructions[0]
            if to_smooth:
                reconstruction = gaussian_convolve(reconstruction, direction = 1)
            if return_error:
                return reconstruction, errors[0]
            return reconstruction
        else:
            if return_error:
                return reconstructions, errors
            return reconstructions
    
    def predict(self, x0, n_steps, coefficients=None):
        """
        Predict future trajectory from initial condition.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state, shape (n_dims,) or (n_dims, 1).
        n_steps : int
            Number of steps to predict.
        coefficients : np.ndarray, optional
            Coefficients to use, shape (M, n_steps).
            If None, uses mean of learned coefficients.
            
        Returns
        -------
        trajectory : np.ndarray
            Predicted trajectory, shape (n_dims, n_steps + 1).
            First column is x0, subsequent columns are predictions.
        """
        # === Check fitted ===
        assert self._is_fitted, (
            "Model must be fitted before calling predict(). Call fit() first."
        )
        
        # === Validate x0 ===
        assert isinstance(x0, np.ndarray), (
            "x0 must be np.ndarray, got %s" % type(x0).__name__
        )
        x0 = x0.flatten()
        
        latent_dim = self.result_.latent_dim
        assert x0.shape[0] == latent_dim, (
            "x0 has %d elements but model latent_dim is %d" % (x0.shape[0], latent_dim)
        )
        
        # === Validate n_steps ===
        assert isinstance(n_steps, int) and n_steps >= 1, (
            "n_steps must be a positive integer, got %s" % n_steps
        )
        
        # === Get coefficients ===
        num_subdyns = self.config.num_subdyns
        
        if coefficients is None:
            # Use mean of learned coefficients
            all_c = np.hstack(self.dynamic_coefficients_)  # (M, total_time)
            mean_c = np.mean(all_c, axis=1, keepdims=True)  # (M, 1)
            coefficients = np.tile(mean_c, (1, n_steps))  # (M, n_steps)
        else:
            assert isinstance(coefficients, np.ndarray), (
                "coefficients must be np.ndarray, got %s" % type(coefficients).__name__
            )
            assert coefficients.shape == (num_subdyns, n_steps), (
                "coefficients shape %s doesn't match (num_subdyns=%d, n_steps=%d)" % (
                    coefficients.shape, num_subdyns, n_steps
                )
            )
        
        # === Predict ===
        trajectory = np.zeros((latent_dim, n_steps + 1))
        trajectory[:, 0] = x0
        
        for t in range(n_steps):
            x_t = trajectory[:, t]
            x_next = np.zeros(latent_dim)
            
            for m, f in enumerate(self.F_):
                x_next += coefficients[m, t] * (f @ x_t)
            
            trajectory[:, t + 1] = x_next
        
        return trajectory
    
    def get_combined_dynamics(self, trial_idx=0):
        """
        Get the combined dynamics matrix F_t = Σ_m c_{m,t} * f_m for a trial.
        
        Parameters
        ----------
        trial_idx : int
            Index of the trial (default: 0).
            
        Returns
        -------
        F_combined : np.ndarray
            Shape (p, p, T-1) where F_combined[:, :, t] is the combined
            dynamics matrix at time t.
        """
        assert self._is_fitted, (
            "Model must be fitted first. Call fit()."
        )
        
        return self.result_.get_combined_F(trial_idx)
