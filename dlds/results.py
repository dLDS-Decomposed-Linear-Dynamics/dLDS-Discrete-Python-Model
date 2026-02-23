# -*- coding: utf-8 -*-
"""
dLDS Results
============

Results dataclass for dLDS model fitting.

Author: Noga Mudrik (refactored)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class dLDS_result:
    """
    Results from dLDS model fitting.
    
    Attributes
    ----------
    F : List[np.ndarray]
        List of M learned dynamics operators, each shape (p, p)
        where p is the latent dimension.
        
    dynamic_coefficients : List[np.ndarray]
        List of coefficient matrices, one per trial.
        Each matrix has shape (M, T_trial - 1) where T_trial is 
        the number of time points in that trial.
        
    error_history : List[float]
        Reconstruction error (scalar) at each iteration.
        
    n_iterations : int
        Total number of iterations run.
        
    converged : bool
        True if training converged (error < max_error) before max_iter.
        
    final_error : float
        Final reconstruction error.
        
    n_trials : int
        Number of trials in the fitted data.
        
    trial_lengths : List[int]
        Number of time points in each trial.
        
    latent_dim : int
        Dimension of the latent space (p).
        
    config : dLDS_config
        Configuration used for fitting.
        
    F_evolution : List[List[np.ndarray]], optional
        F at each iteration. Only populated if config.track_evolution=True.
        
    c_evolution : List[List[np.ndarray]], optional
        Coefficients at each iteration. Only populated if config.track_evolution=True.
        
    data_centers : List[np.ndarray], optional
        Centers used for centering data. Only populated if config.center_data=True.
        
    Examples
    --------
    >>> result = model.fit(data)
    >>> print(result.F)  # List of dynamics operators
    >>> print(result.dynamic_coefficients)  # List of coefficient matrices
    >>> print(result.final_error)  # Final reconstruction error
    """
    
    # === Learned parameters ===
    F: List[np.ndarray]
    dynamic_coefficients: List[np.ndarray]
    
    # === Training info ===
    error_history: List[float]
    n_iterations: int
    converged: bool
    final_error: float
    
    # === Data info ===
    n_trials: int
    trial_lengths: List[int]
    latent_dim: int
    
    # === Config used ===
    config: object = None  # dLDS_config, but avoid circular import
    
    # === Optional evolution tracking ===
    F_evolution: Optional[List] = None
    c_evolution: Optional[List] = None
    
    # === Optional centering info ===
    data_centers: Optional[List[np.ndarray]] = None
    
    def __post_init__(self):
        """Validate result contents."""
        # Validate F
        assert isinstance(self.F, list), (
            "F must be a list, got %s" % type(self.F).__name__
        )
        assert len(self.F) > 0, "F must not be empty"
        for i, f in enumerate(self.F):
            assert isinstance(f, np.ndarray), (
                "F[%d] must be np.ndarray, got %s" % (i, type(f).__name__)
            )
            assert f.ndim == 2, (
                "F[%d] must be 2D, got %dD with shape %s" % (i, f.ndim, f.shape)
            )
            assert f.shape[0] == f.shape[1], (
                "F[%d] must be square, got shape %s" % (i, f.shape)
            )
        
        # Validate dynamic_coefficients
        assert isinstance(self.dynamic_coefficients, list), (
            "dynamic_coefficients must be a list, got %s" % type(self.dynamic_coefficients).__name__
        )
        assert len(self.dynamic_coefficients) > 0, "dynamic_coefficients must not be empty"
        for i, c in enumerate(self.dynamic_coefficients):
            assert isinstance(c, np.ndarray), (
                "dynamic_coefficients[%d] must be np.ndarray, got %s" % (i, type(c).__name__)
            )
            assert c.ndim == 2, (
                "dynamic_coefficients[%d] must be 2D, got %dD with shape %s" % (i, c.ndim, c.shape)
            )
        
        # Validate consistency
        num_subdyns = len(self.F)
        for i, c in enumerate(self.dynamic_coefficients):
            assert c.shape[0] == num_subdyns, (
                "dynamic_coefficients[%d] has %d rows but there are %d operators in F. "
                "These must match." % (i, c.shape[0], num_subdyns)
            )
        
        # Validate error_history
        assert isinstance(self.error_history, list), (
            "error_history must be a list, got %s" % type(self.error_history).__name__
        )
        
        # Validate n_iterations
        assert isinstance(self.n_iterations, int), (
            "n_iterations must be int, got %s" % type(self.n_iterations).__name__
        )
        assert self.n_iterations >= 0, (
            "n_iterations must be >= 0, got %d" % self.n_iterations
        )
        
        # Validate n_trials matches dynamic_coefficients
        assert self.n_trials == len(self.dynamic_coefficients), (
            "n_trials (%d) must match len(dynamic_coefficients) (%d)" % (
                self.n_trials, len(self.dynamic_coefficients)
            )
        )
        
        # Validate trial_lengths
        assert len(self.trial_lengths) == self.n_trials, (
            "len(trial_lengths) (%d) must match n_trials (%d)" % (
                len(self.trial_lengths), self.n_trials
            )
        )
        
        # Validate trial_lengths matches coefficient shapes
        for i, (length, c) in enumerate(zip(self.trial_lengths, self.dynamic_coefficients)):
            expected_c_cols = length - 1
            assert c.shape[1] == expected_c_cols, (
                "dynamic_coefficients[%d] has %d columns but trial %d has length %d "
                "(expected %d columns = length - 1)" % (
                    i, c.shape[1], i, length, expected_c_cols
                )
            )
    
    @property
    def num_subdyns(self):
        """Number of dynamics operators M."""
        return len(self.F)
    
    def summary(self):
        """Print a summary of the results."""
        print("=" * 50)
        print("dLDS Fitting Results")
        print("=" * 50)
        print("Converged: %s" % self.converged)
        print("Iterations: %d" % self.n_iterations)
        #print("Final error: %.6f" % self.final_error)
        print("-" * 50)
        print("Number of dynamics operators (M): %d" % self.num_subdyns)
        print("Latent dimension (p): %d" % self.latent_dim)
        print("Number of trials: %d" % self.n_trials)
        print("Trial lengths: %s" % self.trial_lengths)
        print("-" * 50)
        print("F shapes: %s" % [f.shape for f in self.F])
        print("Coefficient shapes: %s" % [c.shape for c in self.dynamic_coefficients])
        print("=" * 50)
    
    def get_combined_F(self, trial_idx=0):
        """
        Get the combined dynamics matrix F_t = Σ_m c_{m,t} * f_m for a trial.
        
        Parameters
        ----------
        trial_idx : int
            Index of the trial (default: 0)
            
        Returns
        -------
        F_combined : np.ndarray
            Shape (p, p, T-1) where F_combined[:, :, t] is the combined
            dynamics matrix at time t.
        """
        assert 0 <= trial_idx < self.n_trials, (
            "trial_idx must be in [0, %d), got %d" % (self.n_trials, trial_idx)
        )
        
        c = self.dynamic_coefficients[trial_idx]  # (M, T-1)
        n_times = c.shape[1]
        p = self.latent_dim
        
        F_combined = np.zeros((p, p, n_times))
        for t in range(n_times):
            for m, f in enumerate(self.F):
                F_combined[:, :, t] += c[m, t] * f
        
        return F_combined



@dataclass
class dLDS_latent_result:
    """Results from dLDS with latents fitting."""
    F: List[np.ndarray]
    D: np.ndarray
    x: List[np.ndarray]
    dynamic_coefficients: List[np.ndarray]
    error_history: List[float]
    n_iterations: int
    converged: bool
    final_error: float
    n_trials: int
    trial_lengths: List[int]
    latent_dim: int
    obs_dim: int
    config: object = None