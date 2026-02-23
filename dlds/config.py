# -*- coding: utf-8 -*-
"""
dLDS Configuration
==================

Configuration dataclass for dLDS models.

Author: Noga Mudrik (refactored)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class dLDS_config:
    """
    Configuration for dLDS model.
    
    Parameters
    ----------
    num_subdyns : int
        Number of dynamics operators M (default: 3)
    latent_dim : int, optional
        Latent dimension p. If None, inferred from data.
        
    step_f : float
        Initial learning rate for F gradient descent (default: 10.0)
    gd_decay : float
        Decay rate for step_f each iteration. 1.0 = no decay (default: 1.0)
    max_iter : int
        Maximum training iterations (default: 5000)
    max_error : float
        Convergence threshold - stop if error < max_error (default: 1e-3)
    min_step_f : float
        Minimum step size for F (default: 1e-6)
        
    solver : str
        Coefficient solver. One of: 'inv', 'spgl1', 'lasso', 'fista', 
        'ista', 'omp', 'irls', 'nnls' (default: 'spgl1')
    reg_term : float
        L1 regularization strength τ for coefficients (default: 0.1)
    smooth_term : float
        Temporal smoothness penalty on c (default: 0.0)
    solver_max_iters : int
        Max iterations for iterative solvers (default: 10)
    decaying_reg : float
        Decay factor for reg_term each iteration. 1.0 = no decay (default: 1.0)
        
    init_distant_f : bool
        If True, initialize F's to be uncorrelated (default: True)
    max_corr : float
        Max allowed correlation between F pairs when init_distant_f=True (default: 0.1)
    max_init_retries : int
        Max retries for init_distant_f before warning (default: 100)
    normalize_eig : bool
        If True, normalize each F by its spectral radius (default: True)
    center_data : bool
        If True, center data around 0 before fitting (default: True)
    seed : int
        Random seed for reproducibility (default: 0)
        
    use_mini_batch : bool
        If True, use mini-batching over time (default: False)
    batch_size : int
        Number of time points per batch when use_mini_batch=True (default: 200)
        
    num_no_change_thresh : int
        Iterations without improvement before perturbing F (default: 5)
    sigma_perturbation : float
        Std of Gaussian noise added to F when stuck (default: 0.1)
        
    error_order_max : int
        Multi-step prediction error. 1 = single step only (default: 1)
    action_along_time : str
        How to aggregate gradients over time: 'mean' or 'median' (default: 'mean')
        
    track_evolution : bool
        If True, store F and c at each iteration (default: False)
        
    verbose : int
        Verbosity level: 0=silent, 1=progress bar, 2=detailed (default: 1)
        
    Examples
    --------
    >>> config = dLDS_config(num_subdyns=3, solver='spgl1', reg_term=0.1)
    >>> config = dLDS_config(num_subdyns=5, step_f=20.0, max_iter=10000)
    """
    
    # === Core ===
    num_subdyns: int = 3  # How many dynamics operators (F matrices) to learn. More = more complex model
    latent_dim: Optional[int] = 5# None  # Dimension of latent space. If None, will be set from data
    
    # === For indirect observation model (dLDS_with_latents) ===
    #lambda_dyn: float = 0.1  # Balance between observation fit and dynamics fit. 0 = trust observations, 1 = trust dynamics
    
    # === Optimization (how the model learns) ===
    step_f: float = 10.0  # Learning rate for F. Bigger = faster but may overshoot. Smaller = slower but more stable
    gd_decay: float = 1.0  # Multiply step_f by this each iteration. 1.0 = no decay, 0.99 = slow decay
    max_iter: int = 5000  # Maximum number of training iterations. Stop after this even if not converged
    
    # === Error thresholds (when to stop training) ===
    # === For indirect observation model (dLDS_with_latents) ===
    lambda_dyn: float = 0.1  # Balance between observation fit and dynamics fit. 0 = trust observations, 1 = trust dynamics
    w_dyn: float = 0.7  # Weight for dynamics error in total error (0 to 1). Obs weight = 1 - w_dyn
    # NOTE: lambda_dyn controls HOW we infer x from y (in _update_x). w_dyn controls WHEN we stop training (in _compute_error).
    # They serve different purposes: lambda_dyn affects the optimization process, w_dyn affects the convergence criterion.


    max_error: float = 0.007  # Stop if error goes below this (absolute value)
    max_error_relative: float = 0.05  # Stop if error < this fraction of data variance (e.g., 0.05 = 5%)
    use_relative_error: bool = True  # If True, use max_error_relative. If False, use max_error
    
    min_step_f: float = 1e-6  # Never let step_f go below this. Prevents getting stuck at zero
    
    l2_reg: float = 1e-3  # Ridge regularization for 'inv' solver. Keeps coefficients from exploding
    solver_params = {'l2_reg':l2_reg}
    # === Backtracking (what to do when update makes error worse) ===
    max_backtrack: int = 25  # How many times to try reducing step_f when error increases
    backtrack_factor: float = 0.5  # Multiply step_f by this when backtracking. 0.5 = cut in half each time
    
    # === Coefficient Solver (how to find c given F) ===
    solver: str = 'spgl1'  # Which solver to use: 'spgl1' (recommended), 'lasso', 'inv' (fast, no sparsity), 'nnls' (non-negative)
    reg_term: float = 0.1  # Sparsity control. For spgl1: higher = less sparse. For lasso: higher = more sparse
    smooth_term: float = 0.0  # Smoothness of coefficients over time. 0 = no smoothness, higher = smoother c(t)
    solver_max_iters: int = 10  # Max iterations for iterative solvers (fista, ista, etc.)
    decaying_reg: float = 1.0  # Multiply reg_term by this each iteration. 1.0 = no decay
    
    # === Initialization (how to start F matrices) ===
    init_distant_f: bool = True  # If True, initialize F matrices to be uncorrelated (diverse)
    max_corr: float = 0.1  # Max allowed correlation between F pairs when init_distant_f=True
    max_init_retries: int = 100  # How many times to try finding uncorrelated F before giving up
    normalize_eig: bool = True  # If True, normalize each F so its largest eigenvalue = 1 (keeps dynamics stable)
    center_data: bool = True  # If True, subtract mean from data before fitting
    seed: int = 0  # Random seed for reproducibility. Same seed = same results
    
    # === Perturbations (what to do when stuck) ===
    relative_change_param: float = 1e-3  # If error changes less than this fraction, consider "stuck"
    num_no_change_thresh: int = 40  # How many iterations of no improvement before perturbing F
    sigma_perturbation: float = 0.5  # How much noise to add to F when stuck. Bigger = bigger shake-up
    
    # === Mini-batching (for very long time series) ===
    use_mini_batch: bool = False  # If True, use random chunks of data each iteration (faster for long data)
    batch_size: int = 200  # How many time points per batch when use_mini_batch=True
    
    # === Advanced ===
    error_order_max: int = 1  # How many steps ahead to predict. 1 = one-step prediction only
    action_along_time: str = 'mean'  # How to combine gradients over time: 'mean' or 'median'
    
    # === Tracking (for debugging/visualization) ===
    track_evolution: bool = False  # If True, save F and c at every iteration (uses more memory)
    
    # === Verbosity (how much to print) ===
    verbose: int = 1  # 0 = silent, 1 = progress bar, 2 = detailed messages
    
    def __post_init__(self):
        """
        Validate configuration parameters.
        
        This runs automatically after creating the config.
        It checks that all values make sense and raises errors if not.
        """
        # Check num_subdyns is a positive integer
        assert isinstance(self.num_subdyns, int), "num_subdyns must be int, got %s" % type(self.num_subdyns).__name__
        assert self.num_subdyns >= 1, "num_subdyns must be >= 1, got %d" % self.num_subdyns
        
        # Check latent_dim if provided
        if self.latent_dim is not None:
            assert isinstance(self.latent_dim, int), "latent_dim must be int or None, got %s" % type(self.latent_dim).__name__
            assert self.latent_dim >= 1, "latent_dim must be >= 1, got %d" % self.latent_dim
        
        # Check step_f is positive number
        assert isinstance(self.step_f, (int, float)), "step_f must be numeric, got %s" % type(self.step_f).__name__
        assert self.step_f > 0, "step_f must be > 0, got %s" % self.step_f
        
        # Check gd_decay is between 0 and 1
        assert isinstance(self.gd_decay, (int, float)), "gd_decay must be numeric, got %s" % type(self.gd_decay).__name__
        assert 0 < self.gd_decay <= 1, "gd_decay must be in (0, 1], got %s" % self.gd_decay
        
        # Check max_iter is positive integer
        assert isinstance(self.max_iter, int), "max_iter must be int, got %s" % type(self.max_iter).__name__
        assert self.max_iter >= 1, "max_iter must be >= 1, got %d" % self.max_iter
        
        # Check max_error is positive number
        assert isinstance(self.max_error, (int, float)), "max_error must be numeric, got %s" % type(self.max_error).__name__
        assert self.max_error > 0, "max_error must be > 0, got %s" % self.max_error
        
        # Check solver is valid
        valid_solvers = ['inv', 'spgl1', 'lasso', 'fista', 'ista', 'omp', 'irls', 'nnls']
        assert self.solver.lower() in valid_solvers, "solver must be one of %s, got '%s'" % (valid_solvers, self.solver)
        
        # Check reg_term is non-negative
        assert isinstance(self.reg_term, (int, float)), "reg_term must be numeric, got %s" % type(self.reg_term).__name__
        assert self.reg_term >= 0, "reg_term must be >= 0, got %s" % self.reg_term
        
        # Check smooth_term is non-negative
        assert isinstance(self.smooth_term, (int, float)), "smooth_term must be numeric, got %s" % type(self.smooth_term).__name__
        assert self.smooth_term >= 0, "smooth_term must be >= 0, got %s" % self.smooth_term
        
        # Check max_corr is between 0 and 1
        assert isinstance(self.max_corr, (int, float)), "max_corr must be numeric, got %s" % type(self.max_corr).__name__
        assert 0 < self.max_corr <= 1, "max_corr must be in (0, 1], got %s" % self.max_corr
        
        # Check action_along_time is valid
        assert self.action_along_time in ['mean', 'median'], "action_along_time must be 'mean' or 'median', got '%s'" % self.action_along_time
        
        # Check verbose is valid
        assert self.verbose in [0, 1, 2], "verbose must be 0, 1, or 2, got %s" % self.verbose
        
        # Check decaying_reg is between 0 and 1
        assert isinstance(self.decaying_reg, (int, float)), "decaying_reg must be numeric, got %s" % type(self.decaying_reg).__name__
        assert 0 < self.decaying_reg <= 1, "decaying_reg must be in (0, 1], got %s" % self.decaying_reg
    
    def copy(self):
        """
        Return a copy of this config.
        
        Use this if you want to make a new config based on an existing one
        without changing the original.
        """
        import copy
        return copy.deepcopy(self)
    
    def to_dict(self):
        """
        Convert config to a dictionary.
        
        Useful for saving config to a file or printing all settings.
        """
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        """
        Create config from a dictionary.
        
        Useful for loading config from a file.
        """
        return cls(**d)


