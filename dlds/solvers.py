# -*- coding: utf-8 -*-
"""
dLDS Solvers
============

Coefficient solving utilities for dLDS models.
Contains solve_lasso_style which supports multiple optimization backends.

Author: Noga Mudrik (refactored)
"""

import numpy as np
from scipy import linalg
from scipy.optimize import nnls
import warnings


def solve_lasso_style(A, b, l1, l2=0, x0=None, solver='spgl1', solver_params=None, random_state=0):
    """
    Solve the L1-regularized least squares problem:
    
        minimize (1/2) * ||A @ x - b||_2^2 + l1 * ||x||_1
    
    Parameters
    ----------
    A : np.ndarray
        Design matrix, shape (m, n)
    b : np.ndarray
        Target vector, shape (m,) or (m, 1)
    l1 : float
        L1 regularization strength (τ). Must be >= 0.
        If l1 == 0 or solver == 'inv', uses pseudoinverse (no regularization).
    l2 : float
        L2 regularization strength. Must be >= 0. Only used when solver == 'inv'.
    x0 : np.ndarray, optional
        Initial guess for warm start. Shape (n,) or (n, 1).
        Not all solvers support warm start.
    solver : str
        Solver to use. One of:
        - 'inv': Pseudoinverse (least squares, no L1)
        - 'nnls': Non-negative least squares
        - 'lasso': sklearn Lasso
        - 'fista': Fast Iterative Shrinkage-Thresholding Algorithm (pylops)
        - 'ista': Iterative Shrinkage-Thresholding Algorithm (pylops)
        - 'omp': Orthogonal Matching Pursuit (pylops)
        - 'spgl1': Spectral Projected Gradient L1 (pylops) [recommended]
        - 'irls': Iteratively Reweighted Least Squares (pylops)
    solver_params : dict, optional
        Additional parameters for specific solvers:
        - 'num_iters': int, max iterations for iterative solvers (default: 10)
        - 'threshkind': str, 'soft' or 'hard' for FISTA/ISTA (default: 'soft')
    random_state : int
        Random seed for reproducibility (used by sklearn solvers).
        
    Returns
    -------
    x : np.ndarray
        Solution vector, shape (n,)
        
    Raises
    ------
    ValueError
        If solver is unknown or inputs are invalid.
    ImportError
        If required library (pylops, sklearn) is not installed.
        
    Examples
    --------
    >>> A = np.random.randn(100, 50)
    >>> x_true = np.zeros(50)
    >>> x_true[:5] = np.random.randn(5)
    >>> b = A @ x_true + 0.1 * np.random.randn(100)
    >>> x_hat = solve_lasso_style(A, b, l1=0.1, solver='spgl1')
    
    Notes
    -----
    For dLDS coefficient inference, we solve:
        x_{t+1} = (Σ_m c_m * f_m) @ x_t
        
    Which can be written as:
        x_{t+1} = [f_1 @ x_t, f_2 @ x_t, ..., f_M @ x_t] @ c
        
    So A is the stacked f_m @ x_t and we solve for c.
    """
    # === Input validation ===
    assert isinstance(A, np.ndarray), "A must be np.ndarray, got %s" % type(A).__name__
    assert A.ndim == 2, "A must be 2D, got %dD with shape %s" % (A.ndim, A.shape)
    assert isinstance(b, np.ndarray), "b must be np.ndarray, got %s" % type(b).__name__
    assert isinstance(l1, (int, float)) and l1 >= 0, "l1 must be numeric >= 0, got %s" % l1
    assert isinstance(l2, (int, float)) and l2 >= 0, "l2 must be numeric >= 0, got %s" % l2
    
    # === Flatten b if needed ===
    if b.ndim == 2:
        if b.shape[1] == 1 or b.shape[0] == 1:
            b = b.flatten()
        else:
            raise ValueError("b must be 1D or have shape (m, 1) or (1, m), got shape %s" % (b.shape,))
    
    assert b.ndim == 1, "After flattening, b must be 1D, got shape %s" % (b.shape,)
    
    # === Check dimension compatibility ===
    m, n = A.shape
    assert b.shape[0] == m, "A has %d rows but b has %d elements. These must match." % (m, b.shape[0])
    
    # === Check for NaN ===
    if np.isnan(A).any():
        warnings.warn("Warning: A contains NaN values. Results may be unreliable.")
    if np.isnan(b).any():
        warnings.warn("Warning: b contains NaN values. Results may be unreliable.")
    
    # === Setup solver params ===
    if solver_params is None:
        solver_params = {}
    solver_params = {'num_iters': 10, 'threshkind': 'soft', **solver_params}
    
    solver = solver.lower()
    
    # === Solve ===
    
    if solver == 'inv' or l1 == 0:
        if l2 > 0:
            # Ridge via augmented system: [A; sqrt(l2)*I] @ x = [b; 0]
            sqrt_l2 = np.sqrt(l2)
            A_aug = np.vstack([A, sqrt_l2 * np.eye(n)])
            b_aug = np.hstack([b, np.zeros(n)])
            x = linalg.pinv(A_aug) @ b_aug
        else:
            x = linalg.pinv(A) @ b
        
    elif solver == 'nnls':
        x, _ = nnls(A, b)
        
    elif solver == 'lasso':
        try:
            from sklearn import linear_model
        except ImportError:
            raise ImportError("sklearn is required for solver='lasso'. Install with: pip install scikit-learn")
        
        clf = linear_model.Lasso(alpha=l1, random_state=random_state, max_iter=solver_params['num_iters'] * 100)
        clf.fit(A, b)
        x = np.array(clf.coef_)
        
    elif solver == 'fista':
        try:
            import pylops
        except ImportError:
            raise ImportError("pylops is required for solver='fista'. Install with: pip install pylops")
        
        Aop = pylops.MatrixMult(A)
        x = pylops.optimization.sparsity.fista(Aop, b, niter=solver_params['num_iters'], eps=l1, threshkind=solver_params['threshkind'])[0]
        
    elif solver == 'ista':
        try:
            import pylops
        except ImportError:
            raise ImportError("pylops is required for solver='ista'. Install with: pip install pylops")
        
        Aop = pylops.MatrixMult(A)
        x = pylops.optimization.sparsity.ista(Aop, b, niter=solver_params['num_iters'], eps=l1, threshkind=solver_params['threshkind'])[0]
        
    elif solver == 'omp':
        try:
            import pylops
        except ImportError:
            raise ImportError("pylops is required for solver='omp'. Install with: pip install pylops")
        
        Aop = pylops.MatrixMult(A)
        x = pylops.optimization.sparsity.omp(Aop, b, niter_outer=solver_params['num_iters'], sigma=l1)[0]
        
    elif solver == 'spgl1':
        try:
            import pylops
        except ImportError:
            raise ImportError("pylops is required for solver='spgl1'. Install with: pip install pylops")
        
        Aop = pylops.MatrixMult(A)
        x = pylops.optimization.sparsity.spgl1(Aop, b, iter_lim=solver_params['num_iters'], tau=l1)[0]
        
    elif solver == 'irls':
        try:
            import pylops
        except ImportError:
            raise ImportError("pylops is required for solver='irls'. Install with: pip install pylops")
        
        Aop = pylops.MatrixMult(A)
        x = pylops.optimization.sparsity.irls(Aop, b, nouter=50, espI=l1)[0]
        
    else:
        raise ValueError("Unknown solver '%s'. Must be one of: inv, nnls, lasso, fista, ista, omp, spgl1, irls" % solver)
    
    # === Ensure output is 1D ===
    x = np.asarray(x).flatten()
    
    # === Validate output shape ===
    assert x.shape[0] == n, "Solution x has %d elements but A has %d columns. Bug in solver." % (x.shape[0], n)
    
    return x


def solve_coefficients_single_time(f_x_stacked, x_next, l1, solver='spgl1', 
                                    solver_params=None, random_state=0,
                                    smooth_term=0.0, c_prev=None):
    """
    Solve for coefficients c at a single time point.
    
    Given x_{t+1} = (Σ_m c_m * f_m) @ x_t, solve for c.
    
    Parameters
    ----------
    f_x_stacked : np.ndarray
        Stacked [f_1 @ x_t, f_2 @ x_t, ..., f_M @ x_t], shape (p, M)
        where p is latent dimension and M is number of dynamics operators.
    x_next : np.ndarray
        Target x_{t+1}, shape (p,) or (p, 1)
    l1 : float
        L1 regularization strength
    solver : str
        Solver to use (see solve_lasso_style)
    solver_params : dict, optional
        Additional solver parameters
    random_state : int
        Random seed
    smooth_term : float
        If > 0, add smoothness penalty ||c - c_prev||^2
    c_prev : np.ndarray, optional
        Previous time's coefficients for smoothness term
        
    Returns
    -------
    c : np.ndarray
        Coefficients, shape (M,)
    """
    l2 = solver_params.get('l2', 1e-7)



    # === Input validation ===
    p, M = f_x_stacked.shape
    assert isinstance(f_x_stacked, np.ndarray) and f_x_stacked.ndim == 2, "f_x_stacked must be 2D np.ndarray, got %s" % type(f_x_stacked).__name__
    assert isinstance(x_next, np.ndarray), "x_next must be np.ndarray, got %s" % type(x_next).__name__
    x_next = x_next.flatten()
    assert x_next.shape[0] == p, "x_next has %d elements but f_x_stacked has %d rows" % (x_next.shape[0], p)
    
    # === Add smoothness term if specified ===
    if smooth_term > 0 and c_prev is not None:
        assert isinstance(c_prev, np.ndarray), (
            "c_prev must be np.ndarray when smooth_term > 0, got %s" % type(c_prev).__name__
        )
        c_prev = c_prev.flatten()
        assert c_prev.shape[0] == M, (
            "c_prev has %d elements but there are %d operators (M). These must match." % (
                c_prev.shape[0], M
            )
        )
        
        # Augment the system:
        # minimize ||A @ c - b||^2 + smooth_term * ||c - c_prev||^2
        # = ||[A; sqrt(smooth_term)*I] @ c - [b; sqrt(smooth_term)*c_prev]||^2
        sqrt_smooth = np.sqrt(smooth_term)
        A_aug = np.vstack([f_x_stacked, sqrt_smooth * np.eye(M)])
        b_aug = np.hstack([x_next, sqrt_smooth * c_prev])
        
        c = solve_lasso_style(A_aug, b_aug, l1, solver=solver, 
                              solver_params=solver_params, random_state=random_state)
    else:
        c = solve_lasso_style(f_x_stacked, x_next, l1, solver=solver,
                              solver_params=solver_params, random_state=random_state)
    
    # === Validate output ===
    assert c.shape[0] == M, (
        "Solved c has %d elements but there are %d operators (M). Bug in solver." % (c.shape[0], M)
    )
    
    return c
