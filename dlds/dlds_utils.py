# -*- coding: utf-8 -*-
"""
dLDS Utilities
==============

Utility functions for dLDS models including initialization,
normalization, and validation.

Author: Noga Mudrik (refactored)
"""

import numpy as np
from scipy import linalg
import random
import warnings
from typing import List, Tuple, Union


def init_mat(size_mat, r_seed=0, dist_type='norm', init_params=None, normalize=False):
    """
    Initialize a matrix with specified distribution.
    
    Parameters
    ----------
    size_mat : tuple of int
        Shape of the matrix to create, (rows, cols).
    r_seed : int
        Random seed for reproducibility.
    dist_type : str
        Distribution type for initialization:
        - 'norm': Normal distribution (default)
        - 'uni': Uniform distribution
        - 'inti': Integer uniform distribution
        - 'sparse': Sparse matrix with random non-zeros per column
        - 'regional': Block-diagonal style initialization
    init_params : dict, optional
        Parameters for the distribution:
        - For 'norm': {'loc': 0, 'scale': 1}
        - For 'uni'/'inti': {'low': 0, 'high': 1}
        - For 'sparse': {'k': max_nonzeros_per_column}
        - For 'regional': {'k': num_repeats}
    normalize : bool
        If True, normalize by spectral radius (for square matrices).
        
    Returns
    -------
    mat : np.ndarray
        Initialized matrix of shape size_mat.
        
    Raises
    ------
    ValueError
        If dist_type is unknown.
    KeyError
        If required init_params keys are missing.
        
    Examples
    --------
    >>> mat = init_mat((5, 5), r_seed=42, dist_type='norm')
    >>> mat = init_mat((5, 5), dist_type='uni', init_params={'low': -1, 'high': 1})
    """
    # === Input validation ===
    assert isinstance(size_mat, (tuple, list)), (
        "size_mat must be tuple or list, got %s" % type(size_mat).__name__
    )
    assert len(size_mat) == 2, (
        "size_mat must have 2 elements (rows, cols), got %d elements" % len(size_mat)
    )
    assert all(isinstance(s, int) and s > 0 for s in size_mat), (
        "size_mat elements must be positive integers, got %s" % (size_mat,)
    )
    
    # === Set default init_params ===
    if init_params is None:
        init_params = {}
    
    # === Set seeds ===
    np.random.seed(r_seed)
    random.seed(r_seed)
    
    # === Generate matrix ===
    if dist_type == 'norm':
        init_params = {**{'loc': 0, 'scale': 1}, **init_params}
        rand_mat = np.random.normal(
            loc=init_params['loc'],
            scale=init_params['scale'],
            size=size_mat
        )
        
    elif dist_type == 'uni':
        if 'high' not in init_params or 'low' not in init_params:
            raise KeyError(
                "For dist_type='uni', init_params must contain 'low' and 'high' keys. "
                "Got keys: %s" % list(init_params.keys())
            )
        rand_mat = np.random.uniform(
            init_params['low'],
            init_params['high'],
            size=size_mat
        )
        
    elif dist_type == 'inti':
        if 'high' not in init_params or 'low' not in init_params:
            raise KeyError(
                "For dist_type='inti', init_params must contain 'low' and 'high' keys. "
                "Got keys: %s" % list(init_params.keys())
            )
        rand_mat = np.random.randint(
            init_params['low'],
            init_params['high'],
            size=size_mat
        )
        
    elif dist_type == 'sparse':
        if 'k' not in init_params:
            raise KeyError(
                "For dist_type='sparse', init_params must contain 'k' key (max nonzeros per col). "
                "Got keys: %s" % list(init_params.keys())
            )
        k = init_params['k']
        rows, cols = size_mat
        
        # For each column, randomly select up to k rows to be nonzero
        rand_mat = np.zeros(size_mat)
        for j in range(cols):
            num_nonzero = np.random.randint(1, min(rows, k) + 1)
            nonzero_rows = random.sample(range(rows), num_nonzero)
            rand_mat[nonzero_rows, j] = 1
            
    elif dist_type == 'regional':
        if 'k' not in init_params:
            raise KeyError(
                "For dist_type='regional', init_params must contain 'k' key (num repeats). "
                "Got keys: %s" % list(init_params.keys())
            )
        k = init_params['k']
        rows, cols = size_mat
        
        # Create block-diagonal style pattern
        splits = np.array_split(np.arange(cols), k)
        split_lens = [len(s) for s in splits]
        
        blocks = []
        for split_len in split_lens:
            # Repeat identity to fill the block
            n_repeats = int(np.ceil(split_len / rows))
            block = np.tile(np.eye(rows), n_repeats)[:, :split_len]
            blocks.append(block)
        
        rand_mat = np.hstack(blocks)
        # Trim to exact size
        rand_mat = rand_mat[:rows, :cols]
        
    else:
        raise ValueError(
            "Unknown dist_type '%s'. Must be one of: norm, uni, inti, sparse, regional" % dist_type
        )
    
    # === Normalize if requested ===
    if normalize:
        rand_mat = norm_mat(rand_mat, type_norm='evals', to_norm=True)
    
    # === Final validation ===
    assert rand_mat.shape == tuple(size_mat), (
        "Generated matrix has shape %s but expected %s. Bug in init_mat." % (
            rand_mat.shape, size_mat
        )
    )
    
    return rand_mat


def norm_mat(mat, type_norm='evals', to_norm=True):
    """
    Normalize a matrix by its spectral radius.
    
    Parameters
    ----------
    mat : np.ndarray
        Matrix to normalize (must be square for eigenvalue normalization).
    type_norm : str
        Normalization type: 'evals' (spectral radius).
    to_norm : bool
        If False, returns mat unchanged.
        
    Returns
    -------
    normalized_mat : np.ndarray
        Normalized matrix (same shape as input).
        
    Notes
    -----
    Spectral radius normalization ensures the largest eigenvalue 
    has magnitude 1, which helps with stability of the dynamics.
    """
    if not to_norm:
        return mat
    
    # === Input validation ===
    assert isinstance(mat, np.ndarray), (
        "mat must be np.ndarray, got %s" % type(mat).__name__
    )
    assert mat.ndim == 2, (
        "mat must be 2D, got %dD with shape %s" % (mat.ndim, mat.shape)
    )
    
    if type_norm == 'evals':
        assert mat.shape[0] == mat.shape[1], (
            "For eigenvalue normalization, mat must be square. Got shape %s" % (mat.shape,)
        )
        
        eigenvalues, _ = linalg.eig(mat)
        spectral_radius = np.max(np.abs(eigenvalues))
        
        if spectral_radius > 1e-10:
            mat = mat / spectral_radius
        else:
            warnings.warn(
                "Matrix has near-zero spectral radius (%.2e). "
                "Normalization skipped to avoid division by zero." % spectral_radius
            )
    else:
        raise ValueError(
            "Unknown type_norm '%s'. Currently only 'evals' is supported." % type_norm
        )
    
    return mat


def validate_data(data) -> Tuple[List[np.ndarray], bool]:
    """
    Validate and normalize input data for dLDS.
    
    Parameters
    ----------
    data : np.ndarray or List[np.ndarray]
        Input data. If np.ndarray, must be 2D with shape (n_dims, T).
        If list, each element must be 2D with shape (n_dims, T_i).
        All trials must have the same n_dims.
        
    Returns
    -------
    data_list : List[np.ndarray]
        List of 2D arrays, one per trial.
    single_trial : bool
        True if input was a single array (not a list).
        
    Raises
    ------
    AssertionError
        If data format is invalid.
        
    Examples
    --------
    >>> data = np.random.randn(5, 100)
    >>> data_list, single = validate_data(data)
    >>> len(data_list)
    1
    >>> single
    True
    
    >>> data = [np.random.randn(5, 100), np.random.randn(5, 80)]
    >>> data_list, single = validate_data(data)
    >>> len(data_list)
    2
    >>> single
    False
    """
    single_trial = False
    
    # === Handle single array ===
    if isinstance(data, np.ndarray):
        assert data.ndim == 2, (
            "If data is np.ndarray, it must be 2D (n_dims, T). "
            "Got %dD array with shape %s" % (data.ndim, data.shape)
        )
        assert data.shape[0] > 0 and data.shape[1] > 1, (
            "Data must have shape (n_dims, T) with n_dims > 0 and T > 1. "
            "Got shape %s" % (data.shape,)
        )
        data_list = [data]
        single_trial = True
        
    # === Handle list ===
    elif isinstance(data, list):
        assert len(data) > 0, "Data list must not be empty"
        
        # Check if single-element list
        if len(data) == 1:
            single_trial = True
        
        # Validate each element
        n_dims = None
        data_list = []
        
        for i, trial in enumerate(data):
            assert isinstance(trial, np.ndarray), (
                "data[%d] must be np.ndarray, got %s" % (i, type(trial).__name__)
            )
            assert trial.ndim == 2, (
                "data[%d] must be 2D, got %dD with shape %s" % (i, trial.ndim, trial.shape)
            )
            assert trial.shape[1] > 1, (
                "data[%d] must have T > 1 time points. Got shape %s" % (i, trial.shape)
            )
            
            if n_dims is None:
                n_dims = trial.shape[0]
            else:
                assert trial.shape[0] == n_dims, (
                    "All trials must have same n_dims. "
                    "data[0] has %d dims but data[%d] has %d dims" % (
                        n_dims, i, trial.shape[0]
                    )
                )
            
            data_list.append(trial)
    else:
        raise AssertionError(
            "data must be np.ndarray or list of np.ndarray, got %s" % type(data).__name__
        )
    
    return data_list, single_trial


def check_f_correlation(F_list):
    """
    Compute maximum correlation between pairs of dynamics operators.
    
    Parameters
    ----------
    F_list : List[np.ndarray]
        List of M dynamics operators, each (p, p).
        
    Returns
    -------
    max_corr : float
        Maximum absolute correlation between any pair.
    pair : tuple
        Indices (i, j) of the most correlated pair.
    """
    assert len(F_list) >= 2, (
        "Need at least 2 operators to compute correlation, got %d" % len(F_list)
    )
    
    max_corr = 0.0
    pair = (0, 1)
    
    for i in range(len(F_list)):
        for j in range(i + 1, len(F_list)):
            # Flatten and compute correlation
            f_i = F_list[i].flatten()
            f_j = F_list[j].flatten()
            
            # Normalize
            norm_i = np.linalg.norm(f_i)
            norm_j = np.linalg.norm(f_j)
            
            if norm_i > 1e-10 and norm_j > 1e-10:
                corr = np.abs(np.dot(f_i, f_j) / (norm_i * norm_j))
            else:
                corr = 0.0
            
            if corr > max_corr:
                max_corr = corr
                pair = (i, j)
    
    return max_corr, pair


def init_distant_f(latent_dim, num_subdyns, max_corr=0.1, max_retries=100, 
                   normalize=True, seed=0):
    """
    Initialize dynamics operators with low mutual correlation.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent space (p).
    num_subdyns : int
        Number of dynamics operators (M).
    max_corr : float
        Maximum allowed correlation between any pair.
    max_retries : int
        Maximum attempts to achieve low correlation.
    normalize : bool
        If True, normalize each operator by spectral radius.
    seed : int
        Random seed.
        
    Returns
    -------
    F_list : List[np.ndarray]
        List of M operators, each (p, p).
    achieved_corr : float
        Achieved maximum correlation.
    """
    assert latent_dim >= 1, "latent_dim must be >= 1, got %d" % latent_dim
    assert num_subdyns >= 1, "num_subdyns must be >= 1, got %d" % num_subdyns
    assert 0 < max_corr <= 1, "max_corr must be in (0, 1], got %s" % max_corr
    
    best_F = None
    best_corr = np.inf
    
    for attempt in range(max_retries):
        # Initialize with different seed each attempt
        F_list = [
            init_mat((latent_dim, latent_dim), r_seed=seed + attempt * num_subdyns + i, 
                     normalize=normalize)
            for i in range(num_subdyns)
        ]
        
        if num_subdyns == 1:
            return F_list, 0.0
        
        current_corr, _ = check_f_correlation(F_list)
        
        if current_corr < best_corr:
            best_corr = current_corr
            best_F = F_list
        
        if current_corr <= max_corr:
            return F_list, current_corr
    
    # Did not achieve desired correlation
    warnings.warn(
        "Could not achieve max_corr=%.3f after %d attempts. "
        "Best achieved: %.3f. Consider either:\n"
        "  1. Reducing num_subdyns from %d\n"
        "  2. Increasing max_corr from %.3f\n"
        "  3. Increasing max_init_retries from %d\n"
        "Using best found initialization." % (
            max_corr, max_retries, best_corr, num_subdyns, max_corr, max_retries
        )
    )
    
    return best_F, best_corr


def center_data(data_list):
    """
    Center each trial around zero.
    
    Parameters
    ----------
    data_list : List[np.ndarray]
        List of trials, each (n_dims, T).
        
    Returns
    -------
    centered_data : List[np.ndarray]
        Centered trials.
    centers : List[np.ndarray]
        Mean of each trial, shape (n_dims, 1).
    """
    centered_data = []
    centers = []
    
    for trial in data_list:
        center = np.mean(trial, axis=1, keepdims=True)
        centered = trial - center
        centered_data.append(centered)
        centers.append(center)
    
    return centered_data, centers


def compute_spectral_radius(mat):
    """
    Compute the spectral radius of a matrix.
    
    Parameters
    ----------
    mat : np.ndarray
        Square matrix.
        
    Returns
    -------
    spectral_radius : float
        Maximum absolute eigenvalue.
    """
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], (
        "mat must be square, got shape %s" % (mat.shape,)
    )
    
    eigenvalues, _ = linalg.eig(mat)
    return np.max(np.abs(eigenvalues))
