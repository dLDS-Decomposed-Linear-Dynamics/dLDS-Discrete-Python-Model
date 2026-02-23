# -*- coding: utf-8 -*-
"""
dLDS: Decomposed Linear Dynamical Systems
==========================================

A Python package for learning decomposed linear dynamical systems
from time series data.

The model decomposes dynamics into a weighted combination of 
basis operators:
    x_t = (Σ_m c_{m,t} * f_m) @ x_{t-1}

where:
    - f_m are learned dynamics operators (M matrices of size p×p)
    - c_{m,t} are time-varying, sparse coefficients
    - x_t is the latent state at time t

References
----------
Mudrik, N., et al. (2024). "Decomposed Linear Dynamical Systems (dLDS) 
for learning the latent components of neural dynamics." JMLR.

Example Usage
-------------
>>> from dlds import Direct_dLDS, dLDS_config
>>>
>>> # Simple usage
>>> model = Direct_dLDS(num_subdyns=3)
>>> result = model.fit(data)  # data: (n_dims, T) or list of arrays
>>>
>>> # Access learned parameters
>>> print(result.F)  # List of dynamics operators
>>> print(result.dynamic_coefficients)  # List of coefficient matrices
>>>
>>> # Reconstruct
>>> x_hat = model.reconstruct(data)

Author: Noga Mudrik
"""

__version__ = '0.1.0'
__author__ = 'Noga Mudrik'

# Core classes
from .config import dLDS_config
from .results import dLDS_result
from .direct import Direct_dLDS

# Utilities
from .solvers import solve_lasso_style
from .dlds_utils import (
    init_mat,
    norm_mat,
    validate_data,
    init_distant_f,
    center_data,
    check_f_correlation,
    compute_spectral_radius
)
from .dlds_with_latents import dLDS_with_latents
from .results import dLDS_latent_result
# Base class (for advanced users / extending)
#from .base import Base_dLDS

__all__ = [
    # Main classes
    'dLDS_config',
    'dLDS_result',
    'dLDS_latent_result',
    'Direct_dLDS',
    'dLDS_with_latents',
    
    # Solvers
    'solve_lasso_style',
    
    # Utilities
    'init_mat',
    'norm_mat',
    'validate_data',
    'init_distant_f',
    'center_data',
    'check_f_correlation',
    'compute_spectral_radius',
]
