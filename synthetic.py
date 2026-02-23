# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 10:11:44 2026

@author: noga mudrik
"""
import numpy as np
def generate_dlds_synthetic_data(latent_dim=5, T=600, num_subdyns=3, noise_std=0.01, seed=42, structured=''):
    """
    Generate synthetic data from a true dLDS model.
    """
    # === Input validation ===
    assert isinstance(latent_dim, int) and latent_dim >= 1, "latent_dim must be int >= 1, got %s" % latent_dim
    assert isinstance(T, int) and T >= 2, "T must be int >= 2, got %s" % T
    assert isinstance(num_subdyns, int) and num_subdyns >= 1, "num_subdyns must be int >= 1, got %s" % num_subdyns
    assert isinstance(noise_std, (int, float)) and noise_std >= 0, "noise_std must be numeric >= 0, got %s" % noise_std
    assert isinstance(seed, int), "seed must be int, got %s" % type(seed).__name__
    assert isinstance(structured, str), "structured must be str, got %s" % type(structured).__name__
    
    np.random.seed(seed)
    
    # === Structured data ===
    if structured:
        if structured.lower() == 'lorenz':
            assert latent_dim == 3, 'if lorenz latent dim need to be 3!'
            x = _generate_lorenz(T=T, noise_std=noise_std, seed=seed)
            assert x.shape == (3, T), "Lorenz output shape mismatch: expected (3, %d), got %s" % (T, x.shape)
            return x, None, None
        else:
            raise NotImplementedError("structured='%s' not implemented. Options: 'lorenz'" % structured)
    
    # === Random dLDS generation ===
    true_F = []
    for i in range(num_subdyns):
        f = np.random.randn(latent_dim, latent_dim) * 0.3
        eig = np.max(np.abs(np.linalg.eigvals(f)))
        if eig > 0.1:
            f = f / eig
        true_F.append(f)
        assert f.shape == (latent_dim, latent_dim), "F[%d] shape mismatch" % i
    
    true_c = np.abs(np.random.randn(num_subdyns, T - 1))
    for t in range(T - 1):
        mask = np.random.rand(num_subdyns) > 0.5
        if mask.sum() == 0:
            mask[np.random.randint(num_subdyns)] = True
        true_c[:, t] *= mask
    true_c = true_c / (np.sum(true_c, axis=0, keepdims=True) + 1e-10)
    assert true_c.shape == (num_subdyns, T - 1), "true_c shape mismatch"
    
    x = np.zeros((latent_dim, T))
    x[:, 0] = np.random.randn(latent_dim)
    
    for t in range(T - 1):
        F_combined = sum(true_c[m, t] * true_F[m] for m in range(num_subdyns))
        x[:, t + 1] = F_combined @ x[:, t] + noise_std * np.random.randn(latent_dim)
    
    assert x.shape == (latent_dim, T), "x shape mismatch"
    assert not np.isnan(x).any(), "x contains NaN"
    assert not np.isinf(x).any(), "x contains Inf"
    
    return x, true_F, true_c


def _generate_lorenz(T=600, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0, noise_std=0.01, seed=42):
    """Generate Lorenz attractor trajectory. Returns shape (3, T)."""
    assert isinstance(T, int) and T >= 2, "T must be int >= 2, got %s" % T
    assert isinstance(dt, (int, float)) and dt > 0, "dt must be positive, got %s" % dt
    assert isinstance(sigma, (int, float)), "sigma must be numeric"
    assert isinstance(rho, (int, float)), "rho must be numeric"
    assert isinstance(beta, (int, float)), "beta must be numeric"
    assert isinstance(noise_std, (int, float)) and noise_std >= 0, "noise_std must be >= 0"
    
    np.random.seed(seed)
    
    x = np.zeros((3, T))
    if seed == 42:
        x[:, 0] = [1.0, 1.0, 1.0]
    else:
        x[:,0] = np.random.rand(3)
    
    for t in range(T - 1):
        x0, y0, z0 = x[:, t]
        dx = sigma * (y0 - x0)
        dy = x0 * (rho - z0) - y0
        dz = x0 * y0 - beta * z0
        x[0, t + 1] = x0 + dt * dx + noise_std * np.random.randn()
        x[1, t + 1] = y0 + dt * dy + noise_std * np.random.randn()
        x[2, t + 1] = z0 + dt * dz + noise_std * np.random.randn()
    
    assert x.shape == (3, T) and not np.isnan(x).any() and not np.isinf(x).any(), "Lorenz output invalid"
    
    return x