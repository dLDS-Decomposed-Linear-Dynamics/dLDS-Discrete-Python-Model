# -*- coding: utf-8 -*-
"""
dLDS Example Usage
==================

This script demonstrates how to use the Direct_dLDS model
to fit decomposed linear dynamical systems to data.

Author: Noga Mudrik (refactored)
"""

import numpy as np
import matplotlib.pyplot as plt

# Add parent to path if running directly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlds import Direct_dLDS, dLDS_config
from dlds.vis import *

def generate_dlds_data(latent_dim=5, T=600, num_subdyns=3, noise_std=0.01, seed=42):
    """
    Generate synthetic data from a true dLDS model.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent space (p).
    T : int
        Number of time points.
    num_subdyns : int
        Number of true dynamics operators (M).
    noise_std : float
        Standard deviation of process noise.
    seed : int
        Random seed.
        
    Returns
    -------
    x : np.ndarray
        Generated trajectory, shape (latent_dim, T).
    true_F : List[np.ndarray]
        True dynamics operators.
    true_c : np.ndarray
        True coefficients, shape (num_subdyns, T-1).
    """
    np.random.seed(seed)
    
    # Generate random dynamics operators
    true_F = []
    for _ in range(num_subdyns):
        f = np.random.randn(latent_dim, latent_dim) * 0.3
        # Normalize by spectral radius
        eig = np.max(np.abs(np.linalg.eigvals(f)))
        if eig > 0.1:
            f = f / eig
        true_F.append(f)
    
    # Generate sparse, time-varying coefficients
    true_c = np.abs(np.random.randn(num_subdyns, T - 1))
    # Make sparse (only 1-2 active at each time)
    for t in range(T - 1):
        mask = np.random.rand(num_subdyns) > 0.5
        if mask.sum() == 0:
            mask[np.random.randint(num_subdyns)] = True
        true_c[:, t] *= mask
    # Normalize
    true_c = true_c / (np.sum(true_c, axis=0, keepdims=True) + 1e-10)
    
    # Generate trajectory
    x = np.zeros((latent_dim, T))
    x[:, 0] = np.random.randn(latent_dim)
    
    for t in range(T - 1):
        F_combined = sum(true_c[m, t] * true_F[m] for m in range(num_subdyns))
        x[:, t + 1] = F_combined @ x[:, t] + noise_std * np.random.randn(latent_dim)
    
    return x, true_F, true_c


def example_single_trial():
    """Example: Fit dLDS to a single trial."""
    print("=" * 60)
    print("Example 1: Single Trial")
    print("=" * 60)
    
    # Generate data
    latent_dim = 5
    T = 600
    num_subdyns = 3
    
    x, true_F, true_c = generate_dlds_data(
        latent_dim=latent_dim, 
        T=T, 
        num_subdyns=num_subdyns,
        noise_std=0.01,
        seed=42
    )
    
    print("Data shape: %s" % (x.shape,))
    
    # Create and fit model
    model = Direct_dLDS(
        num_subdyns=num_subdyns,
        solver='spgl1',
        reg_term=0.1,
        step_f=5.0,
        max_iter=200,
        verbose=1
    )
    
    result = model.fit(x)
    
    # Print summary
    result.summary()
    
    # Reconstruct
    x_hat = model.reconstruct(x)
    
    print("\nReconstruction MSE: %.6f" % np.mean((x - x_hat) ** 2))
    
    true_vars = {'x':x, 'F': true_F, 'c': true_c}
    return model, result, true_vars, x_hat


def example_multiple_trials():
    """Example: Fit dLDS to multiple trials with shared dynamics."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Trials (Variable Length)")
    print("=" * 60)
    
    latent_dim = 4
    num_subdyns = 2
    
    # Generate multiple trials with different lengths
    trials = []
    for i, T in enumerate([100, 150, 80]):
        x, _, _ = generate_dlds_data(
            latent_dim=latent_dim,
            T=T,
            num_subdyns=num_subdyns,
            noise_std=0.02,
            seed=100 + i
        )
        trials.append(x)
        print("Trial %d shape: %s" % (i, x.shape))
    
    # Fit model
    model = Direct_dLDS(
        num_subdyns=num_subdyns,
        step_f=5.0,
        max_iter=100,
        verbose=1
    )
    
    result = model.fit(trials)
    
    # Reconstruct all trials
    reconstructions = model.reconstruct(trials)
    
    print("\nReconstruction MSE per trial:")
    for i, (x_orig, x_hat) in enumerate(zip(trials, reconstructions)):
        mse = np.mean((x_orig - x_hat) ** 2)
        print("  Trial %d: %.6f" % (i, mse))
    
    return model, result


def example_prediction():
    """Example: Predict future trajectory."""
    print("\n" + "=" * 60)
    print("Example 3: Prediction")
    print("=" * 60)
    
    # Generate training data
    latent_dim = 3
    T = 100
    num_subdyns = 2
    
    x, _, _ = generate_dlds_data(
        latent_dim=latent_dim,
        T=T,
        num_subdyns=num_subdyns,
        noise_std=0.005,
        seed=999
    )
    
    # Fit on first 80 time points
    x_train = x[:, :80]
    x_test = x[:, 80:]
    
    model = Direct_dLDS(
        num_subdyns=num_subdyns,
        step_f=10.0,
        max_iter=200,
        verbose=1
    )
    
    result = model.fit(x_train)
    
    # Predict future 20 steps from last training point
    x0 = x_train[:, -1]
    trajectory = model.predict(x0, n_steps=20)
    
    print("\nPredicted trajectory shape: %s" % (trajectory.shape,))
    print("True test trajectory shape: %s" % (x_test.shape,))
    
    # Compare prediction to true
    prediction_error = np.mean((trajectory[:, 1:] - x_test) ** 2)
    print("Prediction MSE: %.6f" % prediction_error)
    
    return model, trajectory, x_test


def example_track_evolution():
    """Example: Track F and c evolution during training."""
    print("\n" + "=" * 60)
    print("Example 4: Track Evolution")
    print("=" * 60)
    
    # Generate data
    x, _, _ = generate_dlds_data(
        latent_dim=3,
        T=50,
        num_subdyns=2,
        seed=123
    )
    
    # Fit with evolution tracking
    model = Direct_dLDS(
        num_subdyns=2,
        step_f=5.0,
        max_iter=50,
        track_evolution=True,
        verbose=1
    )
    
    result = model.fit(x)
    
    print("\nEvolution tracking:")
    print("  F_evolution length: %d" % len(result.F_evolution))
    print("  c_evolution length: %d" % len(result.c_evolution))
    
    # Analyze convergence of F
    print("\nF[0] Frobenius norm evolution (first 10 iterations):")
    for i in range(min(10, len(result.F_evolution))):
        f_norm = np.linalg.norm(result.F_evolution[i][0])
        print("  Iter %d: %.4f" % (i, f_norm))
    
    return model, result


def plot_results(x, x_hat, result, save_path=None):
    """Plot reconstruction results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original vs reconstructed (first 2 dims)
    ax = axes[0, 0]
    ax.plot(x[0, :], label='$x_1$ original', alpha=0.7)
    ax.plot(x_hat[0, :], '--', label='$x_1$ reconstructed', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('$x_1$: Original vs Reconstructed')
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(x[1, :], label='$x_2$ original', alpha=0.7)
    ax.plot(x_hat[1, :], '--', label='$x_2$ reconstructed', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('$x_2$: Original vs Reconstructed')
    ax.legend()
    
    # Error history
    ax = axes[1, 0]
    ax.plot(result.error_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('Training Error History')
    ax.set_yscale('log')
    
    # Coefficients
    ax = axes[1, 1]
    c = result.dynamic_coefficients[0]
    for m in range(c.shape[0]):
        ax.plot(c[m, :], label='$c_{%d}$' % (m + 1), alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coefficient')
    ax.set_title('Learned Coefficients $c_m(t)$')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    if save_path:
        # fig.savefig(save_path)
        # save_fig(save_path, fig, path_save)
        pass
    
    return fig


#if __name__ == '__main__':
print("\n" + "#" * 60)
print("# dLDS Examples")
print("#" * 60 + "\n")

#%%
# Run examples
model1, result, true_vars, x_hat1 = example_single_trial()
x1 = true_vars['x']

# To get the dynamics operators F (list of M matrices, each p x p):
F_list = result.F

# To get a single F matrix:
F_0 = result.F[0]  # First operator

# To get the dynamics coefficients (list of arrays, one per trial, each M x T-1):
coefficients = result.dynamic_coefficients

# To get coefficients for trial 0:
c_trial0 = result.dynamic_coefficients[0]

# To get the combined F_t = sum_m c_mt * f_m (shape: p x p x T-1):
F_combined = result.get_combined_F(trial_idx=0)

# To get the error history:
errors = result.error_history

# To get final error:
final_err = result.final_error

# To print a summary:
result.summary()



# Plot all F matrices as heatmaps:
plot_F(result.F, path_save='figures/')

# Plot single F:
plot_single_F(result.F[0], idx=0, path_save='figures/')

# Plot coefficients over time (line plot):
plot_coefficients(result.dynamic_coefficients, trial_idx=0, path_save='figures/')

# Plot coefficients as heatmap:
plot_coefficients_heatmap(result.dynamic_coefficients, trial_idx=0, path_save='figures/')

# Plot original vs reconstructed:
x_hat = model.reconstruct(data)
plot_reconstruction(data, x_hat, dims=[0, 1, 2], path_save='figures/')

# Plot trajectory:
plot_trajectory(data, dims=[0, 1, 2], path_save='figures/')

# Plot 3D trajectory:
plot_trajectory_3d(data, dims=[0, 1, 2], path_save='figures/')

# Plot error history:
plot_error_history(result.error_history, path_save='figures/')

# Plot combined F_t at time t:
F_combined = result.get_combined_F(trial_idx=0)
plot_combined_F(F_combined, time_idx=50, path_save='figures/')

# Plot F evolution during training (requires track_evolution=True):
# model = Direct_dLDS(num_subdyns=3, track_evolution=True)
plot_F_evolution(result.F_evolution, operator_idx=0, path_save='figures/')


#%%
# model2, result2 = example_multiple_trials()
# model3, traj, x_test = example_prediction()
# model4, result4 = example_track_evolution()

print("\n" + "#" * 60)
print("# All examples completed!")
print("#" * 60 + "\n")

# Optionally plot
    # plot_results(x1, x_hat1, result1)
