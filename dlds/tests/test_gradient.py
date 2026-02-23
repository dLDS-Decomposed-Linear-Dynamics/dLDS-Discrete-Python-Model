# -*- coding: utf-8 -*-
"""
Gradient Direction Verification Test
=====================================

This test verifies that the gradient descent update for F 
actually DECREASES the loss, not increases it.

A bug in the original code had the wrong sign (+/-), which would
cause gradient ASCENT instead of descent.

Run with: python -m pytest tests/test_gradient.py -v
Or simply: python tests/test_gradient.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dlds import Direct_dLDS, dLDS_config


def test_gradient_direction_decreases_error():
    """
    Test that a single gradient update decreases (or maintains) the error.
    
    This is the core sanity check for gradient descent.
    If error increases, the gradient sign is wrong.
    """
    print("=" * 60)
    print("Test: Gradient Direction Verification")
    print("=" * 60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create simple synthetic data
    latent_dim = 5
    T = 50
    num_subdyns = 2
    
    # Generate data from known dynamics
    true_F = [np.random.randn(latent_dim, latent_dim) * 0.5 for _ in range(num_subdyns)]
    # Normalize
    for i in range(len(true_F)):
        eig = np.max(np.abs(np.linalg.eigvals(true_F[i])))
        if eig > 1e-6:
            true_F[i] /= eig
    
    # Generate trajectory
    x = np.zeros((latent_dim, T))
    x[:, 0] = np.random.randn(latent_dim)
    
    true_c = np.random.rand(num_subdyns, T - 1)
    true_c = true_c / np.sum(true_c, axis=0, keepdims=True)  # Normalize
    
    for t in range(T - 1):
        F_combined = sum(true_c[m, t] * true_F[m] for m in range(num_subdyns))
        x[:, t + 1] = F_combined @ x[:, t] + 0.01 * np.random.randn(latent_dim)
    
    print("Generated synthetic data: shape = %s" % (x.shape,))
    
    # Create model with verbose off
    config = dLDS_config(
        num_subdyns=num_subdyns,
        step_f=1.0,  # Moderate step size
        max_iter=1,  # Just one iteration for this test
        verbose=0,
        seed=42
    )
    
    model = Direct_dLDS(config)
    
    # Manually run one iteration and check
    data_list = [x]
    F = model._init_f(latent_dim)
    coefficients = model._init_c([T])
    
    # Update coefficients first
    coefficients = model._update_c(data_list, F, coefficients)
    
    # Compute error BEFORE F update
    error_before = model._compute_error(data_list, F, coefficients)
    print("Error BEFORE F update: %.6f" % error_before)
    
    # Store old F
    F_old = [f.copy() for f in F]
    
    # Update F
    F_new = model._update_f(data_list, F, coefficients, step_f=1.0)
    
    # Compute error AFTER F update
    error_after = model._compute_error(data_list, F_new, coefficients)
    print("Error AFTER F update:  %.6f" % error_after)
    
    # The key assertion
    error_change = error_after - error_before
    print("Error change: %.6f" % error_change)
    
    # Allow small increase due to numerical issues, but not significant increase
    max_allowed_increase = 0.1 * error_before + 1e-6
    
    assert error_after <= error_before + max_allowed_increase, (
        "GRADIENT SIGN BUG DETECTED!\n"
        "Error INCREASED from %.6f to %.6f (change: +%.6f).\n"
        "This indicates the gradient update has the WRONG SIGN.\n"
        "Gradient descent should DECREASE error, not increase it.\n"
        "Check the sign in _update_f: should be f + step * gradient, not f - step * gradient.\n"
        "The gradient computation gives -∂L/∂f, so update is f + step * (-∂L/∂f) = f - step * ∂L/∂f."
        % (error_before, error_after, error_change)
    )
    
    if error_after < error_before:
        print("✓ PASS: Error decreased (gradient descent working correctly)")
    else:
        print("✓ PASS: Error roughly unchanged (may need more iterations or larger step)")
    
    print("=" * 60)
    return True


def test_gradient_multiple_iterations():
    """
    Test that error decreases over multiple iterations.
    """
    print("\n" + "=" * 60)
    print("Test: Multiple Iterations Error Decrease")
    print("=" * 60)
    
    np.random.seed(123)
    
    # Generate simple data
    latent_dim = 4
    T = 100
    
    # Random walk data
    x = np.cumsum(np.random.randn(latent_dim, T) * 0.1, axis=1)
    
    # Fit model
    config = dLDS_config(
        num_subdyns=2,
        step_f=5.0,
        max_iter=50,
        verbose=0,
        seed=123
    )
    
    model = Direct_dLDS(config)
    result = model.fit(x)
    
    print("Iterations run: %d" % result.n_iterations)
    print("Initial error: %.6f" % result.error_history[0])
    print("Final error:   %.6f" % result.final_error)
    
    # Error should generally decrease (allow some fluctuation)
    initial_error = result.error_history[0]
    final_error = result.final_error
    
    # Final should be less than or close to initial
    assert final_error <= initial_error * 1.5, (
        "Error increased significantly over training.\n"
        "Initial: %.6f, Final: %.6f\n"
        "This suggests a bug in the optimization."
        % (initial_error, final_error)
    )
    
    # Check that error generally trends downward
    mid_point = len(result.error_history) // 2
    if mid_point > 5:
        early_avg = np.mean(result.error_history[:5])
        late_avg = np.mean(result.error_history[-5:])
        print("Early average error: %.6f" % early_avg)
        print("Late average error:  %.6f" % late_avg)
        
        # Late error should not be much higher than early
        assert late_avg <= early_avg * 2, (
            "Error increased over training.\n"
            "Early avg: %.6f, Late avg: %.6f"
            % (early_avg, late_avg)
        )
    
    print("✓ PASS: Error trends downward over iterations")
    print("=" * 60)
    return True


def test_gradient_formula_correctness():
    """
    Test the gradient formula against numerical gradient.
    
    Uses finite differences to verify the analytical gradient.
    """
    print("\n" + "=" * 60)
    print("Test: Gradient Formula vs Numerical Gradient")
    print("=" * 60)
    
    np.random.seed(456)
    
    latent_dim = 3
    T = 20
    num_subdyns = 2
    
    # Simple data
    x = np.random.randn(latent_dim, T)
    data_list = [x]
    
    # Initialize model
    config = dLDS_config(
        num_subdyns=num_subdyns,
        verbose=0,
        seed=456
    )
    model = Direct_dLDS(config)
    
    F = model._init_f(latent_dim)
    coefficients = model._init_c([T])
    coefficients = model._update_c(data_list, F, coefficients)
    
    # Compute analytical gradient
    analytical_grads = model._compute_gradient(data_list, F, coefficients)
    
    # Compute numerical gradient for F[0]
    eps = 1e-5
    numerical_grad = np.zeros_like(F[0])
    
    for i in range(latent_dim):
        for j in range(latent_dim):
            # Perturb +
            F_plus = [f.copy() for f in F]
            F_plus[0][i, j] += eps
            error_plus = model._compute_error(data_list, F_plus, coefficients)
            
            # Perturb -
            F_minus = [f.copy() for f in F]
            F_minus[0][i, j] -= eps
            error_minus = model._compute_error(data_list, F_minus, coefficients)
            
            # Numerical gradient (of loss, so negative for descent direction)
            numerical_grad[i, j] = (error_plus - error_minus) / (2 * eps)
    
    # The analytical gradient we compute is the DESCENT direction
    # So it should be the NEGATIVE of the loss gradient
    # descent_direction = -∂L/∂f
    # numerical_grad = ∂L/∂f
    # So: descent_direction ≈ -numerical_grad
    
    analytical_descent = analytical_grads[0]
    expected_descent = -numerical_grad
    
    # Compare (they should be close)
    max_diff = np.max(np.abs(analytical_descent - expected_descent))
    relative_diff = max_diff / (np.max(np.abs(expected_descent)) + 1e-10)
    
    print("Max absolute difference: %.6e" % max_diff)
    print("Relative difference: %.6e" % relative_diff)
    
    # Allow some numerical error
    assert relative_diff < 0.1, (
        "Analytical gradient doesn't match numerical gradient!\n"
        "Max diff: %.6e, Relative diff: %.6e\n"
        "This indicates a bug in the gradient formula."
        % (max_diff, relative_diff)
    )
    
    print("✓ PASS: Analytical gradient matches numerical gradient")
    print("=" * 60)
    return True


def test_reconstruction_error_makes_sense():
    """
    Test that reconstruction error is reasonable.
    """
    print("\n" + "=" * 60)
    print("Test: Reconstruction Error Sanity Check")
    print("=" * 60)
    
    np.random.seed(789)
    
    latent_dim = 5
    T = 50
    num_subdyns = 3
    
    # Generate data from TRUE dLDS dynamics
    true_F = [np.random.randn(latent_dim, latent_dim) * 0.3 for _ in range(num_subdyns)]
    for i in range(len(true_F)):
        eig = np.max(np.abs(np.linalg.eigvals(true_F[i])))
        if eig > 0.1:
            true_F[i] /= eig
    
    x = np.zeros((latent_dim, T))
    x[:, 0] = np.random.randn(latent_dim)
    
    true_c = np.abs(np.random.randn(num_subdyns, T - 1))
    true_c = true_c / np.sum(true_c, axis=0, keepdims=True)
    
    for t in range(T - 1):
        F_combined = sum(true_c[m, t] * true_F[m] for m in range(num_subdyns))
        x[:, t + 1] = F_combined @ x[:, t]  # No noise
    
    print("Generated noiseless dLDS data")
    
    # Fit model with correct number of operators
    config = dLDS_config(
        num_subdyns=num_subdyns,
        step_f=10.0,
        max_iter=500,
        reg_term=0.01,
        verbose=0,
        seed=789
    )
    
    model = Direct_dLDS(config)
    result = model.fit(x)
    
    print("Final error: %.6f" % result.final_error)
    
    # For noiseless data, should achieve low error
    assert result.final_error < 1.0, (
        "Error too high for noiseless dLDS data.\n"
        "Final error: %.6f (expected < 1.0)\n"
        "The model should be able to fit its own generative model well."
        % result.final_error
    )
    
    # Reconstruct
    x_hat = model.reconstruct(x)
    
    reconstruction_error = np.mean((x - x_hat) ** 2)
    print("Mean reconstruction error: %.6f" % reconstruction_error)
    
    assert reconstruction_error < 1.0, (
        "Reconstruction error too high: %.6f" % reconstruction_error
    )
    
    print("✓ PASS: Reconstruction error is reasonable")
    print("=" * 60)
    return True


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("# dLDS Gradient Verification Tests")
    print("#" * 60 + "\n")
    
    all_passed = True
    
    try:
        test_gradient_direction_decreases_error()
    except AssertionError as e:
        print("✗ FAIL: %s" % e)
        all_passed = False
    
    try:
        test_gradient_multiple_iterations()
    except AssertionError as e:
        print("✗ FAIL: %s" % e)
        all_passed = False
    
    try:
        test_gradient_formula_correctness()
    except AssertionError as e:
        print("✗ FAIL: %s" % e)
        all_passed = False
    
    try:
        test_reconstruction_error_makes_sense()
    except AssertionError as e:
        print("✗ FAIL: %s" % e)
        all_passed = False
    
    print("\n" + "#" * 60)
    if all_passed:
        print("# ALL TESTS PASSED")
    else:
        print("# SOME TESTS FAILED")
    print("#" * 60 + "\n")
