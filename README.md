# dLDS: Decomposed Linear Dynamical Systems

A Python package for learning **Decomposed Linear Dynamical Systems** from time series data.

📄 **Paper:** <a href="https://jmlr.org/papers/volume25/23-0777/23-0777.pdf" target="_blank">Decomposed Linear Dynamical Systems (dLDS) for learning the latent components of neural dynamics</a>

📧 **Contact:** nmudrik1 [at] jhu.edu 

**Citation:**
If you use this code, please cite:

```bibtex
@article{mudrik2024decomposed,
  title={Decomposed linear dynamical systems (dlds) for learning the latent components of neural dynamics},
  author={Mudrik, Noga and Chen, Yenho and Yezerets, Eva and Rozell, Christopher J and Charles, Adam S},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={59},
  pages={1--44},
  year={2024}
}
 ``` 

## What is dLDS?

dLDS decomposes complex dynamics into a weighted combination of simple basis operators:

$$\mathbf{y}_t = \mathbf{D} \mathbf{x}_t$$

$$
\mathbf{x}_{t+1} = \left( \sum_{m} c_{m,t} \cdot f_m \right) \mathbf{x}_t
$$


Where:
- $\mathbf{y}_t \in \mathbb{R}^{N}$ — observed neural activity ($N$ = number of neurons)
- $\mathbf{Y} \in \mathbb{R}^{N \times T}$ — full observation matrix ($T$ = number of time steps)
- $\mathbf{x}_t \in \mathbb{R}^{p}$ — latent state at time $t$
- $\mathbf{X} \in \mathbb{R}^{p \times T}$ — full latent state matrix
- $D \in \mathbb{R}^{N \times p}$ — observation/mixing matrix
- $F_m \in \mathbb{R}^{p \times p}$ — learned sub-dynamics  ($M$ matrices)
- $c_{m,t} \in \mathbb{R}$ — the coefficient of subdynamic $m$ at time $t$

**Use cases:** Neural data analysis, time series decomposition, dynamical systems discovery.

---

## Installation

```bash
git clone https://github.com/NogaMudrik/dLDS-Discrete-Python-Model
cd dlds
```

**Dependencies:** numpy, scipy, matplotlib, tqdm, pylops (optional but recommended)

---

## Quick Start

### Option 1: Direct observation (you observe the state directly)

```python
from dlds import Direct_dLDS

# Load your data: shape (n_dims, T) or list of arrays
data = your_data  # e.g., shape (10, 500)

# Create and fit model
model = Direct_dLDS(num_subdyns=3)
result = model.fit(data)

# Get results
F_operators = result.F                    # List of learned F matrices
coefficients = result.dynamic_coefficients # Learned c(t)

# Reconstruct
x_hat = model.reconstruct(data)
```

### Option 2: Indirect observation (you observe through a mixing matrix D)

```python
from dlds import dLDS_with_latents

# Load observations: shape (obs_dim, T)
y = your_observations  # e.g., shape (100, 500)

# Create and fit model
model = dLDS_with_latents(num_subdyns=3, latent_dim=10)
result = model.fit(y)

# Get results
F_operators = result.F       # Dynamics operators
D_matrix = result.D          # Observation matrix
x_latents = result.x         # Inferred latent states
coefficients = result.dynamic_coefficients
```

### Option 3: Try the example notebook

See **`lorenz_example.ipynb`** for a complete walkthrough using Lorenz attractor data.

---

## Data Format

| Format | Shape | Description |
|--------|-------|-------------|
| Single trial | `(n_dims, T)` | 2D numpy array |
| Multiple trials | `[(n_dims, T1), (n_dims, T2), ...]` | List of 2D arrays |

**Notes:**
- Rows = dimensions/features
- Columns = time points
- Variable trial lengths are supported
- Data is automatically centered if `center_data=True`

---

## Key Parameters

### Essential

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_subdyns` | 3 | Number of dynamics operators (M). More = more complex dynamics |
| `solver` | `'spgl1'` | Coefficient solver. Options: `'spgl1'`, `'lasso'`, `'inv'`, `'nnls'` |
| `reg_term` | 0.1 | Sparsity regularization. Higher = sparser coefficients |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 5000 | Maximum training iterations |
| `step_f` | 10.0 | Learning rate for F update |
| `gd_decay` | 1.0 | Learning rate decay (set < 1 to decay, e.g., 0.99) |
| `smooth_term` | 0.0 | Smoothness penalty on coefficients (higher = smoother c) |

### Convergence

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_error_relative` | 0.01 | Convergence threshold (fraction of data variance) |
| `use_relative_error` | True | Use relative error threshold |

### Tips

- **Spiky coefficients?** Increase `smooth_term` (try 0.1 - 1.0)
- **Too sparse?** Increase `reg_term` for spgl1 (it's a bound, not penalty)
- **Not converging?** Increase `max_iter`, decrease `step_f`
- **Overfitting?** Reduce `num_subdyns`
- **Slow?** Use `solver='inv'` (faster but no sparsity)

---

## Visualization

```python
from dlds import plot_F, plot_coefficients, plot_reconstruction, plot_error_history

# Plot learned dynamics operators
plot_F(result.F, path_save='figures/', to_save=True)

# Plot coefficients over time
plot_coefficients(result.dynamic_coefficients, path_save='figures/', to_save=True)

# Plot original vs reconstructed
x_hat = model.reconstruct(data)
plot_reconstruction(data, x_hat, dims=[0, 1, 2], path_save='figures/', to_save=True)

# Plot training error
plot_error_history(result.error_history, path_save='figures/', to_save=True)

# Scatter colored by dominant dynamics
plot_scatter_by_dominant_subdynamics(data, result.dynamic_coefficients)
```

---

## Accessing Results

```python
result = model.fit(data)

# Dynamics operators
result.F                      # List of M matrices, each (p, p)
result.F[0]                   # First operator

# Coefficients
result.dynamic_coefficients   # List of arrays, each (M, T-1)
result.dynamic_coefficients[0] # Coefficients for trial 0

# Combined dynamics: F_t = Σ c_mt * F_m
F_combined = result.get_combined_F(trial_idx=0)  # Shape (p, p, T-1)

# Training info
result.error_history          # Error at each iteration
result.final_error            # Final reconstruction error
result.converged              # True/False
result.n_iterations           # Number of iterations run

# Summary
result.summary()              # Print summary
```

---

## File Structure

```
dlds/
├── __init__.py          # Package exports
├── config.py            # dLDS_config dataclass (all parameters)
├── results.py           # dLDS_result dataclass (output container)
├── direct.py            # Direct_dLDS class (D=I case)
├── dlds_with_latents.py # dLDS_with_latents class (D≠I case)
├── solvers.py           # Coefficient solvers (spgl1, lasso, etc.)
├── utils.py             # Initialization, normalization utilities
└── vis.py               # Visualization functions
```

---

## Example: Lorenz Attractor

```python
from dlds import Direct_dLDS
from synthetic import generate_dlds_synthetic_data

# Generate Lorenz data
x, _, _ = generate_dlds_synthetic_data(T=1000, structured='lorenz')

# Fit model
model = Direct_dLDS(
    num_subdyns=5,
    solver='spgl1',
    reg_term=1.0,
    smooth_term=0.5,
    step_f=5.0,
    max_iter=500
)
result = model.fit(x)

# Visualize
from dlds import plot_scatter_by_dominant_subdynamics
plot_scatter_by_dominant_subdynamics(x, result.dynamic_coefficients)
```

See **`lorenz_example.ipynb`** for the full example with visualizations.

---

## Q & A 
**Q: How to make coefficients more sparse?**
- Avoid `solver='inv'` (no sparsity at all)
- Use `solver='spgl1'` (recommended)
- For spgl1: smaller `reg_term` = more sparse
- For lasso: larger `reg_term` = more sparse

**Q: How to make coefficients smoother over time?**
- Increase `smooth_term` (try 0.1, 0.5, 1.0)
- Default is 0 (no smoothness)

**Q: Model not converging?**
- Increase `max_iter` (e.g., 5000 or 10000)
- Decrease `step_f` (e.g., 1.0 instead of 10.0)
- Enable decay: `gd_decay=0.99`

**Q: Error too high?**
- Increase `num_subdyns` (more operators = better fit)
- Decrease `reg_term` (less sparsity = better fit)
- Run more iterations

**Q: Training too slow?**
- Use `solver='inv'` (fastest, but no sparsity)
- Reduce `max_iter`
- Reduce `num_subdyns`

**Q: How many `num_subdyns` to use?**
- Start with 3-5
- More = better reconstruction but slower and risk of overfitting
- Less = faster but may miss complex dynamics

**Q: Which solver to use?**
| Solver | Speed | Sparsity | When to use |
|--------|-------|----------|-------------|
| `spgl1` | Medium | Yes | Default, recommended |
| `lasso` | Medium | Yes | Alternative to spgl1 |
| `inv` | Fast | No | Quick tests, no sparsity needed |
| `nnls` | Medium | No | When coefficients must be positive |

**Q: What does `reg_term` do?**
- Controls sparsity of coefficients
- For `spgl1`: it's a bound (higher = less sparse)
- For `lasso`: it's a penalty (higher = more sparse)
- Start with 0.1, adjust based on results

**Q: Data not fitting well?**
- Check data format: should be `(n_dims, T)` not `(T, n_dims)`
- Try `center_data=True`
- Increase `num_subdyns`
- Check for NaNs in your data

**Q: How to check if model is good?**
```python
# Check final error
print(result.final_error)

# Compare to data variance
print("Error / Variance:", result.final_error / np.var(data))

# Visual check
plot_reconstruction(data, model.reconstruct(data))
```

**Q: Can I use variable length trials?**
- Yes! Pass as list: `[trial1, trial2, trial3]`
- Each trial can have different T
- All trials must have same `n_dims`

**Q: What if I only observe through sensors (not the state directly)?**
- Use `dLDS_with_latents` instead of `Direct_dLDS`
- Set `latent_dim` to your desired latent dimension
- Model learns both dynamics F and observation matrix D

---


