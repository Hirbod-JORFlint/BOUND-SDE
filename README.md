# BOUND-SDE  
## A JAX-Native Framework for Constrained Trait Evolution on Phylogenetic Trees

BOUND-SDE is a high-performance, fully differentiable framework for modeling constrained stochastic trait evolution on phylogenetic trees. It implements Stochastic Differential Equations (SDEs) with reflective boundaries and Riemannian manifold constraints, enabling rigorous likelihood-based inference for bounded or geometrically structured traits.

The system is designed for computational biologists, statisticians, and machine learning researchers who require scalable, geometry-aware phylogenetic comparative methods (PCM) implemented in modern differentiable computing frameworks.

---

## Core Idea

Trait evolution along each branch of a phylogenetic tree is modeled as an Itô diffusion:

\[
dX_t = \mu(X_t)\,dt + \sigma(X_t)\,dW_t
\]

with either:

- **Reflecting boundary conditions** on an interval \([L, U]\), or  
- **Riemannian manifold constraints**, specifically:
  - \( S^1 \) (the circle)
  - \( \Delta^d \) (the probability simplex)

On a Riemannian manifold \( (\mathcal{M}, g) \), the infinitesimal generator is:

\[
\mathcal{L} f = \frac{1}{2}\Delta_g f + \langle b, \nabla f \rangle_g
\]

where:
- \( \Delta_g \) is the Laplace–Beltrami operator,
- \( \nabla \) is the Riemannian gradient,
- \( b \) is the drift vector field.

Transition densities are approximated using spectral expansions:

\[
p_t(x, y) \approx \sum_{k=0}^{K} e^{-\lambda_k t}\,\phi_k(x)\phi_k(y)
\]

where \( (\lambda_k, \phi_k) \) are eigenpairs of the negative generator subject to boundary or manifold constraints.

These approximations feed into a novel **Boundary-Propagating Pruning (BPP)** algorithm for likelihood computation on trees.

---

## Key Features

### Reflective Boundaries
Implements Neumann-type (zero-flux) reflecting conditions on bounded intervals \([L, U]\).

### Manifold Support
Supports:
- Circular traits (angles, periodic phenotypes)
- Compositional data on simplices

### Spectral Transition Approximations
Efficient eigen-decomposition–based transition approximations compatible with JAX autodiff.

### Differentiable Likelihood
Fully compatible with JAX automatic differentiation for gradient-based parameter estimation.

### Vectorized Tree Traversal
Uses `jax.vmap` and `jax.lax.scan` to ensure:
- \( O(N) \) tree traversal complexity
- Efficient batching over parameter sets

### R Interface
Includes a reticulate-ready bridge for seamless integration with R-based PCM workflows.

---

## Repository Structure
kernels.py # SDE generators and spectral basis functions
manifolds.py # Geometry definitions (S1, Δd, metric tensors)
tree_ops.py # JAX-compatible tree traversal utilities
spectral_solver.py # Eigenvalue/eigenvector computation for transitions
pruning.py # Boundary-Propagating Pruning (BPP) algorithm
likelihood.py # Log-likelihood wrappers and objective functions
optimizers.py # Gradient descent and L-BFGS routines
simulations.py # Forward SDE simulators on trees
bridge_r.py # R interface (reticulate compatible)
main.py # Pipeline execution and validation tests
tests/ # Unit tests for all modules
comprehensive_demo.py # Comprehensive demo script
tree_actions_demo.py # Tree operations demo
requirements.txt # Python dependencies

---

## Quick Start

1. **Install dependencies**:
 ```bash
 pip install -r requirements.txt
 ```

2. **Run tests** to verify installation:
   ```bash
   python -m pytest
   ```
   or
    ```bash
    bash run_tests.sh
    ```

3. **Run the comprehensive demo**:
   ```bash
   python comprehensive_demo.py
   ```

4. **Generate an example dataset**:
   ```bash
   python scripts/generate_dataset.py --config configs/example_config.json --output data/example_dataset.npz
   ```

---

## Usage

### Command-Line Interface

Use `main.py` for full pipelines:

- **Simulate traits**:
  ```bash
  python main.py simulate --config config.json --output traits.npz
  ```

- **Fit model parameters**:
  ```bash
  python main.py fit --data traits.npz --config config.json --output params.npz
  ```

- **Run tests**:
  ```bash
  python main.py test
  ```

 - **Generate example data**:
   ```bash
   python scripts/generate_dataset.py
   ```

### Python API

Import modules for custom workflows:

```python
from tree_ops import build_tree_arrays
from manifolds import create_circle_manifold
from simulations import simulate_tree_traits
from likelihood import likelihood_objective

# Build tree
tree = build_tree_arrays(parent_array, child_array, branch_lengths)

# Define manifold
manifold = create_circle_manifold()

# Simulate or compute likelihood
# ... (see comprehensive_demo.py for examples)
```

### R Integration

Use `reticulate` to call from R:

```r
library(reticulate)
source_python("bridge_r.py")

# Convert R tree to JAX
tree_data <- r_tree_to_arrays(edge_matrix, edge_lengths)

# Compute likelihood
loglik <- r_compute_likelihood(params, tree_data$parents, tree_data$branch_lengths, ...)
```

---

## Testing

Run the full test suite:
```bash
python -m pytest
```

Or test specific modules:
```bash
python -m pytest tests/test_tree_ops.py
```

Use `bash run_tests.sh` for the same suite with the provided helper.

---

## Example Configuration

Sample configuration lives in `configs/example_config.json`. It encodes the tree topology, manifold type,
and solver settings so you can reproduce dataset generation and fitting without building your own config.

---

## Usage Demonstration

Use `scripts/demo_usage.py` to generate a dataset, fit the model, and watch console logs for each step. The script
prints the configuration path, dataset location, and subprocess return codes so you can trace the workflow.

```bash
python scripts/demo_usage.py --output-dir demo --steps 30
```

This produces `demo/demo_dataset.npz` and `demo/demo_params.npz` (plus any logs you enable). Inspect these files with
`numpy.load` or feed them back into the fitting CLI.

---

## Contributing

Contributions welcome! Please:
- Add tests for new features
- Follow JAX best practices for vectorization
- Update documentation

---

## License

[Specify license, e.g., MIT]

---

## Citation

If you use BOUND-SDE in your research, please cite:

[Add citation details]
