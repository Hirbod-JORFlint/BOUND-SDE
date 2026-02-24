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


---

## Installation

Requirements:

- Python ≥ 3.9  
- JAX (CPU or GPU backend)  
- NumPy, SciPy  
- Optional: optax, matplotlib  
- Optional (R side): reticulate  

Example installation:

```bash
pip install -r requirements.txt