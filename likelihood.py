# ============================================================
# Likelihood Interface
# ============================================================

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from spectral_solver import compute_spectral_decomposition
from pruning import compute_tree_loglik

# ============================================================
# Model Parameter Structure
# ============================================================

class ModelParams(NamedTuple):
    """
    Container for BOUND-SDE model parameters.

    Attributes
    ----------
    drift : jnp.ndarray
        Drift parameters of the SDE.

        Shape
        -----
        (P_d,)

    diffusion : jnp.ndarray
        Diffusion parameters.

        Shape
        -----
        (P_s,)

    boundary_params : jnp.ndarray
        Parameters controlling reflective boundaries
        or manifold geometry.

        Shape
        -----
        (P_b,)

    root_prior : jnp.ndarray
        Prior distribution at root.

        Shape
        -----
        (M,)
    """

    drift: jnp.ndarray
    diffusion: jnp.ndarray
    boundary_params: jnp.ndarray
    root_prior: jnp.ndarray

# ============================================================
# Tree Data Structure
# ============================================================

class TreeData(NamedTuple):
    """
    Static tree information used by pruning.

    Attributes
    ----------
    postorder_nodes : jnp.ndarray
        Shape
        -----
        (N,)

    parent_index : jnp.ndarray
        Shape
        -----
        (N,)

    parent_branch : jnp.ndarray
        Shape
        -----
        (N,)

    branch_lengths : jnp.ndarray
        Shape
        -----
        (B,)

    root_index : int
    """

    postorder_nodes: jnp.ndarray
    parent_index: jnp.ndarray
    parent_branch: jnp.ndarray
    branch_lengths: jnp.ndarray
    root_index: int

# ============================================================
# Tip Initialization
# ============================================================

def initialize_tip_likelihood(
    tip_traits: jnp.ndarray,
    basis_functions
) -> jnp.ndarray:
    """
    Initialize likelihood vectors at tips.

    Mathematical definition
    -----------------------

    For trait observation x_i at tip i

        L_i(k) = log φ_k(x_i)

    where φ_k are spectral basis functions.

    Parameters
    ----------
    tip_traits : jnp.ndarray

        Shape
        -----
        (T,)

    basis_functions : callable

        Maps

            x → φ(x)

        returning spectral basis vector.

    Returns
    -------
    tip_loglik : jnp.ndarray

        Shape
        -----
        (T, M)
    """

    def compute_row(x):

        phi = basis_functions(x)

        return jnp.log(jnp.clip(phi, 1e-30))

    return jax.vmap(compute_row)(tip_traits)

# ============================================================
# Differentiable Likelihood Objective
# ============================================================

def likelihood_objective(
    params: ModelParams,
    tree: TreeData,
    node_loglik: jnp.ndarray,
    manifold,
    spectral_dim: int
) -> jnp.ndarray:
    """
    Compute log-likelihood under the BOUND-SDE model.

    Mathematical definition
    -----------------------

    Given parameters θ defining generator

        L_θ

    compute spectral decomposition

        (λ_k, φ_k)

    evaluate transition densities

        P_t(x,y)

    and compute tree likelihood

        log L_tree(θ)

    Parameters
    ----------
    params : ModelParams

    tree : TreeData

    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)

        Initial node likelihood vectors
        with tips initialized.

    manifold : object

        Manifold geometry instance
        (S1 or Δ_d).

    spectral_dim : int

        Number of spectral basis functions.

    Returns
    -------
    loglik : float
    """

    eigenvalues, eigenvectors, inv_eigenvectors = \
        compute_spectral_decomposition(
            params,
            manifold,
            spectral_dim
        )

    loglik = compute_tree_loglik(
        tree.postorder_nodes,
        node_loglik,
        tree.parent_index,
        tree.parent_branch,
        tree.branch_lengths,
        eigenvalues,
        eigenvectors,
        inv_eigenvectors,
        tree.root_index,
        params.root_prior
    )

    return loglik

# ============================================================
# Negative Log-Likelihood
# ============================================================

def neg_loglik(
    params: ModelParams,
    tree: TreeData,
    node_loglik: jnp.ndarray,
    manifold,
    spectral_dim: int
):
    """
    Negative log-likelihood objective.

    Used for gradient-based optimization.

    Returns
    -------
    value : float
    """

    return -likelihood_objective(
        params,
        tree,
        node_loglik,
        manifold,
        spectral_dim
    )

# ============================================================
# Parameter Shape Metadata
# ============================================================

class ParamShape:
    """
    Stores shapes of parameter components.

    Attributes
    ----------
    drift_dim : int
    diffusion_dim : int
    boundary_dim : int
    prior_dim : int
    """

    def __init__(self, drift_dim, diffusion_dim, boundary_dim, prior_dim):
        self.drift_dim = drift_dim
        self.diffusion_dim = diffusion_dim
        self.boundary_dim = boundary_dim
        self.prior_dim = prior_dim

    @property
    def total_dim(self):
        return (
            self.drift_dim
            + self.diffusion_dim
            + self.boundary_dim
            + self.prior_dim
        )

# ============================================================
# Flatten Model Parameters
# ============================================================

def flatten_params(params) -> jnp.ndarray:
    """
    Convert ModelParams into flat vector.

    Mathematical definition
    -----------------------

    θ = (μ, σ, b, π)

    vectorized as

        v = [μ, σ, b, π]

    Parameters
    ----------
    params : ModelParams

    Returns
    -------
    vec : jnp.ndarray

        Shape
        -----
        (D,)
    """

    return jnp.concatenate(
        [
            params.drift,
            params.diffusion,
            params.boundary_params,
            params.root_prior,
        ]
    )

# ============================================================
# Reconstruct Structured Parameters
# ============================================================

def unflatten_params(
    vec: jnp.ndarray,
    shape: ParamShape
):
    """
    Convert flat vector back to ModelParams.

    Parameters
    ----------
    vec : jnp.ndarray
        Shape
        -----
        (D,)

    shape : ParamShape

    Returns
    -------
    params : ModelParams
    """

    i = 0

    drift = vec[i : i + shape.drift_dim]
    i += shape.drift_dim

    diffusion = vec[i : i + shape.diffusion_dim]
    i += shape.diffusion_dim

    boundary = vec[i : i + shape.boundary_dim]
    i += shape.boundary_dim

    prior = vec[i : i + shape.prior_dim]

    from likelihood import ModelParams

    return ModelParams(
        drift=drift,
        diffusion=diffusion,
        boundary_params=boundary,
        root_prior=prior,
    )

# ============================================================
# Parameter Transformations
# ============================================================

def transform_params(
    vec: jnp.ndarray,
    shape: ParamShape
):
    """
    Apply unconstrained → constrained parameter transforms.

    Mathematical definition
    -----------------------

    Diffusion parameters:

        σ = exp(η)

    Root prior (softmax):

        π_k = exp(η_k) / Σ_j exp(η_j)

    Parameters
    ----------
    vec : jnp.ndarray
        Unconstrained parameter vector.

    shape : ParamShape

    Returns
    -------
    params : ModelParams
    """

    params = unflatten_params(vec, shape)

    diffusion = jnp.exp(params.diffusion)

    prior_logits = params.root_prior
    prior = jax.nn.softmax(prior_logits)

    from likelihood import ModelParams

    return ModelParams(
        drift=params.drift,
        diffusion=diffusion,
        boundary_params=params.boundary_params,
        root_prior=prior,
    )

# ============================================================
# Inverse Transform
# ============================================================

def inverse_transform_params(params) -> jnp.ndarray:
    """
    Map constrained parameters to unconstrained space.

    Mathematical definition
    -----------------------

    diffusion:

        η = log(σ)

    prior:

        η_k = log(π_k)

    Returns
    -------
    vec : jnp.ndarray
    """

    diffusion = jnp.log(params.diffusion)

    prior_logits = jnp.log(params.root_prior)

    unconstrained = jnp.concatenate(
        [
            params.drift,
            diffusion,
            params.boundary_params,
            prior_logits,
        ]
    )

    return unconstrained

# ============================================================
# Vectorized Likelihood Wrapper
# ============================================================

def vectorized_neg_loglik(
    vec: jnp.ndarray,
    shape: ParamShape,
    tree,
    node_loglik,
    manifold,
    spectral_dim
):
    """
    Negative log-likelihood with vector parameters.

    Parameters
    ----------
    vec : jnp.ndarray
        Shape
        -----
        (D,)

    Returns
    -------
    value : float
    """

    params = transform_params(vec, shape)

    from likelihood import neg_loglik

    return neg_loglik(
        params,
        tree,
        node_loglik,
        manifold,
        spectral_dim
    )

# ============================================================
# Gradient for Optimizers
# ============================================================

grad_vectorized_neg_loglik = jax.grad(vectorized_neg_loglik)

# ============================================================
# Batched Tree Likelihood
# ============================================================

def likelihood_batch(
    params,
    trees,
    node_loglik_batch,
    manifold,
    spectral_dim
):
    """
    Evaluate log-likelihood for multiple phylogenetic trees.

    Mathematical definition
    -----------------------

    For datasets D_k

        log L_k(θ)

    the batch likelihood vector is

        L(θ) =
        [ log L_1(θ), ..., log L_K(θ) ]

    Parameters
    ----------
    params : ModelParams

    trees : TreeData
        Batched tree structure.

    node_loglik_batch : jnp.ndarray

        Shape
        -----
        (K, N, M)

        Initial node likelihood vectors
        for each dataset.

    manifold : object

    spectral_dim : int

    Returns
    -------
    loglik_batch : jnp.ndarray

        Shape
        -----
        (K,)
    """

    from likelihood import likelihood_objective

    def single_likelihood(node_loglik):

        return likelihood_objective(
            params,
            trees,
            node_loglik,
            manifold,
            spectral_dim
        )

    return jax.vmap(single_likelihood)(node_loglik_batch)

# ============================================================
# Summed Batch Likelihood
# ============================================================

def likelihood_total(
    params,
    trees,
    node_loglik_batch,
    manifold,
    spectral_dim
):
    """
    Compute total log-likelihood across datasets.

    Mathematical definition
    -----------------------

        log L_total(θ)
        =
        Σ_k log L_k(θ)

    Returns
    -------
    loglik : float
    """

    logliks = likelihood_batch(
        params,
        trees,
        node_loglik_batch,
        manifold,
        spectral_dim
    )

    return jnp.sum(logliks)

# ============================================================
# Batched Negative Log-Likelihood
# ============================================================

def neg_loglik_batch(
    params,
    trees,
    node_loglik_batch,
    manifold,
    spectral_dim
):
    """
    Negative total log-likelihood for batch.

    Returns
    -------
    value : float
    """

    return -likelihood_total(
        params,
        trees,
        node_loglik_batch,
        manifold,
        spectral_dim
    )

# ============================================================
# Vectorized Batch Objective
# ============================================================

def vectorized_neg_loglik_batch(
    vec,
    shape,
    trees,
    node_loglik_batch,
    manifold,
    spectral_dim
):
    """
    Negative log-likelihood for batched datasets
    using flattened parameter vector.

    Parameters
    ----------
    vec : jnp.ndarray

        Shape
        -----
        (D,)

    Returns
    -------
    value : float
    """

    from likelihood import transform_params

    params = transform_params(vec, shape)

    return neg_loglik_batch(
        params,
        trees,
        node_loglik_batch,
        manifold,
        spectral_dim
    )

# ============================================================
# Gradient for Batched Optimization
# ============================================================

grad_vectorized_neg_loglik_batch = jax.grad(
    vectorized_neg_loglik_batch
)
