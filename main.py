# ============================================================
# BOUND-SDE Main Entry Module
# ============================================================

import jax
import jax.numpy as jnp
import numpy as np

from typing import NamedTuple

from likelihood import (
    vectorized_neg_loglik,
    grad_vectorized_neg_loglik
)

from optimizers import (
    run_adam,
    run_lbfgs
)

from simulations import (
    simulate_tree_traits
)

# ============================================================
# Model Configuration
# ============================================================

class ModelConfig(NamedTuple):
    """
    Configuration for BOUND-SDE model.

    Attributes
    ----------
    spectral_dim : int
        Number of spectral basis functions.

    dt : float
        Integration time step for simulations.

    manifold : object
        Manifold instance (S¹ or Δᵈ).

    optimizer : str
        Optimization method ("adam" or "lbfgs").
    """

    spectral_dim: int
    dt: float
    manifold: object
    optimizer: str

# ============================================================
# Likelihood Objective Builder
# ============================================================

def build_objective(
    shape,
    parents,
    branch_lengths,
    node_loglik,
    manifold,
    spectral_dim
):
    """
    Construct likelihood objective and gradient.

    Parameters
    ----------
    shape : tuple
        Parameter tensor shape.

    parents : jnp.ndarray
        Shape
        -----
        (N,)

    branch_lengths : jnp.ndarray
        Shape
        -----
        (N,)

    node_loglik : jnp.ndarray
        Shape
        -----
        (N, M)

    spectral_dim : int

    Returns
    -------
    grad_fn : callable
        Function returning (loss, gradient).
    """

    def objective(params):

        loss = vectorized_neg_loglik(
            params,
            shape,
            parents,
            branch_lengths,
            node_loglik,
            manifold,
            spectral_dim
        )

        grad = grad_vectorized_neg_loglik(
            params,
            shape,
            parents,
            branch_lengths,
            node_loglik,
            manifold,
            spectral_dim
        )

        return loss, grad

    return objective

# ============================================================
# Parameter Optimization
# ============================================================

def fit_model(
    init_params,
    shape,
    parents,
    branch_lengths,
    node_loglik,
    config: ModelConfig,
    steps=200
):
    """
    Estimate model parameters by likelihood maximization.

    Parameters
    ----------
    init_params : jnp.ndarray
        Shape
        -----
        (D,)

    shape : tuple

    parents : jnp.ndarray

    branch_lengths : jnp.ndarray

    node_loglik : jnp.ndarray

    config : ModelConfig

    steps : int

    Returns
    -------
    params : jnp.ndarray
        Optimized parameter vector.

    loss_history : jnp.ndarray
    """

    grad_fn = build_objective(
        shape,
        parents,
        branch_lengths,
        node_loglik,
        config.manifold,
        config.spectral_dim
    )

    if config.optimizer == "adam":

        params, losses = run_adam(
            init_params,
            grad_fn,
            num_steps=steps
        )

    elif config.optimizer == "lbfgs":

        params, losses = run_lbfgs(
            init_params,
            grad_fn,
            num_steps=steps
        )

    else:

        raise ValueError("Unknown optimizer")

    return params, losses

# ============================================================
# Trait Simulation Pipeline
# ============================================================

def simulate_dataset(
    seed,
    root_state,
    parents,
    branch_lengths,
    topo_order,
    drift_fn,
    diffusion_fn,
    config: ModelConfig
):
    """
    Generate synthetic trait data.

    Parameters
    ----------
    seed : int

    root_state : jnp.ndarray
        Shape
        -----
        (d,)

    parents : jnp.ndarray

    branch_lengths : jnp.ndarray

    topo_order : jnp.ndarray

    Returns
    -------
    traits : jnp.ndarray
        Shape
        -----
        (N, d)
    """

    key = jax.random.PRNGKey(seed)

    traits = simulate_tree_traits(
        key,
        root_state,
        parents,
        branch_lengths,
        topo_order,
        drift_fn,
        diffusion_fn,
        config.manifold,
        config.dt
    )

    return traits

# ============================================================
# Full Training Pipeline
# ============================================================

def run_experiment(
    init_params,
    shape,
    parents,
    branch_lengths,
    node_loglik,
    config: ModelConfig,
    steps=200
):
    """
    Full parameter estimation workflow.

    Returns
    -------
    params : jnp.ndarray

    losses : jnp.ndarray
    """

    params, losses = fit_model(
        init_params,
        shape,
        parents,
        branch_lengths,
        node_loglik,
        config,
        steps
    )

    return params, losses

