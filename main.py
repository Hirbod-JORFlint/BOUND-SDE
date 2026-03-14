# ============================================================
# BOUND-SDE Main Entry Module
# ============================================================

import jax
import jax.numpy as jnp
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from typing import NamedTuple

from likelihood import (
    vectorized_neg_loglik,
    grad_vectorized_neg_loglik,
    ParamShape,
)

from manifolds import (
    create_interval_manifold,
    create_circle_manifold,
    create_simplex_manifold,
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


def _instantiate_manifold(spec):
    r"""
    Construct manifold instances from configuration.
    """

    kind = spec.get("type", "interval")

    if kind == "interval":
        return create_interval_manifold(spec.get("L", 0.0), spec.get("U", 1.0))

    if kind == "circle":
        return create_circle_manifold()

    if kind == "simplex":
        dplus1 = spec.get("dplus1", spec.get("dimension", 3))
        return create_simplex_manifold(dplus1, concentration=spec.get("concentration", 1.0))

    raise ValueError(f"Unknown manifold type: {kind}")

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


def _load_json_config(path: str) -> dict:
    with open(Path(path), "r") as fp:
        return json.load(fp)


def _constant_drift(drift_value):
    value = jnp.asarray(drift_value)

    def fn(x):
        return jnp.broadcast_to(value, x.shape)

    return fn


def _constant_diffusion(diff_value, dim: int):
    matrix = jnp.eye(dim) * diff_value

    def fn(x):
        return matrix

    return fn


def _write_traits(path, traits):
    np.savez(path, traits=np.asarray(traits))


def _load_param_shape(data: dict) -> ParamShape:
    return ParamShape(
        int(data.get("drift_dim", 1)),
        int(data.get("diffusion_dim", 1)),
        int(data.get("boundary_dim", 1)),
        int(data.get("prior_dim", 1)),
    )


def _run_cli():
    parser = argparse.ArgumentParser(description="BOUND-SDE experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--output", required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--data", required=True)
    fit.add_argument("--config", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--steps", type=int, default=200)

    test = subparsers.add_parser("test")
    test.add_argument("--which", default="all")

    args = parser.parse_args()

    if args.command == "simulate":
        conf = _load_json_config(args.config)
        manifold = _instantiate_manifold(conf["manifold"])
        root_state = jnp.array(conf["root_state"])
        parents = jnp.array(conf["parents"])
        branch_lengths = jnp.array(conf["branch_lengths"])
        topo_order = jnp.array(conf["topo_order"])
        drift_fn = _constant_drift(conf.get("drift", [0.0]))
        diffusion_fn = _constant_diffusion(conf.get("diffusion", 1.0), root_state.shape[0])
        config = ModelConfig(
            spectral_dim=conf.get("spectral_dim", 8),
            dt=conf.get("dt", 0.01),
            manifold=manifold,
            optimizer=conf.get("optimizer", "adam"),
        )
        traits = simulate_dataset(
            args.seed,
            root_state,
            parents,
            branch_lengths,
            topo_order,
            drift_fn,
            diffusion_fn,
            config
        )
        _write_traits(args.output, traits)

    elif args.command == "fit":
        data = np.load(args.data)
        conf = _load_json_config(args.config)
        manifold = _instantiate_manifold(conf["manifold"])
        shape = ParamShape(*data["param_shape"])
        init_params = jnp.array(data["init_params"])
        parents = jnp.array(data["parents"])
        branch_lengths = jnp.array(data["branch_lengths"])
        node_loglik = jnp.array(data["node_loglik"])
        config = ModelConfig(
            spectral_dim=conf.get("spectral_dim", 8),
            dt=conf.get("dt", 0.01),
            manifold=manifold,
            optimizer=conf.get("optimizer", "adam"),
        )
        params, losses = fit_model(
            init_params,
            shape,
            parents,
            branch_lengths,
            node_loglik,
            config,
            steps=args.steps
        )
        np.savez(
            args.output,
            params=np.asarray(params),
            losses=np.asarray(losses)
        )

    elif args.command == "test":
        if args.which == "all":
            subprocess.run(["python", "-m", "pytest"], check=True)
        else:
            subprocess.run(["python", "-m", "pytest", args.which], check=True)

    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    _run_cli()

