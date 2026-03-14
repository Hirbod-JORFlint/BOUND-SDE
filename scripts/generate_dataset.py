"""Small dataset generator for BOUND-SDE experiments."""

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from main import ModelConfig, simulate_dataset
from manifolds import (
    create_circle_manifold,
    create_interval_manifold,
    create_simplex_manifold,
)


def _load_config(path: Path) -> dict:
    """Load experiment configuration from JSON.

    Parameters
    ----------
    path : Path
        Path to the configuration file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.
    """

    with open(path, "r") as fp:
        return json.load(fp)


def _instantiate_manifold(spec: dict):
    """Instantiate a manifold specification from configuration data.

    Parameters
    ----------
    spec : dict
        Dictionary describing the manifold type and hyperparameters.

    Returns
    -------
    manifold : ManifoldSpec
        Geometric manifold instance used for simulation.
    """

    kind = spec.get("type", "interval")

    if kind == "interval":
        return create_interval_manifold(spec.get("L", -1.0), spec.get("U", 1.0))

    if kind == "circle":
        return create_circle_manifold()

    if kind == "simplex":
        dplus1 = spec.get("dplus1", spec.get("dimension", 3))
        return create_simplex_manifold(dplus1, concentration=spec.get("concentration", 1.0))

    raise ValueError(f"Unknown manifold type: {kind}")


def _constant_drift(drift: jnp.ndarray, dim: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a constant drift function on the manifold.

    Parameters
    ----------
    drift : jnp.ndarray
        Drift vector with shape (dim,).

    dim : int
        Manifold state dimension.

    Returns
    -------
    fn : Callable[[jnp.ndarray], jnp.ndarray]
        Function returning the constant drift at any input.
    """

    if drift.shape[0] != dim:
        raise ValueError("Drift vector length must match manifold dimension.")

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(drift, x.shape)

    return fn


def _constant_diffusion(diffusion: jnp.ndarray, dim: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a constant diagonal diffusion matrix.

    Parameters
    ----------
    diffusion : jnp.ndarray
        Diffusion entries producing a diagonal matrix of shape (dim, dim).

    dim : int
        Manifold state dimension.

    Returns
    -------
    fn : Callable[[jnp.ndarray], jnp.ndarray]
        Function returning the diffusion matrix at any input.
    """

    if diffusion.shape[0] != dim:
        raise ValueError("Diffusion vector length must match manifold dimension.")

    matrix = jnp.diag(diffusion)

    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return matrix

    return fn


def _build_node_loglik(
    traits: jnp.ndarray,
    manifold,
    spectral_dim: int,
    tip_nodes: Sequence[int],
) -> jnp.ndarray:
    """Evaluate tip likelihoods by projecting tip states onto the spectral basis.

    Parameters
    ----------
    traits : jnp.ndarray
        Simulated trait matrix of shape (N, d).

    manifold : ManifoldSpec
        Manifold providing basis evaluation.

    spectral_dim : int
        Number of spectral basis functions.

    tip_nodes : Sequence[int]
        Indices of tip nodes in the tree.

    Returns
    -------
    node_loglik : jnp.ndarray
        Log-likelihood matrix of shape (N, spectral_dim) with tip rows populated.
    """

    if len(tip_nodes) == 0:
        raise ValueError("Configuration must list at least one tip index.")

    tip_indices = tuple(int(idx) for idx in tip_nodes)
    tip_states = traits[tip_indices]
    basis_vals = manifold.evaluate_basis(tip_states, spectral_dim)
    log_basis = jnp.log(jnp.clip(basis_vals, 1e-12))

    node_loglik = jnp.zeros((traits.shape[0], spectral_dim))
    node_loglik = node_loglik.at[tip_indices, :].set(log_basis)
    return node_loglik


def generate_dataset(config_path: Path, output_path: Path, seed: int = 0) -> Path:
    """Generate a small synthetic dataset according to configuration data.

    Parameters
    ----------
    config_path : Path
        File path to the JSON configuration.

    output_path : Path
        Destination path for the generated `.npz` dataset.

    seed : int, default=0
        Random seed used for trait simulation.

    Returns
    -------
    output_path : Path
        Path where the dataset has been written.
    """

    conf = _load_config(config_path)
    manifold_conf = conf["manifold"]
    manifold = _instantiate_manifold(manifold_conf)

    parents = jnp.array(conf["parents"], dtype=jnp.int32)
    branch_lengths = jnp.array(conf["branch_lengths"], dtype=jnp.float32)
    topo_order = jnp.array(conf["topo_order"], dtype=jnp.int32)
    root_state = jnp.array(conf["root_state"], dtype=jnp.float32)

    spectral_dim = int(conf.get("spectral_dim", 8))
    config = ModelConfig(
        spectral_dim=spectral_dim,
        dt=float(conf.get("dt", 0.01)),
        manifold=manifold,
        optimizer=conf.get("optimizer", "adam"),
    )

    dim = int(root_state.shape[-1])
    drift = jnp.asarray(conf.get("drift", [0.0]), dtype=jnp.float32).reshape((-1,))
    if drift.shape[0] not in {1, dim}:
        raise ValueError("Drift configuration must be scalar or match the state dimension.")
    if drift.shape[0] == 1 and dim != 1:
        drift = jnp.full((dim,), drift[0], dtype=jnp.float32)

    diffusion = jnp.asarray(conf.get("diffusion", 1.0), dtype=jnp.float32).reshape((-1,))
    if diffusion.shape[0] not in {1, dim}:
        raise ValueError("Diffusion configuration must be scalar or match the state dimension.")
    if diffusion.shape[0] == 1 and dim != 1:
        diffusion = jnp.full((dim,), diffusion[0], dtype=jnp.float32)

    drift_fn = _constant_drift(drift, dim)
    diffusion_fn = _constant_diffusion(diffusion, dim)

    traits = simulate_dataset(
        seed,
        root_state,
        parents,
        branch_lengths,
        topo_order,
        drift_fn,
        diffusion_fn,
        config,
    )

    tip_nodes = conf.get("tips", [])
    node_loglik = _build_node_loglik(traits, manifold, spectral_dim, tip_nodes)

    boundary_dim = int(conf.get("boundary_dim", 0))
    root_prior = jnp.full((spectral_dim,), 1.0 / spectral_dim)
    init_params = jnp.concatenate(
        [
            drift,
            diffusion,
            jnp.zeros((boundary_dim,), dtype=jnp.float32),
            root_prior,
        ]
    )

    param_shape = np.array([drift.shape[0], diffusion.shape[0], boundary_dim, spectral_dim], dtype=np.int32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        traits=np.asarray(traits),
        node_loglik=np.asarray(node_loglik),
        parents=np.asarray(parents),
        branch_lengths=np.asarray(branch_lengths),
        param_shape=param_shape,
        init_params=np.asarray(init_params),
    )

    print(f"Wrote dataset to {output_path}")

    return output_path


def main() -> None:
    """Command-line entry point for dataset generation."""

    parser = argparse.ArgumentParser(description="BOUND-SDE dataset generator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/example_config.json"),
        help="Path to the example configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/example_dataset.npz"),
        help="Destination file for the generated dataset.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    generate_dataset(args.config, args.output, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
