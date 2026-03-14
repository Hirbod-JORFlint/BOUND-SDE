import jax
import jax.numpy as jnp

import simulations

from manifolds import create_interval_manifold
from simulations import (
    simulate_sde_step,
    simulate_sde_path,
    simulate_tree_traits,
)


def constant_drift(x):
    return jnp.zeros_like(x)


def constant_diffusion(x):
    d = x.shape[0]
    return jnp.eye(d)


def test_simulate_sde_step_shapes():
    manifold = create_interval_manifold(0.0, 1.0)
    key = jax.random.PRNGKey(0)
    x_next = simulate_sde_step(
        key,
        jnp.array([0.5]),
        constant_drift,
        constant_diffusion,
        manifold,
        dt=0.01
    )
    assert x_next.shape == (1,)


def test_simulate_sde_path_length():
    manifold = create_interval_manifold(0.0, 1.0)
    key = jax.random.PRNGKey(1)
    path = simulate_sde_path(
        key,
        jnp.array([0.2]),
        constant_drift,
        constant_diffusion,
        manifold,
        dt=0.05,
        steps=5
    )
    assert path.shape == (5, 1)


def test_simulate_tree_traits_returns_all_nodes(monkeypatch):
    def fake_branch(key, x_parent, branch_length, drift_fn, diffusion_fn, manifold, dt):
        return x_parent

    monkeypatch.setattr("simulations.simulate_branch", fake_branch)
    manifold = create_interval_manifold(0.0, 1.0)
    key = jax.random.PRNGKey(2)
    parents = jnp.array([-1, 0, 0])
    branch_lengths = jnp.array([0.0, 0.1, 0.1])
    topo_order = jnp.array([0, 1, 2])
    traits = simulate_tree_traits(
        key,
        jnp.array([0.5]),
        parents,
        branch_lengths,
        topo_order,
        constant_drift,
        constant_diffusion,
        manifold,
        dt=0.05
    )
    assert traits.shape == (3, 1)
