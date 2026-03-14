import jax.numpy as jnp
import numpy as np
import pytest

import simulations

from bridge_r import (
    r_tree_to_arrays,
    r_traits_to_jax,
    r_compute_likelihood,
    r_simulate_traits,
    r_get_postorder,
)


def test_r_tree_to_arrays_basic():
    edge = np.array([[0, 1]])
    lengths = np.array([0.1])
    tree = r_tree_to_arrays(edge, lengths)
    assert tree["parents"].shape[0] >= 2
    assert tree["branch_lengths"].shape[0] >= 2


def test_r_traits_to_jax():
    traits = np.array([[0.1, 0.2]])
    converted = r_traits_to_jax(traits)
    assert isinstance(converted, jnp.ndarray)


def test_r_compute_likelihood_monkeypatched(monkeypatch):
    def fake_neg_loglik(*args, **kwargs):
        return jnp.array(2.0)

    monkeypatch.setattr("bridge_r.vectorized_neg_loglik", fake_neg_loglik)
    val = r_compute_likelihood(
        params=np.zeros(4),
        parents=np.array([-1, 0]),
        branch_lengths=np.array([0.0, 0.1]),
        node_loglik=np.zeros((2, 1)),
        manifold=None,
        spectral_dim=1,
    )
    assert val == pytest.approx(-2.0)


def test_r_simulate_traits_monkeypatched(monkeypatch):
    def fake_simulate(*args, **kwargs):
        return jnp.array([[0.1], [0.2]])

    import simulations

    monkeypatch.setattr("simulations.simulate_tree_traits", fake_simulate)
    result = r_simulate_traits(
        key_seed=0,
        root_state=np.array([0.5]),
        parents=np.array([-1, 0]),
        branch_lengths=np.array([0.0, 0.1]),
        topo_order=np.array([0, 1]),
        drift_fn=lambda x: x,
        diffusion_fn=lambda x: np.eye(x.shape[0]),
        manifold=None,
        dt=0.1,
    )
    assert isinstance(result, np.ndarray)


def test_r_get_postorder_simple():
    parents = np.array([-1, 0, 0])
    postorder = r_get_postorder(parents)
    assert hasattr(postorder, "shape")
    assert postorder[0] in (1, 2)
    assert postorder[-1] == 0
