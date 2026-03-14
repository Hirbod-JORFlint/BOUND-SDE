import jax
import jax.numpy as jnp
import numpy as np

from main import (
    build_objective,
    simulate_dataset,
    ModelConfig,
)


def test_simulate_dataset_monkeypatched(monkeypatch):
    def fake_simulate(*args, **kwargs):
        return jnp.array([[0.1], [0.2]])

    monkeypatch.setattr("main.simulate_tree_traits", fake_simulate)
    config = ModelConfig(
        spectral_dim=1,
        dt=0.1,
        manifold=None,
        optimizer="adam",
    )
    traits = simulate_dataset(
        seed=0,
        root_state=jnp.array([0.5]),
        parents=jnp.array([-1, 0]),
        branch_lengths=jnp.array([0.0, 0.1]),
        topo_order=jnp.array([0, 1]),
        drift_fn=lambda x: x,
        diffusion_fn=lambda x: jnp.eye(x.shape[0]),
        config=config
    )
    assert traits.shape == (2, 1)


def test_build_objective_uses_vectorized_neg_loglik(monkeypatch):
    def fake_loss(vec, shape, parents, branch_lengths, node_loglik, manifold, spectral_dim):
        return jnp.array(1.0)

    def fake_grad(vec, shape, parents, branch_lengths, node_loglik, manifold, spectral_dim):
        return jnp.zeros_like(vec)

    monkeypatch.setattr("main.vectorized_neg_loglik", fake_loss)
    monkeypatch.setattr("main.grad_vectorized_neg_loglik", fake_grad)

    shape = (4,)
    grad_fn = build_objective(
        shape,
        parents=jnp.array([-1, 0]),
        branch_lengths=jnp.array([0.0, 0.1]),
        node_loglik=jnp.zeros((2, 1)),
        manifold=None,
        spectral_dim=1
    )
    loss, grad = grad_fn(jnp.zeros(shape))
    assert jnp.isscalar(loss)
    assert grad.shape == shape
