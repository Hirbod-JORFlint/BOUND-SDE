import jax.numpy as jnp

from optimizers import run_adam, run_lbfgs


def quadratic_objective(params):
    loss = 0.5 * jnp.dot(params, params)
    grad = params
    return loss, grad


def test_run_adam_descends():
    init = jnp.array([1.0, -1.0])
    final, losses = run_adam(init, quadratic_objective, num_steps=30)
    assert losses[-1] < losses[0]
    assert losses.shape[0] == 30


def test_run_lbfgs_converges():
    init = jnp.array([1.0, 2.0])
    final, losses = run_lbfgs(init, quadratic_objective, num_steps=10, memory=5)
    assert losses[-1] <= losses[0]
    assert final.shape == init.shape
    assert losses.shape[0] == 10
