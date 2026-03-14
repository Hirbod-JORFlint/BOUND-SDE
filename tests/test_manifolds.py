import jax
import jax.numpy as jnp

from manifolds import (
    s1_project,
    s1_gradient,
    s1_laplace_beltrami,
    simplex_project,
    simplex_fisher_metric,
    simplex_laplace_beltrami,
)


def test_s1_project_maps_range():
    angles = jnp.array([-3.0, 0.0, 2 * jnp.pi, 4.0])
    projected = s1_project(angles)
    assert jnp.all(projected >= 0.0)
    assert jnp.all(projected < 2 * jnp.pi)


def test_s1_gradient_of_sin_is_cos():
    f = lambda theta: jnp.sin(theta)
    theta = jnp.linspace(0.0, 1.0, 5)
    grad = s1_gradient(f, theta)
    assert jnp.allclose(grad, jnp.cos(theta), atol=1e-6)


def test_s1_laplace_of_sin_is_negative():
    f = lambda theta: jnp.sin(theta)
    theta = jnp.linspace(0.0, 2 * jnp.pi, 9)
    lap = s1_laplace_beltrami(f, theta)
    assert jnp.allclose(lap, -jnp.sin(theta), atol=1e-5)


def test_simplex_project_returns_valid_probabilities():
    points = jnp.array([[0.2, -0.1, 0.9], [2.0, 0.5, -0.2]])
    projected = simplex_project(points)
    assert jnp.all(projected >= 0.0)
    sums = jnp.sum(projected, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-8)


def test_simplex_fisher_metric_diagonal():
    p = jnp.array([[0.2, 0.3, 0.5]])
    metric = simplex_fisher_metric(p)
    diag = jnp.diagonal(metric[0])
    assert jnp.allclose(diag, 1.0 / p[0])


def test_simplex_laplace_quadric():
    def f(point):
        return jnp.sum(point**2)

    key = jax.random.PRNGKey(0)
    p = jax.random.dirichlet(key, jnp.ones(3), shape=(10,))
    lap = simplex_laplace_beltrami(f, p)
    # Expected value for ∑ p_i^2: Δf = (d+1) + 2 with d+1=3 => 5
    assert jnp.allclose(lap, 5.0, atol=0.2)
