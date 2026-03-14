import jax
import jax.numpy as jnp

from kernels import (
    interval_cosine_basis,
    interval_laplacian_eigenvalues,
    simplex_monomial_basis,
    s1_fourier_basis,
    s1_laplace_eigenvalues,
)


def test_interval_cosine_basis_orthogonality():
    L, U = 0.0, 1.0
    grid = jnp.linspace(L, U, 401)
    basis_vals = interval_cosine_basis(grid, K=3, L=L, U=U)
    products = basis_vals[..., :, None] * basis_vals[..., None, :]
    dx = grid[1] - grid[0]
    trapezoids = (products[:-1] + products[1:]) / 2
    integrals = jnp.sum(trapezoids, axis=0) * dx
    off_diag = integrals - jnp.diag(jnp.diag(integrals))
    assert jnp.allclose(off_diag, 0.0, atol=1e-6)


def test_interval_laplacian_eigenvalues_match_formula():
    K = 4
    sigma = 1.5
    L, U = -1.0, 2.0
    eigenvalues = interval_laplacian_eigenvalues(K, sigma, L, U)
    width = U - L
    ks = jnp.arange(0, K + 1)
    expected = -0.5 * sigma**2 * (ks * jnp.pi / width) ** 2
    assert jnp.allclose(eigenvalues, expected)


def test_s1_fourier_basis_structure():
    theta = jnp.array([0.0, jnp.pi / 4])
    basis_vals = s1_fourier_basis(theta, K=2)
    assert basis_vals.shape == (2, 5)
    # First entry is constant; cos(0)=1
    assert jnp.allclose(basis_vals[:, 0], 1.0 / jnp.sqrt(2.0 * jnp.pi))
    # Cosine and sine entries for θ=0 simplify
    assert jnp.isclose(basis_vals[0, 1], 1.0 / jnp.sqrt(jnp.pi))
    assert jnp.isclose(basis_vals[0, 2], 0.0)


def test_s1_laplace_eigenvalues_ordering():
    K = 3
    sigma = 2.0
    eigenvalues = s1_laplace_eigenvalues(K, sigma)
    expected = jnp.concatenate(
        [
            jnp.array([0.0]),
            jnp.repeat(-0.5 * sigma**2 * jnp.arange(1, K + 1) ** 2, 2),
        ]
    )
    assert jnp.allclose(eigenvalues, expected)


def test_simplex_basis_degree_one_orthonormality():
    key = jax.random.PRNGKey(0)
    p = jax.random.dirichlet(key, jnp.ones(3), shape=(1024,))
    basis_vals = simplex_monomial_basis(p, degree=1)
    gram = basis_vals.T @ basis_vals / basis_vals.shape[0]
    identity = jnp.eye(basis_vals.shape[-1])
    assert jnp.all(jnp.isfinite(gram))
    off_diag = gram - jnp.diag(jnp.diag(gram))
    assert jnp.max(jnp.abs(off_diag)) < 0.35
    assert jnp.all(jnp.diag(gram) > 0)
