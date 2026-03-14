import jax.numpy as jnp

from spectral_solver import (
    branch_transition_matrix,
    spectral_transition_kernel,
    solve_spectral_decomposition,
)


def test_branch_transition_matrix_matches_exponential():
    eigenvalues = jnp.array([-1.0, -4.0])
    eigenvectors = jnp.eye(2)
    branch_length = 0.5
    T = branch_transition_matrix(eigenvalues, eigenvectors, branch_length)
    expected = jnp.diag(jnp.exp(eigenvalues * branch_length))
    assert jnp.allclose(T, expected)


def test_spectral_transition_kernel_shape():
    Psi_x = jnp.ones((3, 2))
    Psi_y = jnp.ones((4, 2))
    eigenvalues = jnp.array([-1.0, -2.0])
    kernel = spectral_transition_kernel(Psi_x, Psi_y, eigenvalues, t=0.1)
    assert kernel.shape == (3, 4)


def test_solve_spectral_decomposition_with_identity_gram():
    A = jnp.diag(jnp.array([-2.0, -1.0]))
    G = jnp.eye(2)
    eigenvalues, eigenvectors = solve_spectral_decomposition(A, G)
    assert jnp.allclose(eigenvalues, jnp.array([-2.0, -1.0]), atol=1e-6)
    assert jnp.allclose(eigenvectors @ eigenvectors.T, jnp.eye(2), atol=1e-6)
