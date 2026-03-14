import jax.numpy as jnp
import jax.scipy as jsp

from pruning import (
    BoundaryDomain,
    compute_branch_transitions,
    construct_boundary_operator,
    propagate_branch_likelihood,
    normalize_all_nodes,
    compute_root_loglik,
)


def test_compute_branch_transitions_returns_diagonal():
    eigenvalues = jnp.array([-1.0, -2.0])
    branch_lengths = jnp.array([0.1, 0.2])
    transitions = compute_branch_transitions(eigenvalues, branch_lengths)
    assert transitions.shape == (2, 2, 2)
    expected = jnp.stack(
        [jnp.diag(jnp.exp(eigenvalues * t)) for t in branch_lengths]
    )
    assert jnp.allclose(transitions, expected)


def test_propagate_branch_likelihood_logspace():
    child_loglik = jnp.array([0.0, 0.0])
    transition = jnp.eye(2)
    message = propagate_branch_likelihood(child_loglik, transition)
    assert jnp.allclose(message, jnp.log(jnp.array([1.0, 1.0])))


def test_construct_boundary_operator_circle_identity():
    M = 3
    eig = jnp.array([-1.0, -2.0, -3.0])
    B = construct_boundary_operator(BoundaryDomain.CIRCLE, eig, M)
    assert jnp.allclose(B, jnp.eye(M))


def test_normalize_all_nodes_centers_rows():
    loglik = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    normalized, scales = normalize_all_nodes(loglik)
    assert normalized.shape == loglik.shape
    assert scales.shape == (2,)
    row_logsum = jsp.special.logsumexp(normalized, axis=1)
    assert jnp.allclose(row_logsum, jnp.zeros_like(scales))


def test_compute_root_loglik_uniform_prior():
    node_loglik = jnp.array([[0.0, 0.0]])
    scales = jnp.array([0.0])
    prior = jnp.array([0.5, 0.5])
    loglik = compute_root_loglik(node_loglik, scales, 0, prior)
    assert jnp.isclose(loglik, 0.0)
