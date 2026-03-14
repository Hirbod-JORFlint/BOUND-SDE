"""
Spectral solver for constrained SDE transition operators.

This module implements a Galerkin projection method for approximating
the infinitesimal generator of a diffusion process using a finite
spectral basis.

Mathematical Framework
----------------------

Consider the SDE

    dX_t = μ(X_t) dt + σ(X_t) dW_t

The infinitesimal generator is

    L f = μ · ∇f + (1/2) Tr(a ∇²f)

where

    a = σ σ^T

Given basis functions {φ_i(x)}, we form

    G_ij = <φ_i, φ_j>
    A_ij = <φ_i, L φ_j>

The generalized eigenproblem

    A v = λ G v

produces eigenvalues λ_k and eigenvectors v_k.

The transition kernel is approximated by

    p_t(x,y) ≈ Σ_k exp(λ_k t) ψ_k(x) ψ_k(y)

where

    ψ_k(x) = Σ_j v_{jk} φ_j(x)

All routines are designed to be JAX-compatible and vectorized.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from manifolds import ManifoldSpec


# ------------------------------------------------------------
# Basis Evaluation
# ------------------------------------------------------------

def evaluate_basis_grid(
    states: jnp.ndarray,
    basis_fn: Callable
) -> jnp.ndarray:
    r"""
    Evaluate the spectral basis \(\phi_i(x_n)\) over grid states.

    Parameters
    ----------
    states : jnp.ndarray
        Evaluation points.

        Shape
        -----
        (N, d)

    basis_fn : Callable
        Function returning \(\phi(x)\).

    Returns
    -------
    Phi : jnp.ndarray
        Matrix of basis evaluations.

        Shape
        -----
        (N, M)
    """

    Phi = jax.vmap(basis_fn)(states)

    return Phi


# ------------------------------------------------------------
# Generator Projection
# ------------------------------------------------------------

def compute_generator_projection(
    states: jnp.ndarray,
    basis_fn: Callable,
    generator_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Form the Galerkin projection of the generator onto the basis.

    The matrices are computed as

        G_{ij} = \frac{1}{N} \sum_n \phi_i(x_n) \phi_j(x_n),\quad
        A_{ij} = \frac{1}{N} \sum_n \phi_i(x_n) (L\phi_j)(x_n).

    Parameters
    ----------
    states : jnp.ndarray
        Sample points.

        Shape
        -----
        (N, d)

    basis_fn : Callable
        Spectral basis evaluator.

    generator_fn : Callable
        Infinitesimal generator.

    Returns
    -------
    A : jnp.ndarray
        Generator projection.

        Shape
        -----
        (M, M)

    G : jnp.ndarray
        Gram matrix.

        Shape
        -----
        (M, M)

    Phi : jnp.ndarray
        Basis matrix.

        Shape
        -----
        (N, M)
    """

    Phi = evaluate_basis_grid(states, basis_fn)
    Lphi = jax.vmap(generator_fn)(states)
    N = states.shape[0]
    G = (Phi.T @ Phi) / N
    A = (Phi.T @ Lphi) / N
    return A, G, Phi


# ------------------------------------------------------------
# Generalized Eigenvalue Solver
# ------------------------------------------------------------

def solve_spectral_decomposition(
    A: jnp.ndarray,
    G: jnp.ndarray,
    reg: float = 1e-8
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Solve the generalized eigensystem \(A v = \lambda G v\).

    Parameters
    ----------
    A : jnp.ndarray
        Generator matrix.

        Shape
        -----
        (M, M)

    G : jnp.ndarray
        Gram matrix.

        Shape
        -----
        (M, M)

    reg : float
        Regularization added to \(G\)'s diagonal.

    Returns
    -------
    eigenvalues : jnp.ndarray
        Spectral eigenvalues \(\lambda_k\).

        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray
        Basis coefficients.

        Shape
        -----
        (M, M)
    """

    M = G.shape[0]
    G_reg = G + reg * jnp.eye(M)
    L = jnp.linalg.cholesky(G_reg)
    Linv = jsp.linalg.solve_triangular(L, jnp.eye(M), lower=True)
    B = Linv @ A @ Linv.T
    B = 0.5 * (B + B.T)
    eigenvalues, U = jnp.linalg.eigh(B)
    V = Linv.T @ U
    return eigenvalues, V


# ------------------------------------------------------------
# Eigenfunction Construction
# ------------------------------------------------------------

def construct_eigenfunctions(
    Phi: jnp.ndarray,
    eigenvectors: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Reconstruct eigenfunctions \(\psi_k(x) = \sum_j v_{jk} \phi_j(x)\).

    Parameters
    ----------
    Phi : jnp.ndarray
        Basis matrix.

        Shape
        -----
        (N, M)

    eigenvectors : jnp.ndarray
        Spectral coefficients.

        Shape
        -----
        (M, M)

    Returns
    -------
    Psi : jnp.ndarray
        Eigenfunctions evaluated at the sample grid.

        Shape
        -----
        (N, M)
    """

    Psi = Phi @ eigenvectors

    return Psi


# ------------------------------------------------------------
# Spectral Transition Kernel
# ------------------------------------------------------------

def spectral_transition_kernel(
    Psi_x: jnp.ndarray,
    Psi_y: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    t: float
) -> jnp.ndarray:
    r"""
    Approximate the transition kernel

        \(p_t(x,y) ≈ \sum_k e^{\lambda_k t} \psi_k(x) \psi_k(y)\).

    Parameters
    ----------
    Psi_x : jnp.ndarray
        Eigenfunctions at source locations.

        Shape
        -----
        (Nx, M)

    Psi_y : jnp.ndarray
        Eigenfunctions at target locations.

        Shape
        -----
        (Ny, M)

    eigenvalues : jnp.ndarray
        Spectral eigenvalues.

        Shape
        -----
        (M,)

    t : float
        Time argument.

    Returns
    -------
    P : jnp.ndarray
        Approximated transition matrix.

        Shape
        -----
        (Nx, Ny)
    """

    exp_term = jnp.exp(eigenvalues * t)
    weighted = Psi_x * exp_term
    P = weighted @ Psi_y.T
    return P


# ------------------------------------------------------------
# Branch Transition Matrix
# ------------------------------------------------------------

def branch_transition_matrix(
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
    branch_length: float
) -> jnp.ndarray:
    r"""
    Build the branch operator \(T = V \exp(\Lambda t) V^T\).

    Parameters
    ----------
    eigenvalues : jnp.ndarray
        Spectral eigenvalues.

        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray
        Basis coefficients.

        Shape
        -----
        (M, M)

    branch_length : float
        Evolutionary time duration.

    Returns
    -------
    T : jnp.ndarray
        Spectral transition matrix.

        Shape
        -----
        (M, M)
    """

    exp_diag = jnp.exp(eigenvalues * branch_length)
    D = jnp.diag(exp_diag)
    T = eigenvectors @ D @ eigenvectors.T
    return T


def compute_spectral_decomposition(
    params,
    manifold: ManifoldSpec,
    spectral_dim: int,
    num_states: int = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build spectral eigenpairs for the generator defined on `manifold`.

    The generator satisfies

        \mathcal{L} \phi_k = \lambda_k \phi_k

    and the transition operator is approximated as

        T_t(x,y) \approx \sum_k e^{\lambda_k t} \phi_k(x) \phi_k(y).

    Parameters
    ----------
    params : ModelParams-like
        Parameters defining drift, diffusion, and boundaries.

    manifold : ManifoldSpec
        Geometry describing the trait domain.

    spectral_dim : int
        Number of spectral basis functions (M).

    num_states : int, optional
        Number of sample points used for Monte-Carlo inner products.

    Returns
    -------
    eigenvalues : jnp.ndarray
        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray
        Shape
        -----
        (M, M)

    inv_eigenvectors : jnp.ndarray
        Shape
        -----
        (M, M)
    """

    num_states = num_states or max(512, spectral_dim * 12, 256)

    states = manifold.sample_states(num_states, spectral_dim)

    basis_fn = lambda x: manifold.evaluate_basis(x, spectral_dim)

    generator_fn = lambda x: manifold.apply_generator(params, x, spectral_dim)

    A, G, _ = compute_generator_projection(states, basis_fn, generator_fn)

    eigenvalues, eigenvectors = solve_spectral_decomposition(A, G)

    inv_eigenvectors = jnp.linalg.inv(eigenvectors)

    return eigenvalues, eigenvectors, inv_eigenvectors
