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
    """
    Evaluate spectral basis functions over a set of states.

    Parameters
    ----------
    states : jnp.ndarray
        Evaluation points.

        Shape
        -----
        (N, d)

    basis_fn : Callable
        Function mapping state -> basis vector.

    Returns
    -------
    Phi : jnp.ndarray
        Basis matrix.

        Shape
        -----
        (N, M)

    Notes
    -----
    Computes

        Φ_{ni} = φ_i(x_n)

    where φ_i are basis functions.
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
    """
    Compute Galerkin projection matrices.

    Parameters
    ----------
    states : jnp.ndarray
        Quadrature or Monte-Carlo sample points.

        Shape
        -----
        (N, d)

    basis_fn : Callable
        Basis evaluation function.

    generator_fn : Callable
        Generator operator.

    Returns
    -------
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

    Phi : jnp.ndarray
        Basis evaluations.

        Shape
        -----
        (N, M)

    Notes
    -----

    We approximate inner products via Monte Carlo:

        G_ij = (1/N) Σ_n φ_i(x_n) φ_j(x_n)

        A_ij = (1/N) Σ_n φ_i(x_n) (L φ_j)(x_n)

    where L is the infinitesimal generator.
    """

    # Evaluate basis
    Phi = evaluate_basis_grid(states, basis_fn)  # (N, M)

    # Apply generator
    Lphi = jax.vmap(generator_fn)(states)  # (N, M)

    N = states.shape[0]

    # Gram matrix
    G = (Phi.T @ Phi) / N

    # Generator projection
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
    """
    Solve the generalized eigenvalue problem.

        A v = λ G v

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
        Diagonal regularization.

    Returns
    -------
    eigenvalues : jnp.ndarray
        Spectral eigenvalues.

        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray
        Eigenvectors in basis coordinates.

        Shape
        -----
        (M, M)

    Notes
    -----

    We convert

        A v = λ G v

    into a standard symmetric problem.

    Let

        G = L L^T

    then

        B = L^{-1} A L^{-T}

    Solve

        B u = λ u

    Recover

        v = L^{-T} u
    """

    M = G.shape[0]

    # Regularize Gram matrix
    G_reg = G + reg * jnp.eye(M)

    # Cholesky
    L = jnp.linalg.cholesky(G_reg)

    # Compute transformed operator
    Linv = jsp.linalg.solve_triangular(L, jnp.eye(M), lower=True)

    B = Linv @ A @ Linv.T

    # Symmetrize (numerical safety)
    B = 0.5 * (B + B.T)

    eigenvalues, U = jnp.linalg.eigh(B)

    # Recover eigenvectors
    V = Linv.T @ U

    return eigenvalues, V


# ------------------------------------------------------------
# Eigenfunction Construction
# ------------------------------------------------------------

def construct_eigenfunctions(
    Phi: jnp.ndarray,
    eigenvectors: jnp.ndarray
) -> jnp.ndarray:
    """
    Construct eigenfunctions evaluated on grid.

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
        Eigenfunctions evaluated on grid.

        Shape
        -----
        (N, M)

    Notes
    -----

    Compute

        ψ_k(x_n) = Σ_j v_{jk} φ_j(x_n)
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
    """
    Evaluate spectral transition density approximation.

    Parameters
    ----------
    Psi_x : jnp.ndarray
        Eigenfunctions at x locations.

        Shape
        -----
        (Nx, M)

    Psi_y : jnp.ndarray
        Eigenfunctions at y locations.

        Shape
        -----
        (Ny, M)

    eigenvalues : jnp.ndarray
        Spectral eigenvalues.

        Shape
        -----
        (M,)

    t : float
        Time parameter.

    Returns
    -------
    P : jnp.ndarray
        Transition kernel matrix.

        Shape
        -----
        (Nx, Ny)

    Notes
    -----

    Using expansion

        p_t(x,y) ≈ Σ_k exp(λ_k t) ψ_k(x) ψ_k(y)
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
    """
    Construct transition operator along a tree branch.

    Parameters
    ----------
    eigenvalues : jnp.ndarray
        Spectral eigenvalues.

        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray
        Spectral eigenvectors.

        Shape
        -----
        (M, M)

    branch_length : float
        Evolutionary time.

    Returns
    -------
    T : jnp.ndarray
        Branch transition matrix.

        Shape
        -----
        (M, M)

    Notes
    -----

    Using semigroup representation

        T(t) = V exp(Λ t) V^{-1}

    where

        Λ = diag(λ_k)

    For orthonormal bases

        V^{-1} = V^T
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
