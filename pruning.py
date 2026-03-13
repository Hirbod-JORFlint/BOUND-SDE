# ============================================================
# Imports
# ============================================================
from enum import Enum
import jax
import jax.numpy as jnp

# ============================================================
# Spectral Branch Transition Operators
# ============================================================

def compute_branch_transitions(
    eigenvalues: jnp.ndarray,
    branch_lengths: jnp.ndarray
) -> jnp.ndarray:
    """
    Construct spectral transition matrices for every branch.

    Mathematical definition
    -----------------------

    For a diffusion process governed by

        dX_t = μ(X_t) dt + σ(X_t) dW_t

    with infinitesimal generator

        L φ_k = λ_k φ_k

    the transition density admits the spectral expansion

        p_t(x, y) =
        Σ_{k=0}^{M-1} exp(λ_k t) φ_k(x) φ_k(y)

    In spectral coordinates the transition operator is diagonal

        T_k(t) = exp(λ_k t)

    Therefore the matrix form is

        T(t) = diag(exp(λ_0 t), … , exp(λ_{M-1} t))

    For B branches with lengths t_b we construct

        T_b = diag(exp(λ_k t_b))

    Parameters
    ----------
    eigenvalues : jnp.ndarray
        Eigenvalues of the infinitesimal generator.

        Shape
        -----
        (M,)

    branch_lengths : jnp.ndarray
        Branch lengths for every edge in the tree.

        Shape
        -----
        (B,)

    Returns
    -------
    transitions : jnp.ndarray
        Spectral transition matrices.

        Shape
        -----
        (B, M, M)

    Notes
    -----
    The implementation is fully vectorized and compatible with
    JAX automatic differentiation.

    Complexity
    ----------
    O(B M)

    which satisfies the global O(N) requirement since
    the number of branches B scales linearly with nodes.
    """

    # --------------------------------------------------------
    # Compute spectral exponentials
    # --------------------------------------------------------

    # Shape: (B, M)
    spectral_exp = jnp.exp(branch_lengths[:, None] * eigenvalues[None, :])

    # --------------------------------------------------------
    # Convert vectors to diagonal matrices
    # --------------------------------------------------------

    def make_diag(v):
        """
        Convert spectral coefficient vector to diagonal matrix.

        Parameters
        ----------
        v : jnp.ndarray
            Shape (M,)

        Returns
        -------
        diag_matrix : jnp.ndarray
            Shape (M, M)
        """
        return jnp.diag(v)

    transitions = jax.vmap(make_diag)(spectral_exp)

    return transitions

# ============================================================
# Stabilized Transition Operator
# ============================================================

def compute_branch_transitions_stable(
    eigenvalues: jnp.ndarray,
    branch_lengths: jnp.ndarray
) -> jnp.ndarray:
    """
    Numerically stabilized version of spectral transitions.

    This implementation clips exponentials to prevent
    floating-point underflow.

    Parameters
    ----------
    eigenvalues : jnp.ndarray
        Shape (M,)

    branch_lengths : jnp.ndarray
        Shape (B,)

    Returns
    -------
    transitions : jnp.ndarray
        Shape (B, M, M)
    """

    spectral_exp = jnp.exp(branch_lengths[:, None] * eigenvalues[None, :])

    spectral_exp = jnp.clip(
        spectral_exp,
        a_min=1e-300,
        a_max=1e300
    )

    transitions = jax.vmap(jnp.diag)(spectral_exp)

    return transitions

# ============================================================
# Trait Geometry Enumeration
# ============================================================

class BoundaryDomain(Enum):
    """
    Enumeration of supported geometric constraint domains.
    """

    INTERVAL = 0
    CIRCLE = 1
    SIMPLEX = 2

# ============================================================
# Boundary Correction Transform
# ============================================================

def apply_boundary_correction(
    transitions: jnp.ndarray,
    boundary_operator: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply boundary normalization operator to spectral transitions.

    Mathematical definition
    -----------------------

    Boundary corrected operator

        T* = B T B^{-1}

    where

        B     = boundary normalization matrix
        B^{-1} = inverse normalization

    Parameters
    ----------
    transitions : jnp.ndarray
        Raw spectral transition matrices.

        Shape
        -----
        (B, M, M)

    boundary_operator : jnp.ndarray
        Boundary normalization matrix.

        Shape
        -----
        (M, M)

    Returns
    -------
    corrected : jnp.ndarray
        Boundary-corrected transition operators.

        Shape
        -----
        (B, M, M)
    """

    B = boundary_operator
    B_inv = jnp.linalg.inv(B)

    def transform(T):
        return B @ T @ B_inv

    corrected = jax.vmap(transform)(transitions)

    return corrected

# ============================================================
# Identity Boundary Operator
# ============================================================

def identity_boundary_operator(M: int) -> jnp.ndarray:
    """
    Construct identity boundary operator.

    This is used when the spectral basis already
    satisfies the boundary conditions of the domain.

    Parameters
    ----------
    M : int
        Spectral dimension.

    Returns
    -------
    B : jnp.ndarray

        Shape
        -----
        (M, M)
    """

    return jnp.eye(M)

# ============================================================
# Interval Boundary Operator
# ============================================================

def interval_boundary_operator(
    eigenvalues: jnp.ndarray
) -> jnp.ndarray:
    """
    Construct reflective boundary normalization matrix
    for interval diffusion.

    Mathematical basis
    ------------------

    Reflective boundaries impose

        ∂φ_k/∂n = 0

    which is satisfied by cosine eigenfunctions.

    We therefore use a diagonal normalization matrix

        B_k = sqrt(|λ_k| + 1)

    Parameters
    ----------
    eigenvalues : jnp.ndarray

        Shape
        -----
        (M,)

    Returns
    -------
    B : jnp.ndarray

        Shape
        -----
        (M, M)
    """

    scale = jnp.sqrt(jnp.abs(eigenvalues) + 1.0)

    return jnp.diag(scale)

# ============================================================
# Circular Boundary Operator
# ============================================================

def circular_boundary_operator(
    M: int
) -> jnp.ndarray:
    """
    Boundary operator for circular manifold S1.

    Since Fourier eigenfunctions satisfy periodic
    boundary conditions exactly, no correction
    is required.

    Parameters
    ----------
    M : int

    Returns
    -------
    B : jnp.ndarray
        Identity matrix.

        Shape
        -----
        (M, M)
    """

    return jnp.eye(M)

# ============================================================
# Simplex Boundary Operator
# ============================================================

def simplex_boundary_operator(
    eigenvalues: jnp.ndarray
) -> jnp.ndarray:
    """
    Construct boundary normalization operator
    for simplex diffusion.

    Based on Fisher metric scaling.

    Mathematical form
    -----------------

        B_k = sqrt(|λ_k| + ε)

    Parameters
    ----------
    eigenvalues : jnp.ndarray

        Shape
        -----
        (M,)

    Returns
    -------
    B : jnp.ndarray

        Shape
        -----
        (M, M)
    """

    eps = 1e-6

    scale = jnp.sqrt(jnp.abs(eigenvalues) + eps)

    return jnp.diag(scale)

# ============================================================
# Boundary Operator Dispatcher
# ============================================================

def construct_boundary_operator(
    domain: BoundaryDomain,
    eigenvalues: jnp.ndarray,
    M: int
) -> jnp.ndarray:
    """
    Construct appropriate boundary normalization operator.

    Parameters
    ----------
    domain : BoundaryDomain

    eigenvalues : jnp.ndarray
        Shape (M,)

    M : int

    Returns
    -------
    B : jnp.ndarray
        Shape (M, M)
    """

    if domain == BoundaryDomain.INTERVAL:
        return interval_boundary_operator(eigenvalues)

    elif domain == BoundaryDomain.CIRCLE:
        return circular_boundary_operator(M)

    elif domain == BoundaryDomain.SIMPLEX:
        return simplex_boundary_operator(eigenvalues)

    else:
        raise ValueError("Unsupported boundary domain.")


