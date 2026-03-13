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
