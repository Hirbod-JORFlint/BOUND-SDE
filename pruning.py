# ============================================================
# Imports
# ============================================================
from enum import Enum
import jax
import jax.numpy as jnp
import jax.scipy as jsp

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

# ============================================================
# Propagate Likelihood Across Single Branch
# ============================================================

def propagate_branch_likelihood(
    child_loglik: jnp.ndarray,
    transition: jnp.ndarray
) -> jnp.ndarray:
    """
    Propagate likelihood from child node to parent node.

    Mathematical definition
    -----------------------

    Let L_c(m) be the log-likelihood vector at child node c.

    For a branch with transition matrix T_{km}(t),

        M_{c→p}(k)
        =
        log Σ_m T_{km}(t) exp(L_c(m))

    To maintain numerical stability we compute

        M_{c→p}(k)
        =
        logsumexp( log(T_{km}(t)) + L_c(m) )

    Parameters
    ----------
    child_loglik : jnp.ndarray
        Log-likelihood vector at child node.

        Shape
        -----
        (M,)

    transition : jnp.ndarray
        Spectral transition matrix for branch.

        Shape
        -----
        (M, M)

    Returns
    -------
    message : jnp.ndarray
        Log-likelihood message sent to parent node.

        Shape
        -----
        (M,)
    """

    # --------------------------------------------------------
    # Compute log transition matrix
    # --------------------------------------------------------

    logT = jnp.log(transition + 1e-300)

    # --------------------------------------------------------
    # Broadcast addition
    # --------------------------------------------------------

    # shape (M,M)
    combined = logT + child_loglik[None, :]

    # --------------------------------------------------------
    # Log-sum-exp over child states
    # --------------------------------------------------------

    message = jsp.special.logsumexp(combined, axis=1)

    return message

# ============================================================
# Vectorized Branch Propagation
# ============================================================

def propagate_all_branches(
    child_logliks: jnp.ndarray,
    transitions: jnp.ndarray
) -> jnp.ndarray:
    """
    Propagate likelihood messages along all branches.

    Parameters
    ----------
    child_logliks : jnp.ndarray
        Log-likelihood vectors for each branch child node.

        Shape
        -----
        (B, M)

    transitions : jnp.ndarray
        Transition matrices for each branch.

        Shape
        -----
        (B, M, M)

    Returns
    -------
    messages : jnp.ndarray
        Branch likelihood messages.

        Shape
        -----
        (B, M)
    """

    return jax.vmap(propagate_branch_likelihood)(
        child_logliks,
        transitions
    )

# ============================================================
# Propagate Messages Using Tree Structure
# ============================================================

def propagate_tree_branches(
    node_loglik: jnp.ndarray,
    branch_child: jnp.ndarray,
    transitions: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute branch messages using tree child indexing.

    Parameters
    ----------
    node_loglik : jnp.ndarray
        Log-likelihood matrix for all nodes.

        Shape
        -----
        (N, M)

    branch_child : jnp.ndarray
        Child node index for each branch.

        Shape
        -----
        (B,)

    transitions : jnp.ndarray
        Spectral transition matrices.

        Shape
        -----
        (B, M, M)

    Returns
    -------
    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)
    """

    child_logliks = node_loglik[branch_child]

    branch_messages = propagate_all_branches(
        child_logliks,
        transitions
    )

    return branch_messages

# ============================================================
# Aggregate Messages From Children
# ============================================================

def aggregate_child_messages(
    branch_messages: jnp.ndarray,
    node_branches: jnp.ndarray
) -> jnp.ndarray:
    """
    Aggregate incoming branch messages to compute node log-likelihood.

    Mathematical definition
    -----------------------

    Let M_{c→p}(k) be the message from child c to parent p.

    The node likelihood is

        log L_p(k)
        =
        Σ_{c ∈ children(p)} M_{c→p}(k)

    Parameters
    ----------
    branch_messages : jnp.ndarray
        Messages from branches.

        Shape
        -----
        (B, M)

    node_branches : jnp.ndarray
        Indices of branches entering a node.

        Shape
        -----
        (max_degree,)

        Negative values indicate empty entries.

    Returns
    -------
    node_loglik : jnp.ndarray
        Aggregated node log-likelihood vector.

        Shape
        -----
        (M,)
    """

    M = branch_messages.shape[1]

    def get_msg(b):
        return jax.lax.cond(
            b >= 0,
            lambda _: branch_messages[b],
            lambda _: jnp.zeros((M,)),
            operand=None
        )

    msgs = jax.vmap(get_msg)(node_branches)

    node_loglik = jnp.sum(msgs, axis=0)

    return node_loglik

# ============================================================
# Aggregate Messages For Entire Tree
# ============================================================

def aggregate_all_nodes(
    branch_messages: jnp.ndarray,
    node_branches: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute aggregated node likelihoods for all nodes.

    Parameters
    ----------
    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)

    node_branches : jnp.ndarray

        Shape
        -----
        (N, max_degree)

    Returns
    -------
    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)
    """

    return jax.vmap(
        lambda branches: aggregate_child_messages(
            branch_messages,
            branches
        )
    )(node_branches)

# ============================================================
# Efficient Masked Aggregation
# ============================================================

def aggregate_child_messages_masked(
    branch_messages: jnp.ndarray,
    node_branches: jnp.ndarray
) -> jnp.ndarray:
    """
    Efficient aggregation using masked indexing.

    Parameters
    ----------
    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)

    node_branches : jnp.ndarray

        Shape
        -----
        (N, max_degree)

    Returns
    -------
    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)
    """

    B, M = branch_messages.shape

    max_degree = node_branches.shape[1]

    def aggregate(node_branch_row):

        mask = node_branch_row >= 0

        safe_idx = jnp.where(mask, node_branch_row, 0)

        msgs = branch_messages[safe_idx]

        msgs = msgs * mask[:, None]

        return jnp.sum(msgs, axis=0)

    return jax.vmap(aggregate)(node_branches)
