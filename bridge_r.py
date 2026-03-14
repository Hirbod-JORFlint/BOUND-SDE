# ============================================================
# R Bridge Utilities
# ============================================================

import jax
import jax.numpy as jnp
import numpy as np

from typing import Dict

# ============================================================
# Convert R phylo tree to internal arrays
# ============================================================

def r_tree_to_arrays(edge, edge_length):
    """
    Convert R phylo tree structure into JAX arrays.

    Parameters
    ----------
    edge : array_like
        Edge matrix from R.

        Shape
        -----
        (E, 2)

        Format
        ------
        parent → child indices.

    edge_length : array_like

        Shape
        -----
        (E,)

    Returns
    -------
    tree : dict

        parents : jnp.ndarray
            Shape
            -----
            (N,)

        branch_lengths : jnp.ndarray
            Shape
            -----
            (N,)
    """

    edge = np.asarray(edge)
    edge_length = np.asarray(edge_length)

    parent = edge[:, 0]
    child = edge[:, 1]

    N = int(np.max(edge))

    parents = np.zeros(N + 1, dtype=np.int32)
    branch_lengths = np.zeros(N + 1)

    for p, c, l in zip(parent, child, edge_length):
        parents[c] = p
        branch_lengths[c] = l

    return {
        "parents": jnp.array(parents),
        "branch_lengths": jnp.array(branch_lengths),
    }

# ============================================================
# Convert R trait matrix
# ============================================================

def r_traits_to_jax(traits):
    """
    Convert R trait matrix to JAX array.

    Parameters
    ----------
    traits : array_like

        Shape
        -----
        (N, d)

    Returns
    -------
    traits_jax : jnp.ndarray
    """

    traits_np = np.asarray(traits)

    return jnp.array(traits_np)

# ============================================================
# R-accessible likelihood function
# ============================================================

def r_compute_likelihood(
    params,
    parents,
    branch_lengths,
    node_loglik,
    manifold,
    spectral_dim,
):
    """
    Compute phylogenetic likelihood from R.

    Parameters
    ----------
    params : array_like

        Flattened parameter vector.

        Shape
        -----
        (D,)

    parents : array_like
        Tree parent array.

    branch_lengths : array_like

    node_loglik : array_like

        Shape
        -----
        (N, M)

    manifold : object

    spectral_dim : int

    Returns
    -------
    loglik : float
    """

    from likelihood import vectorized_neg_loglik

    params = jnp.array(params)

    parents = jnp.array(parents)
    branch_lengths = jnp.array(branch_lengths)
    node_loglik = jnp.array(node_loglik)

    val = vectorized_neg_loglik(
        params,
        None,
        parents,
        branch_lengths,
        node_loglik,
        manifold,
        spectral_dim,
    )

    return float(-val)

# ============================================================
# R-accessible simulation
# ============================================================

def r_simulate_traits(
    key_seed,
    root_state,
    parents,
    branch_lengths,
    topo_order,
    drift_fn,
    diffusion_fn,
    manifold,
    dt,
):
    """
    Simulate phylogenetic trait evolution from R.

    Returns
    -------
    traits : numpy.ndarray

        Shape
        -----
        (N, d)
    """

    from simulations import simulate_tree_traits

    key = jax.random.PRNGKey(key_seed)

    traits = simulate_tree_traits(
        key,
        jnp.array(root_state),
        jnp.array(parents),
        jnp.array(branch_lengths),
        jnp.array(topo_order),
        drift_fn,
        diffusion_fn,
        manifold,
        dt,
    )

    return np.array(traits)

# ============================================================
# JAX → NumPy conversion
# ============================================================

def to_numpy(x):
    """
    Convert JAX array to NumPy array for R.

    Parameters
    ----------
    x : jnp.ndarray

    Returns
    -------
    numpy.ndarray
    """

    return np.asarray(x)

