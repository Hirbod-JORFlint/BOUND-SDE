# ============================================================
# R Bridge Utilities
# ============================================================

import jax
import jax.numpy as jnp
import numpy as np

from typing import Dict

from likelihood import (
    vectorized_neg_loglik,
    ParamShape,
    TreeData,
)

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


def r_get_postorder(parents):
    """
    Compute postorder node order from parent array.

    Parameters
    ----------
    parents : array_like

        Shape
        -----
        (N,)

    Returns
    -------
    postorder : jnp.ndarray
        Shape
        -----
        (N,)
    """

    parents = jnp.array(parents)
    N = parents.shape[0]
    children = [[] for _ in range(N)]

    for idx, p in enumerate(parents.tolist()):
        if p >= 0:
            children[int(p)].append(idx)

    root_indices = jnp.where(parents == -1)[0]
    root = int(root_indices[0]) if root_indices.size > 0 else 0

    visited = [False] * N
    order = []

    def dfs(node):
        if visited[node]:
            return
        visited[node] = True
        for child in children[node]:
            dfs(int(child))
        order.append(node)

    dfs(root)

    return jnp.array(order, dtype=jnp.int32)

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

    params = jnp.array(params)
    parents = jnp.array(parents)
    branch_lengths = jnp.array(branch_lengths)
    node_loglik = jnp.array(node_loglik)

    N = parents.shape[0]
    postorder = jnp.arange(N)
    root_index = int(jnp.where(parents == -1)[0][0]) if jnp.any(parents == -1) else 0
    tree = TreeData(
        postorder_nodes=postorder,
        parent_index=parents,
        parent_branch=jnp.arange(N),
        branch_lengths=branch_lengths,
        root_index=root_index,
    )
    shape = ParamShape(parents.shape[0], 0, 0, node_loglik.shape[-1])

    val = vectorized_neg_loglik(
        params,
        shape,
        tree,
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

