# ============================================================
# JAX-Compatible Phylogenetic Tree Operations
# ============================================================

from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp


# ============================================================
# Tree Container
# ============================================================

class TreeData(NamedTuple):
    """
    Static representation of a phylogenetic tree.

    Attributes
    ----------
    parent : jnp.ndarray
        Parent index for each node.

        Shape
        -----
        (N,)

    children : jnp.ndarray
        Padded matrix containing children of each node.

        Shape
        -----
        (N, max_degree)

        Missing entries are -1.

    branch_length : jnp.ndarray
        Length of branch from node to parent.

        Shape
        -----
        (N,)

    is_tip : jnp.ndarray
        Boolean mask identifying tip nodes.

        Shape
        -----
        (N,)
    """

    parent: jnp.ndarray
    children: jnp.ndarray
    branch_length: jnp.ndarray
    is_tip: jnp.ndarray


# ============================================================
# Tree Construction
# ============================================================

def build_tree_arrays(
    parent: jnp.ndarray,
    child: jnp.ndarray,
    branch_length: jnp.ndarray
) -> TreeData:
    """
    Convert edge-list representation into static JAX tree arrays.

    Parameters
    ----------
    parent : jnp.ndarray
        Parent node indices.

        Shape
        -----
        (E,)

    child : jnp.ndarray
        Child node indices.

        Shape
        -----
        (E,)

    branch_length : jnp.ndarray
        Branch length for each edge.

        Shape
        -----
        (E,)

    Returns
    -------
    tree : TreeData
        JAX-compatible tree structure.

    Notes
    -----

    Let

        N = number of nodes

    The function constructs:

        parent        (N,)
        children      (N, max_degree)
        branch_length (N,)
        is_tip        (N,)

    where missing children entries are filled with -1.
    """

    parent = jnp.asarray(parent)
    child = jnp.asarray(child)
    branch_length = jnp.asarray(branch_length)

    N = int(jnp.maximum(parent.max(), child.max()) + 1)

    parent_array = -jnp.ones((N,), dtype=jnp.int32)
    branch_array = jnp.zeros((N,))

    parent_array = parent_array.at[child].set(parent)
    branch_array = branch_array.at[child].set(branch_length)

    # Determine degree of each node
    degree = jnp.zeros((N,), dtype=jnp.int32)
    degree = degree.at[parent].add(1)

    max_degree = int(degree.max())

    children_matrix = -jnp.ones((N, max_degree), dtype=jnp.int32)

    # Fill children slots
    child_counter = jnp.zeros((N,), dtype=jnp.int32)

    for p, c in zip(parent.tolist(), child.tolist()):
        idx = child_counter[p]
        children_matrix = children_matrix.at[p, idx].set(c)
        child_counter = child_counter.at[p].add(1)

    # Tip nodes
    is_tip = degree == 0

    return TreeData(
        parent=parent_array,
        children=children_matrix,
        branch_length=branch_array,
        is_tip=is_tip
    )


# ============================================================
# Postorder Traversal
# ============================================================

def compute_postorder(tree: TreeData) -> jnp.ndarray:
    """
    Compute post-order traversal ordering.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    order : jnp.ndarray

        Shape
        -----
        (N,)

    Notes
    -----

    Postorder ensures

        child_index < parent_index

    which enables bottom-up traversal with `lax.scan`.
    """

    children = tree.children
    N = children.shape[0]

    visited = set()
    order = []

    def dfs(node):

        if node in visited:
            return

        visited.add(node)

        for c in children[node]:
            if c >= 0:
                dfs(int(c))

        order.append(node)

    root = int(jnp.where(tree.parent == -1)[0][0])

    dfs(root)

    return jnp.array(order, dtype=jnp.int32)


# ============================================================
# Tree Reordering
# ============================================================

def reorder_tree_postorder(
    tree: TreeData,
    order: jnp.ndarray
) -> TreeData:
    """
    Reindex tree nodes according to postorder traversal.

    Parameters
    ----------
    tree : TreeData
    order : jnp.ndarray

        Shape
        -----
        (N,)

    Returns
    -------
    reordered_tree : TreeData

    Notes
    -----

    Guarantees

        child_index < parent_index

    enabling efficient bottom-up computation.
    """

    N = order.shape[0]

    new_index = -jnp.ones((N,), dtype=jnp.int32)
    new_index = new_index.at[order].set(jnp.arange(N))

    parent = tree.parent
    children = tree.children

    parent_new = jnp.where(parent >= 0, new_index[parent], -1)

    children_new = jnp.where(children >= 0, new_index[children], -1)

    branch_length = tree.branch_length
    is_tip = tree.is_tip

    return TreeData(
        parent=parent_new,
        children=children_new,
        branch_length=branch_length,
        is_tip=is_tip
    )


# ============================================================
# Branch Transition Stack
# ============================================================

def build_branch_transition_stack(
    branch_lengths: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
    branch_transition_fn: Callable
) -> jnp.ndarray:
    """
    Compute spectral transition matrices for every branch.

    Parameters
    ----------
    branch_lengths : jnp.ndarray

        Shape
        -----
        (N,)

    eigenvalues : jnp.ndarray

        Shape
        -----
        (M,)

    eigenvectors : jnp.ndarray

        Shape
        -----
        (M, M)

    branch_transition_fn : Callable
        Function computing spectral transition matrices.

    Returns
    -------
    T : jnp.ndarray

        Shape
        -----
        (N, M, M)
    """

    compute = lambda t: branch_transition_fn(t, eigenvalues, eigenvectors)

    return jax.vmap(compute)(branch_lengths)


# ============================================================
# Tip Likelihood Initialization
# ============================================================

def initialize_tip_states(
    tip_values: jnp.ndarray,
    basis_fn: Callable
) -> jnp.ndarray:
    """
    Convert observed trait values into spectral likelihood vectors.

    Parameters
    ----------
    tip_values : jnp.ndarray

        Shape
        -----
        (N,)

    basis_fn : Callable
        Spectral basis evaluation function.

    Returns
    -------
    L : jnp.ndarray

        Shape
        -----
        (N, M)

    where M is spectral basis size.
    """

    return basis_fn(tip_values)


# ============================================================
# Postorder Tree Scan
# ============================================================

def tree_postorder_scan(
    tree: TreeData,
    branch_transitions: jnp.ndarray,
    node_likelihoods: jnp.ndarray
) -> jnp.ndarray:
    """
    Bottom-up traversal computing node likelihood vectors.

    Parameters
    ----------
    tree : TreeData

    branch_transitions : jnp.ndarray

        Shape
        -----
        (N, M, M)

    node_likelihoods : jnp.ndarray

        Shape
        -----
        (N, M)

    Returns
    -------
    likelihoods : jnp.ndarray

        Shape
        -----
        (N, M)

    Notes
    -----

    For node i

        L_i = Π_c (T_ic @ L_c)

    where

        T_ic = transition matrix for branch (i,c)
    """

    children = tree.children
    N = children.shape[0]
    M = node_likelihoods.shape[1]

    def scan_step(carry, i):

        L = carry

        child_nodes = children[i]

        def process_child(c):

            def valid_child():

                T = branch_transitions[c]
                Lc = L[c]
                return T @ Lc

            return jax.lax.cond(
                c >= 0,
                lambda _: valid_child(),
                lambda _: jnp.ones((M,)),
                operand=None
            )

        child_contrib = jax.vmap(process_child)(child_nodes)

        combined = jnp.prod(child_contrib, axis=0)

        L = L.at[i].set(
            jax.lax.cond(
                tree.is_tip[i],
                lambda _: L[i],
                lambda _: combined,
                operand=None
            )
        )

        return L, None

    final_L, _ = jax.lax.scan(
        scan_step,
        node_likelihoods,
        jnp.arange(N)
    )

    return final_L
