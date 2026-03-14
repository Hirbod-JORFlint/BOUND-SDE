"""Tree utilities for converting edge lists into JAX-ready buffers and traversals."""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class TreeData(NamedTuple):
    """
    Static representation of a rooted phylogenetic tree.

    Attributes
    ----------
    parent : jnp.ndarray
        Parent node index for each node (root has -1).

        Shape
        -----
        (N,)

    children : jnp.ndarray
        Padded children lists (missing entries contain -1).

        Shape
        -----
        (N, max_degree)

    branch_length : jnp.ndarray
        Length of branch connecting node to its parent (root has 0).

        Shape
        -----
        (N,)

    is_tip : jnp.ndarray
        Boolean mask for tip nodes (no children).

        Shape
        -----
        (N,)

    postorder : jnp.ndarray
        Post-order traversal of node indices (children before parents).

        Shape
        -----
        (N,)

    branch_child : jnp.ndarray
        Child node index for every branch.

        Shape
        -----
        (B,)

    branch_parent : jnp.ndarray
        Parent node index for every branch.

        Shape
        -----
        (B,)

    node_branches : jnp.ndarray
        Branch indices entering each node (padded with -1).

        Shape
        -----
        (N, max_degree)
    """

    parent: jnp.ndarray
    children: jnp.ndarray
    branch_length: jnp.ndarray
    is_tip: jnp.ndarray
    postorder: jnp.ndarray
    branch_child: jnp.ndarray
    branch_parent: jnp.ndarray
    node_branches: jnp.ndarray


class TraversalBuffer(NamedTuple):
    """
    Buffers used during pruning traversals.
    """

    node_loglik: jnp.ndarray
    branch_messages: jnp.ndarray
    node_scales: jnp.ndarray


def build_branch_indices(tree: TreeData) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Build branch indexing arrays from a tree structure.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    branch_child : jnp.ndarray
        Shape (B,)

    branch_parent : jnp.ndarray
        Shape (B,)

    node_branches : jnp.ndarray
        Shape (N, max_degree)
    """

    parent = tree.parent
    child_nodes = jnp.where(parent >= 0)[0]
    branch_child = child_nodes
    branch_parent = parent[child_nodes]

    N = parent.shape[0]
    counts = jnp.zeros((N,), dtype=jnp.int32)
    counts = counts.at[branch_parent].add(1)
    max_degree = int(counts.max()) if int(counts.max()) > 0 else 1

    node_branches = -jnp.ones((N, max_degree), dtype=jnp.int32)
    cursor = jnp.zeros((N,), dtype=jnp.int32)
    for idx, parent_idx in enumerate(branch_parent.tolist()):  # noqa: B007
        slot = int(cursor[parent_idx])
        node_branches = node_branches.at[parent_idx, slot].set(idx)
        cursor = cursor.at[parent_idx].add(1)

    return branch_child.astype(jnp.int32), branch_parent.astype(jnp.int32), node_branches


def compute_postorder(tree: TreeData) -> jnp.ndarray:
    r"""
    Compute a post-order traversal of the tree nodes.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    order : jnp.ndarray
        Shape (N,)
    """

    children = tree.children
    N = children.shape[0]
    visited = [False] * N
    order = []

    roots = jnp.where(tree.parent == -1)[0]
    root = int(roots[0]) if roots.size > 0 else 0

    def dfs(node: int) -> None:
        if visited[node]:
            return
        visited[node] = True
        for child in children[node]:
            if child >= 0:
                dfs(int(child))
        order.append(node)

    dfs(root)

    return jnp.array(order, dtype=jnp.int32)


def build_tree_arrays(parent: jnp.ndarray, child: jnp.ndarray, branch_length: jnp.ndarray) -> TreeData:
    r"""
    Convert edge lists into padded tree arrays and traversal buffers.

    Parameters
    ----------
    parent : jnp.ndarray
        Parent indices for each edge.

        Shape
        -----
        (E,)

    child : jnp.ndarray
        Child indices for each edge.

        Shape
        -----
        (E,)

    branch_length : jnp.ndarray
        Branch lengths per edge.

        Shape
        -----
        (E,)

    Returns
    -------
    tree : TreeData
    """

    parent = jnp.asarray(parent, dtype=jnp.int32)
    child = jnp.asarray(child, dtype=jnp.int32)
    branch_length = jnp.asarray(branch_length, dtype=jnp.float32)

    max_node = int(jnp.maximum(parent.max(), child.max()))
    N = max_node + 1

    parent_array = -jnp.ones((N,), dtype=jnp.int32)
    branch_array = jnp.zeros((N,), dtype=branch_length.dtype)

    for p, c, length in zip(parent.tolist(), child.tolist(), branch_length.tolist()):
        parent_array = parent_array.at[c].set(p)
        branch_array = branch_array.at[c].set(length)

    children_lists = [[] for _ in range(N)]
    for p, c in zip(parent.tolist(), child.tolist()):
        children_lists[p].append(c)

    max_degree = max(len(lst) for lst in children_lists) or 1
    children_matrix = -jnp.ones((N, max_degree), dtype=jnp.int32)
    for idx, lst in enumerate(children_lists):
        for slot, c in enumerate(lst):
            children_matrix = children_matrix.at[idx, slot].set(c)

    is_tip = jnp.array([len(lst) == 0 for lst in children_lists], dtype=jnp.bool_)

    temp_tree = TreeData(
        parent=parent_array,
        children=children_matrix,
        branch_length=branch_array,
        is_tip=is_tip,
        postorder=jnp.arange(N, dtype=jnp.int32),
        branch_child=jnp.zeros((0,), dtype=jnp.int32),
        branch_parent=jnp.zeros((0,), dtype=jnp.int32),
        node_branches=jnp.zeros((N, 0), dtype=jnp.int32),
    )

    postorder = compute_postorder(temp_tree)
    branch_child, branch_parent, node_branches = build_branch_indices(temp_tree)

    return TreeData(
        parent=parent_array,
        children=children_matrix,
        branch_length=branch_array,
        is_tip=is_tip,
        postorder=postorder,
        branch_child=branch_child,
        branch_parent=branch_parent,
        node_branches=node_branches,
    )


def reorder_tree_postorder(tree: TreeData, order: jnp.ndarray) -> TreeData:
    r"""
    Reindex all tree arrays to follow a post-order numbering.

    Parameters
    ----------
    tree : TreeData

    order : jnp.ndarray
        Post-order permutation (children precede parents).

        Shape
        -----
        (N,)

    Returns
    -------
    reordered_tree : TreeData
    """

    N = order.shape[0]
    mapping = -jnp.ones((N,), dtype=jnp.int32)
    mapping = mapping.at[order].set(jnp.arange(N, dtype=jnp.int32))

    parent = jnp.where(tree.parent >= 0, mapping[tree.parent], -1)
    children = tree.children[order]
    children = jnp.where(children >= 0, mapping[children], -1)
    branch_length = tree.branch_length[order]
    is_tip = tree.is_tip[order]

    partial = TreeData(
        parent=parent,
        children=children,
        branch_length=branch_length,
        is_tip=is_tip,
        postorder=order,
        branch_child=jnp.zeros((0,), dtype=jnp.int32),
        branch_parent=jnp.zeros((0,), dtype=jnp.int32),
        node_branches=jnp.zeros((N, 0), dtype=jnp.int32),
    )

    branch_child, branch_parent, node_branches = build_branch_indices(partial)

    return TreeData(
        parent=parent,
        children=children,
        branch_length=branch_length,
        is_tip=is_tip,
        postorder=jnp.arange(N, dtype=jnp.int32),
        branch_child=branch_child,
        branch_parent=branch_parent,
        node_branches=node_branches,
    )


def initialize_traversal_buffer(N: int, B: int, M: int) -> TraversalBuffer:
    r"""
    Create zero-initialized traversal buffers.

    Parameters
    ----------
    N : int
        Number of nodes.

    B : int
        Number of branches.

    M : int
        Spectral dimension.

    Returns
    -------
    buffer : TraversalBuffer
    """

    node_loglik = jnp.zeros((N, M))
    branch_messages = jnp.zeros((B, M))
    node_scales = jnp.zeros((N,))

    return TraversalBuffer(
        node_loglik=node_loglik,
        branch_messages=branch_messages,
        node_scales=node_scales,
    )


def normalize_log_likelihood(loglik: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Normalize log-likelihood vectors via log-sum-exp.

    Parameters
    ----------
    loglik : jnp.ndarray
        Log-likelihood array.

        Shape
        -----
        (..., M)

    Returns
    -------
    normalized : jnp.ndarray
        Log-likelihoods shifted to sum to one.

        Shape
        -----
        (..., M)

    scales : jnp.ndarray
        Log-scaling constants used in normalization.

        Shape
        -----
        (...)
    """

    shift = jsp.special.logsumexp(loglik, axis=-1, keepdims=True)
    normalized = loglik - shift
    scales = jnp.squeeze(shift, axis=-1)
    return normalized, scales


def normalize_all_nodes(node_loglik: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Apply `normalize_log_likelihood` row-wise for all nodes.

    Parameters
    ----------
    node_loglik : jnp.ndarray
        Node log-likelihood matrix.

        Shape
        -----
        (N, M)

    Returns
    -------
    normalized : jnp.ndarray
        Normalized log-likelihood matrix.

        Shape
        -----
        (N, M)

    scales : jnp.ndarray
        Scale per node.

        Shape
        -----
        (N,)
    """

    normalized, scales = jax.vmap(normalize_log_likelihood)(node_loglik)
    return normalized, scales


def generate_random_tree(
    num_taxa: int,
    seed: int = 0,
    min_branch_length: float = 0.05,
    max_branch_length: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Generate a simple random rooted tree topology for simulations.

    The tree is constructed incrementally: starting from a single root node,
    each new node attaches to a uniformly chosen existing node. This ensures
    an acyclic rooted tree where parents always have lower indices than
    their children.

    Parameters
    ----------
    num_taxa : int
        Number of nodes in the tree (including the root).

    seed : int, optional
        Seed for the PRNG used to sample topology and branch lengths.

    min_branch_length : float, optional
        Lower bound for uniform branch length sampling.

    max_branch_length : float, optional
        Upper bound for uniform branch length sampling.

    Returns
    -------
    parents : jnp.ndarray
        Parent index for each node (root has -1).

        Shape
        -----
        (N,)

    branch_lengths : jnp.ndarray
        Branch length associated with each node's connection to its parent.
        The root has length 0.0.

        Shape
        -----
        (N,)

    topo_order : jnp.ndarray
        A valid topological ordering of the nodes (parents precede children).
        For this construction this is simply ``jnp.arange(N)``.

        Shape
        -----
        (N,)
    """

    if num_taxa <= 0:
        raise ValueError("num_taxa must be positive")

    key = jax.random.PRNGKey(seed)

    # Parents: root has parent -1, subsequent nodes attach to any previous node.
    parents_list = [-1]
    lengths_list = [0.0]

    for i in range(1, num_taxa):
        key, key_parent, key_length = jax.random.split(key, 3)
        parent_idx = int(jax.random.randint(key_parent, shape=(), minval=0, maxval=i))
        length = float(
            jax.random.uniform(
                key_length,
                shape=(),
                minval=min_branch_length,
                maxval=max_branch_length,
            )
        )
        parents_list.append(parent_idx)
        lengths_list.append(length)

    parents = jnp.array(parents_list, dtype=jnp.int32)
    branch_lengths = jnp.array(lengths_list, dtype=jnp.float32)
    topo_order = jnp.arange(num_taxa, dtype=jnp.int32)

    return parents, branch_lengths, topo_order


def get_root_index(tree: TreeData) -> int:
    r"""
    Return the root node index for a `TreeData` object.

    The root is defined as the unique node with parent -1. If multiple
    candidates exist, the smallest index is returned.
    """

    roots = jnp.where(tree.parent == -1)[0]
    return int(roots[0]) if roots.size > 0 else 0


def get_tip_indices(tree: TreeData) -> jnp.ndarray:
    r"""
    Return the indices of tip (leaf) nodes.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    tips : jnp.ndarray
        Indices where ``tree.is_tip`` is True.

        Shape
        -----
        (T,)
    """

    return jnp.where(tree.is_tip)[0].astype(jnp.int32)


def count_tips(tree: TreeData) -> int:
    r"""
    Count the number of tip (leaf) nodes in the tree.
    """

    return int(tree.is_tip.sum())


def tree_size(tree: TreeData) -> int:
    r"""
    Return the number of nodes in the tree.
    """

    return int(tree.parent.shape[0])


def num_branches(tree: TreeData) -> int:
    r"""
    Return the number of branches (non-root edges) in the tree.
    """

    return int(tree.branch_child.shape[0])


def extract_edge_list(tree: TreeData) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""
    Extract parent, child, and branch length arrays from a `TreeData` object.

    This is the inverse of `build_tree_arrays` up to node ordering.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    parent : jnp.ndarray
        Parent index for each edge.

        Shape
        -----
        (E,)

    child : jnp.ndarray
        Child index for each edge.

        Shape
        -----
        (E,)

    branch_length : jnp.ndarray
        Branch length for each edge.

        Shape
        -----
        (E,)
    """

    child = tree.branch_child.astype(jnp.int32)
    parent = tree.branch_parent.astype(jnp.int32)
    branch_length = tree.branch_length[child]
    return parent, child, branch_length
