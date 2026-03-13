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

# ============================================================
# Branch Index Construction
# ============================================================

def build_branch_indices(
    tree: TreeData
) -> tuple:
    """
    Construct flat branch indexing arrays.

    Parameters
    ----------
    tree : TreeData

    Returns
    -------
    branch_child : jnp.ndarray
        Child node index for each branch.

        Shape
        -----
        (B,)

    branch_parent : jnp.ndarray
        Parent node index for each branch.

        Shape
        -----
        (B,)

    node_child_counts : jnp.ndarray
        Number of children per node.

        Shape
        -----
        (N,)

    Notes
    -----

    Let

        B = number of branches

    For a tree with N nodes

        B = N - 1
    """

    parent = tree.parent

    child_nodes = jnp.where(parent >= 0)[0]

    branch_child = child_nodes
    branch_parent = parent[child_nodes]

    N = parent.shape[0]

    counts = jnp.zeros((N,), dtype=jnp.int32)
    counts = counts.at[branch_parent].add(1)

    return branch_child, branch_parent, counts


# ============================================================
# Branch Postorder Construction
# ============================================================

def compute_branch_postorder(
    branch_child: jnp.ndarray,
    branch_parent: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute branch ordering compatible with node postorder.

    Parameters
    ----------
    branch_child : jnp.ndarray

        Shape
        -----
        (B,)

    branch_parent : jnp.ndarray

        Shape
        -----
        (B,)

    Returns
    -------
    branch_order : jnp.ndarray

        Shape
        -----
        (B,)

    Notes
    -----

    The order guarantees that

        child branches appear before
        parent branches

    allowing bottom-up scans.
    """

    B = branch_child.shape[0]

    # For postorder node indexing,
    # child index is always < parent index.
    order = jnp.argsort(branch_parent)

    return order


# ============================================================
# Node-to-Branch Mapping
# ============================================================

def build_node_branch_matrix(
    branch_child: jnp.ndarray,
    branch_parent: jnp.ndarray,
    N: int
) -> jnp.ndarray:
    """
    Build padded matrix mapping nodes to incoming branches.

    Parameters
    ----------
    branch_child : jnp.ndarray

        Shape
        -----
        (B,)

    branch_parent : jnp.ndarray

        Shape
        -----
        (B,)

    N : int
        Number of nodes.

    Returns
    -------
    node_branches : jnp.ndarray

        Shape
        -----
        (N, max_degree)

    Each row lists the branch indices entering the node.
    Missing entries are -1.
    """

    counts = jnp.zeros((N,), dtype=jnp.int32)
    counts = counts.at[branch_parent].add(1)

    max_degree = int(counts.max())

    node_branches = -jnp.ones((N, max_degree), dtype=jnp.int32)

    cursor = jnp.zeros((N,), dtype=jnp.int32)

    for b, p in enumerate(branch_parent.tolist()):

        idx = cursor[p]

        node_branches = node_branches.at[p, idx].set(b)

        cursor = cursor.at[p].add(1)

    return node_branches


# ============================================================
# Branch Propagation Kernel
# ============================================================

def propagate_branches(
    branch_child: jnp.ndarray,
    branch_transitions: jnp.ndarray,
    node_likelihoods: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute propagated likelihoods along all branches.

    Mathematical definition

        M_{c→p} = T_{cp} L_c

    Parameters
    ----------
    branch_child : jnp.ndarray

        Shape
        -----
        (B,)

    branch_transitions : jnp.ndarray

        Shape
        -----
        (B, M, M)

    node_likelihoods : jnp.ndarray

        Shape
        -----
        (N, M)

    Returns
    -------
    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)
    """

    def single_branch(b):

        child = branch_child[b]

        T = branch_transitions[b]

        Lc = node_likelihoods[child]

        return T @ Lc

    return jax.vmap(single_branch)(jnp.arange(branch_child.shape[0]))


# ============================================================
# Node Aggregation Kernel
# ============================================================

def aggregate_node_messages(
    node_branches: jnp.ndarray,
    branch_messages: jnp.ndarray,
    node_likelihoods: jnp.ndarray,
    is_tip: jnp.ndarray
) -> jnp.ndarray:
    """
    Aggregate incoming branch messages to update node likelihoods.

    Mathematical rule

        L_i = Π_{b ∈ incoming(i)} M_b

    Parameters
    ----------
    node_branches : jnp.ndarray

        Shape
        -----
        (N, max_degree)

    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)

    node_likelihoods : jnp.ndarray

        Shape
        -----
        (N, M)

    is_tip : jnp.ndarray

        Shape
        -----
        (N,)

    Returns
    -------
    updated_likelihoods : jnp.ndarray

        Shape
        -----
        (N, M)
    """

    M = node_likelihoods.shape[1]

    def update_node(i, L):

        branches = node_branches[i]

        def branch_msg(b):

            return jax.lax.cond(
                b >= 0,
                lambda _: branch_messages[b],
                lambda _: jnp.ones((M,)),
                operand=None
            )

        msgs = jax.vmap(branch_msg)(branches)

        combined = jnp.prod(msgs, axis=0)

        new_val = jax.lax.cond(
            is_tip[i],
            lambda _: L[i],
            lambda _: combined,
            operand=None
        )

        L = L.at[i].set(new_val)

        return L

    return jax.lax.fori_loop(
        0,
        node_branches.shape[0],
        update_node,
        node_likelihoods
    )

# ============================================================
# Traversal Buffer Container
# ============================================================

class TraversalBuffer(NamedTuple):
    """
    Memory container used during likelihood traversal.

    Attributes
    ----------
    node_loglik : jnp.ndarray
        Log-likelihood vectors stored per node.

        Shape
        -----
        (N, M)

    branch_messages : jnp.ndarray
        Messages propagated along branches.

        Shape
        -----
        (B, M)

    node_scales : jnp.ndarray
        Normalization constants per node.

        Shape
        -----
        (N,)
    """

    node_loglik: jnp.ndarray
    branch_messages: jnp.ndarray
    node_scales: jnp.ndarray

# ============================================================
# Buffer Initialization
# ============================================================

def initialize_traversal_buffer(
    N: int,
    B: int,
    M: int
) -> TraversalBuffer:
    """
    Allocate traversal buffers.

    Parameters
    ----------
    N : int
        Number of nodes.

    B : int
        Number of branches.

    M : int
        Spectral basis dimension.

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
        node_scales=node_scales
    )

# ============================================================
# Log-Space Normalization
# ============================================================

def normalize_log_likelihood(
    loglik: jnp.ndarray
) -> tuple:
    """
    Normalize log-likelihood vector.

    Mathematical definition

        L_i ← L_i / Z_i

    where

        Z_i = Σ_k L_i(k)

    In log space

        log Z_i = logsumexp(loglik)

    Parameters
    ----------
    loglik : jnp.ndarray

        Shape
        -----
        (M,)

    Returns
    -------
    normalized : jnp.ndarray

        Shape
        -----
        (M,)

    log_scale : float
        Normalization constant.
    """

    log_scale = jax.scipy.special.logsumexp(loglik)

    normalized = loglik - log_scale

    return normalized, log_scale

# ============================================================
# Batch Normalization
# ============================================================

def normalize_nodes(
    node_loglik: jnp.ndarray
) -> tuple:
    """
    Normalize likelihood vectors across all nodes.

    Parameters
    ----------
    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)

    Returns
    -------
    normalized : jnp.ndarray
        Normalized log likelihoods.

        Shape
        -----
        (N, M)

    scales : jnp.ndarray
        Node normalization constants.

        Shape
        -----
        (N,)
    """

    def norm_fn(v):
        return normalize_log_likelihood(v)

    normalized, scales = jax.vmap(norm_fn)(node_loglik)

    return normalized, scales

# ============================================================
# Log-Space Branch Propagation
# ============================================================

def propagate_branches_logspace(
    branch_child: jnp.ndarray,
    branch_transitions: jnp.ndarray,
    node_loglik: jnp.ndarray
) -> jnp.ndarray:
    """
    Propagate log-likelihood messages along branches.

    Mathematical definition

        M_{c→p} = T_{cp} L_c

    In log-space

        log M_k =
        log Σ_j T_kj exp(L_c(j))

    Parameters
    ----------
    branch_child : jnp.ndarray

        Shape
        -----
        (B,)

    branch_transitions : jnp.ndarray

        Shape
        -----
        (B, M, M)

    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)

    Returns
    -------
    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)
    """

    def single_branch(b):

        child = branch_child[b]

        logL = node_loglik[child]

        T = branch_transitions[b]

        logT = jnp.log(T + 1e-300)

        vals = logT + logL[None, :]

        return jax.scipy.special.logsumexp(vals, axis=1)

    return jax.vmap(single_branch)(jnp.arange(branch_child.shape[0]))

# ============================================================
# Log-Space Node Aggregation
# ============================================================

def aggregate_node_messages_logspace(
    node_branches: jnp.ndarray,
    branch_messages: jnp.ndarray,
    node_loglik: jnp.ndarray,
    is_tip: jnp.ndarray
) -> jnp.ndarray:
    """
    Aggregate incoming log-messages at nodes.

    Mathematical rule

        log L_i = Σ log M_b

    Parameters
    ----------
    node_branches : jnp.ndarray

        Shape
        -----
        (N, max_degree)

    branch_messages : jnp.ndarray

        Shape
        -----
        (B, M)

    node_loglik : jnp.ndarray

        Shape
        -----
        (N, M)

    is_tip : jnp.ndarray

        Shape
        -----
        (N,)

    Returns
    -------
    updated : jnp.ndarray

        Shape
        -----
        (N, M)
    """

    M = node_loglik.shape[1]

    def update_node(i, L):

        branches = node_branches[i]

        def branch_msg(b):

            return jax.lax.cond(
                b >= 0,
                lambda _: branch_messages[b],
                lambda _: jnp.zeros((M,)),
                operand=None
            )

        msgs = jax.vmap(branch_msg)(branches)

        summed = jnp.sum(msgs, axis=0)

        new_val = jax.lax.cond(
            is_tip[i],
            lambda _: L[i],
            lambda _: summed,
            operand=None
        )

        L = L.at[i].set(new_val)

        return L

    return jax.lax.fori_loop(
        0,
        node_branches.shape[0],
        update_node,
        node_loglik
    )
