import jax.numpy as jnp

from tree_ops import (
    build_tree_arrays,
    compute_postorder,
    reorder_tree_postorder,
    build_branch_indices,
    initialize_traversal_buffer,
    normalize_log_likelihood,
)


def test_build_tree_arrays_and_postorder():
    parent = jnp.array([0, 0, 2])
    child = jnp.array([1, 2, 3])
    lengths = jnp.array([0.2, 0.3, 0.4])
    tree = build_tree_arrays(parent, child, lengths)
    assert tree.parent.tolist() == [-1, 0, 0, 2]
    assert tree.branch_child.tolist() == [1, 2, 3]
    assert tree.branch_parent.tolist() == [0, 0, 2]
    assert tree.node_branches.shape[0] == 4
    order = tree.postorder
    assert order[0] in (1, 3)
    assert order[-1] == 0


def test_reorder_preserves_structure():
    parent = jnp.array([0, 0, 2])
    child = jnp.array([1, 2, 3])
    lengths = jnp.array([0.2, 0.3, 0.4])
    tree = build_tree_arrays(parent, child, lengths)
    reordered = reorder_tree_postorder(tree, tree.postorder)
    assert jnp.all(reordered.parent >= -1)
    assert reordered.postorder.tolist() == [0, 1, 2, 3]


def test_build_branch_indices_returns_padding():
    parent = jnp.array([0, 0, 2])
    child = jnp.array([1, 2, 3])
    lengths = jnp.array([0.2, 0.3, 0.4])
    tree = build_tree_arrays(parent, child, lengths)
    branch_child, branch_parent, node_branches = build_branch_indices(tree)
    assert branch_child.shape[0] == 3
    assert node_branches.shape[0] == 4


def test_traversal_buffer_and_normalization():
    buffer = initialize_traversal_buffer(N=4, B=3, M=5)
    assert buffer.node_loglik.shape == (4, 5)
    assert buffer.branch_messages.shape == (3, 5)
    normalized, scales = normalize_log_likelihood(
        jnp.array([[0.0, 0.0], [1.0, -1.0]])
    )
    assert normalized.shape == (2, 2)
    assert scales.shape == (2,)
