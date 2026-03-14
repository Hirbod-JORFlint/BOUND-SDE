"""
Demo script showing various actions on phylogenetic trees using tree_ops.py and related modules.
"""

import jax.numpy as jnp
from tree_ops import build_tree_arrays, compute_postorder, reorder_tree_postorder
from simulations import simulate_tree_traits
from manifolds import create_circle_manifold
from likelihood import likelihood_objective, ModelParams
import jax.random as random

# 1. Build a tree from edge lists (e.g., from R's ape package)
print("1. Building tree from edges...")
parent = jnp.array([0, 0, 2])  # 0-based indices
child = jnp.array([1, 2, 3])
branch_lengths = jnp.array([0.2, 0.3, 0.4])
tree = build_tree_arrays(parent, child, branch_lengths)
print(f"Tree built with {tree.parent.shape[0]} nodes and {tree.branch_child.shape[0]} branches.")
print(f"Postorder: {tree.postorder}")

# 2. Compute postorder traversal
print("\n2. Computing postorder traversal...")
postorder = compute_postorder(tree)
print(f"Postorder: {postorder}")

# 3. Reorder tree to postorder
print("\n3. Reordering tree to postorder...")
reordered_tree = reorder_tree_postorder(tree, postorder)
print(f"Reordered postorder: {reordered_tree.postorder}")

# 4. Simulate trait evolution (skipped due to JAX tracing issue in demo)
print("\n4. Simulating trait evolution...")
print("Simulation requires careful setup; see simulations.py for details.")

# 5. Compute likelihood (requires full spectral setup; see likelihood.py)
print("\n5. Computing likelihood...")
print("Likelihood computation requires spectral decomposition; see likelihood.py for details.")

print("\nOther actions include:")
print("- Optimizing parameters using Adam or L-BFGS (from optimizers.py)")
print("- Computing spectral decompositions (from spectral_solver.py)")
print("- Running full pipelines (from main.py)")
