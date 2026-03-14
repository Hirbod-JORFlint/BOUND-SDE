import jax
import jax.numpy as jnp

from manifolds import (
    create_interval_manifold,
    create_circle_manifold,
    create_simplex_manifold,
)
from simulations import simulate_tree_traits
from tree_ops import generate_random_tree
from optimizers import run_adam, run_lbfgs
from pruning import (
    compute_branch_transitions,
    propagate_branch_likelihood,
    normalize_all_nodes,
    compute_root_loglik,
)

def _constant_drift(value: float):
    def fn(x):
        return jnp.full_like(x, value)

    return fn


def _constant_diffusion(value: float):
    def fn(x):
        d = x.shape[-1]
        return jnp.eye(d) * value

    return fn

def demo_optimizers():
    """
    Demonstrate `run_adam` and `run_lbfgs` on a simple quadratic.
    """

    def objective(params):
        # Simple convex quadratic: f(x) = sum((x - 1)^2)
        loss = jnp.sum((params - 1.0) ** 2)
        grad = 2.0 * (params - 1.0)
        return loss, grad

    init = jnp.array([3.0, -2.0, 0.5])

    # Adam optimization
    adam_params, adam_losses = run_adam(init, objective, num_steps=50)
    print("Adam final params:", adam_params)
    print("Adam final loss:", float(adam_losses[-1]))

    # L-BFGS optimization
    lbfgs_params, lbfgs_losses = run_lbfgs(init, objective, num_steps=20)
    print("L-BFGS final params:", lbfgs_params)
    print("L-BFGS final loss:", float(lbfgs_losses[-1]))


def demo_pruning():
    """
    Demonstrate low-level pruning utilities on a tiny tree.
    """

    # Two-node tree: root (0) -> child (1)
    parents = jnp.array([-1, 0], dtype=jnp.int32)
    branch_lengths = jnp.array([0.0, 0.5], dtype=jnp.float32)

    # Dummy node log-likelihoods (N=2, M=2)
    node_loglik = jnp.array([[0.0, 0.0], [0.1, -0.2]])

    # Simple 2D eigen system (identity eigenbasis)
    eigenvalues = jnp.array([-1.0, -2.0])
    eigenvectors = jnp.eye(2)

    # Single non-root branch (index 1)
    transitions = compute_branch_transitions(
        eigenvalues,
        branch_lengths[1:],  # shape (1,)
    )  # shape (1, 2, 2)

    # Propagate likelihood from child (node 1) to parent (node 0)
    child_loglik = node_loglik[1]
    transition = transitions[0]
    message = propagate_branch_likelihood(child_loglik, transition)

    # Update parent node and normalize both nodes
    updated_node_loglik = node_loglik.at[0].set(node_loglik[0] + message)
    normalized, scales = normalize_all_nodes(updated_node_loglik)

    # Uniform prior at root
    root_prior = jnp.array([0.5, 0.5])
    root_index = int(jnp.where(parents == -1)[0][0])
    loglik = compute_root_loglik(normalized, scales, root_index, root_prior)

    print("Pruning demo log-likelihood:", float(loglik))


if __name__ == "__main__":
    # Choose a manifold; here we use a 3D simplex as in the README.
    # You can switch to `create_interval_manifold` or `create_circle_manifold`
    # if desired.

    # Bounded interval [0, 1]
    # manifold = create_interval_manifold(L=0.0, U=1.0)

    # Circle (periodic)
    # manifold = create_circle_manifold()

    # Simplex (probability distribution)
    manifold = create_simplex_manifold(dplus1=3)  # 3D simplex

    # Generate a random rooted tree
    num_taxa = 20
    parents, branch_lengths, topo_order = generate_random_tree(
        num_taxa=num_taxa,
        seed=42,
    )

    # Root state consistent with manifold dimension
    dim = manifold.dimension
    root_state = jnp.full((dim,), 1.0 / dim, dtype=jnp.float32)

    # SDE parameters
    drift_value = 0.5
    diffusion_value = 0.3
    dt = 0.01

    drift_fn = _constant_drift(drift_value)
    diffusion_fn = _constant_diffusion(diffusion_value)

    key = jax.random.PRNGKey(0)

    traits = simulate_tree_traits(
        key,
        root_state,
        parents,
        branch_lengths,
        topo_order,
        drift_fn,
        diffusion_fn,
        manifold,
        dt,
    )
    print(traits)
    print("Simulated traits shape:", traits.shape)

    demo_optimizers()
    demo_pruning()