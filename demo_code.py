import jax
import jax.numpy as jnp

from manifolds import (
    create_interval_manifold,
    create_circle_manifold,
    create_simplex_manifold,
)
from simulations import simulate_tree_traits
from tree_ops import generate_random_tree


def _constant_drift(value: float):
    def fn(x):
        return jnp.full_like(x, value)

    return fn


def _constant_diffusion(value: float):
    def fn(x):
        d = x.shape[-1]
        return jnp.eye(d) * value

    return fn


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