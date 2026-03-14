# ============================================================
# SDE Simulation Engine
# ============================================================

import jax
import jax.numpy as jnp

# ============================================================
# Single Euler–Maruyama Step on Manifold
# ============================================================

def simulate_sde_step(
    key,
    x,
    drift_fn,
    diffusion_fn,
    manifold,
    dt
):
    """
    Perform one Euler–Maruyama step for the SDE

        dX_t = μ(X_t) dt + σ(X_t) dW_t

    with projection onto the manifold.

    Parameters
    ----------
    key : jax.random.PRNGKey

    x : jnp.ndarray
        Current state.

        Shape
        -----
        (d,)

    drift_fn : callable
        Drift function μ(x).

        Input
        -----
        (d,)

        Output
        ------
        (d,)

    diffusion_fn : callable
        Diffusion matrix σ(x).

        Output
        ------
        (d, d)

    manifold : object
        Manifold instance providing projection operator.

    dt : float
        Time step.

    Returns
    -------
    x_next : jnp.ndarray

        Shape
        -----
        (d,)
    """

    mu = drift_fn(x)

    sigma = diffusion_fn(x)

    eps = jax.random.normal(key, shape=x.shape)

    diffusion_term = jnp.sqrt(dt) * (sigma @ eps)

    x_next = x + mu * dt + diffusion_term

    # project back onto manifold
    x_next = manifold.project(x_next)

    return x_next

# ============================================================
# Scan Step Kernel
# ============================================================

def _scan_sde_step(
    carry,
    key,
    drift_fn,
    diffusion_fn,
    manifold,
    dt
):
    """
    Internal scan step.

    Returns
    -------
    new_state, output
    """

    x = carry

    x_next = simulate_sde_step(
        key,
        x,
        drift_fn,
        diffusion_fn,
        manifold,
        dt
    )

    return x_next, x_next

# ============================================================
# Simulate SDE Path
# ============================================================

def simulate_sde_path(
    key,
    x0,
    drift_fn,
    diffusion_fn,
    manifold,
    dt,
    steps
):
    """
    Simulate trajectory of SDE.

    Mathematical definition
    -----------------------

    X_{t+1}
    =
    Π_M(
        X_t
        +
        μ(X_t) dt
        +
        σ(X_t) √dt ε
    )

    Parameters
    ----------
    key : jax.random.PRNGKey

    x0 : jnp.ndarray
        Initial state.

        Shape
        -----
        (d,)

    drift_fn : callable

    diffusion_fn : callable

    manifold : object

    dt : float

    steps : int

    Returns
    -------
    path : jnp.ndarray

        Shape
        -----
        (steps, d)
    """

    keys = jax.random.split(key, steps)

    def step(carry, k):

        x_next = simulate_sde_step(
            k,
            carry,
            drift_fn,
            diffusion_fn,
            manifold,
            dt
        )

        return x_next, x_next

    _, path = jax.lax.scan(
        step,
        x0,
        keys
    )

    return path

# ============================================================
# Batched SDE Simulation
# ============================================================

def simulate_sde_batch(
    key,
    x0_batch,
    drift_fn,
    diffusion_fn,
    manifold,
    dt,
    steps
):
    """
    Simulate multiple trajectories in parallel.

    Parameters
    ----------
    key : PRNGKey

    x0_batch : jnp.ndarray

        Shape
        -----
        (B, d)

    Returns
    -------
    paths : jnp.ndarray

        Shape
        -----
        (B, steps, d)
    """

    B = x0_batch.shape[0]

    keys = jax.random.split(key, B)

    simulate_single = lambda k, x: simulate_sde_path(
        k,
        x,
        drift_fn,
        diffusion_fn,
        manifold,
        dt,
        steps
    )

    return jax.vmap(simulate_single)(keys, x0_batch)

# ============================================================
# Branch Propagation
# ============================================================

def simulate_branch(
    key,
    x_parent,
    branch_length,
    drift_fn,
    diffusion_fn,
    manifold,
    dt
):
    """
    Simulate trait evolution along one branch.

    Mathematical definition
    -----------------------

    X_child = Φ_{τ}(X_parent)

    where Φ is the SDE flow map.

    Parameters
    ----------
    key : PRNGKey

    x_parent : jnp.ndarray
        Parent state.

        Shape
        -----
        (d,)

    branch_length : float

    drift_fn : callable

    diffusion_fn : callable

    manifold : object

    dt : float

    Returns
    -------
    x_child : jnp.ndarray

        Shape
        -----
        (d,)
    """

    steps = max(1, int(jnp.ceil(float(branch_length) / dt)))

    path = simulate_sde_path(
        key,
        x_parent,
        drift_fn,
        diffusion_fn,
        manifold,
        dt,
        steps
    )

    return path[-1]

# ============================================================
# Tree Trait Simulation
# ============================================================

def simulate_tree_traits(
    key,
    root_state,
    parents,
    branch_lengths,
    topo_order,
    drift_fn,
    diffusion_fn,
    manifold,
    dt
):
    """
    Simulate trait evolution across a phylogenetic tree.

    Parameters
    ----------
    key : PRNGKey

    root_state : jnp.ndarray

        Shape
        -----
        (d,)

    parents : jnp.ndarray

        Shape
        -----
        (N,)

    branch_lengths : jnp.ndarray

        Shape
        -----
        (N,)

    topo_order : jnp.ndarray

        Topological node ordering.

        Shape
        -----
        (N,)

    Returns
    -------
    traits : jnp.ndarray

        Shape
        -----
        (N, d)
    """

    N = parents.shape[0]
    d = root_state.shape[0]

    traits = jnp.zeros((N, d))

    traits = traits.at[0].set(root_state)

    keys = jax.random.split(key, N)

    def step(traits, node):

        parent = parents[node]

        x_parent = traits[parent]

        key = keys[node]

        x_child = simulate_branch(
            key,
            x_parent,
            branch_lengths[node],
            drift_fn,
            diffusion_fn,
            manifold,
            dt
        )

        traits = traits.at[node].set(x_child)

        return traits, None

    traits, _ = jax.lax.scan(
        step,
        traits,
        topo_order[1:]
    )

    return traits

# ============================================================
# Batched Tree Simulation
# ============================================================

def simulate_tree_batch(
    key,
    root_states,
    parents,
    branch_lengths,
    topo_order,
    drift_fn,
    diffusion_fn,
    manifold,
    dt
):
    """
    Simulate multiple phylogenetic trait datasets.

    Parameters
    ----------
    root_states : jnp.ndarray

        Shape
        -----
        (B, d)

    Returns
    -------
    traits_batch : jnp.ndarray

        Shape
        -----
        (B, N, d)
    """

    B = root_states.shape[0]

    keys = jax.random.split(key, B)

    simulate_single = lambda k, x0: simulate_tree_traits(
        k,
        x0,
        parents,
        branch_lengths,
        topo_order,
        drift_fn,
        diffusion_fn,
        manifold,
        dt
    )

    return jax.vmap(simulate_single)(keys, root_states)

