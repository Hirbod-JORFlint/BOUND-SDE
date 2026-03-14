# ============================================================
# Optimizer Framework
# ============================================================

import jax
import jax.numpy as jnp
from typing import NamedTuple

# ============================================================
# Adam Optimizer State
# ============================================================

class AdamState(NamedTuple):
    """
    State variables for Adam optimizer.

    Attributes
    ----------
    params : jnp.ndarray
        Current parameter vector.

        Shape
        -----
        (D,)

    m : jnp.ndarray
        First moment estimate.

        Shape
        -----
        (D,)

    v : jnp.ndarray
        Second moment estimate.

        Shape
        -----
        (D,)

    t : int
        Iteration counter.
    """

    params: jnp.ndarray
    m: jnp.ndarray
    v: jnp.ndarray
    t: int

# ============================================================
# Initialize Adam State
# ============================================================

def adam_init(params: jnp.ndarray) -> AdamState:
    """
    Initialize Adam optimizer state.

    Parameters
    ----------
    params : jnp.ndarray

        Shape
        -----
        (D,)

    Returns
    -------
    state : AdamState
    """

    zeros = jnp.zeros_like(params)

    return AdamState(
        params=params,
        m=zeros,
        v=zeros,
        t=0
    )

# ============================================================
# Adam Update Step
# ============================================================

def adam_step(
    state: AdamState,
    grad: jnp.ndarray,
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> AdamState:
    """
    Perform a single Adam update.

    Mathematical definition
    -----------------------

    m_t = β1 m_{t−1} + (1−β1) g_t

    v_t = β2 v_{t−1} + (1−β2) g_t²

    m̂_t = m_t / (1 − β1^t)

    v̂_t = v_t / (1 − β2^t)

    θ_{t+1} = θ_t − α m̂_t / (√v̂_t + ε)

    Parameters
    ----------
    state : AdamState

    grad : jnp.ndarray

        Shape
        -----
        (D,)

    Returns
    -------
    new_state : AdamState
    """

    t = state.t + 1

    m = beta1 * state.m + (1.0 - beta1) * grad
    v = beta2 * state.v + (1.0 - beta2) * (grad ** 2)

    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    params = state.params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

    return AdamState(params, m, v, t)

# ============================================================
# Optimization Kernel
# ============================================================

def adam_scan_step(
    state: AdamState,
    _,
    grad_fn
):
    """
    Scan-compatible Adam update.

    Parameters
    ----------
    state : AdamState

    grad_fn : callable
        Function returning gradient.

    Returns
    -------
    new_state : AdamState
    loss : float
    """

    params = state.params

    loss, grad = grad_fn(params)

    new_state = adam_step(state, grad)

    return new_state, loss

# ============================================================
# Adam Optimization Loop
# ============================================================

def run_adam(
    init_params: jnp.ndarray,
    grad_fn,
    num_steps: int = 1000
):
    """
    Run Adam optimization.

    Parameters
    ----------
    init_params : jnp.ndarray

        Shape
        -----
        (D,)

    grad_fn : callable

        Returns
        -------
        loss, grad

    num_steps : int

    Returns
    -------
    final_params : jnp.ndarray
    loss_history : jnp.ndarray
    """

    state = adam_init(init_params)

    def step_fn(state, _):

        params = state.params

        loss, grad = grad_fn(params)

        new_state = adam_step(state, grad)

        return new_state, loss

    state, losses = jax.lax.scan(
        step_fn,
        state,
        xs=None,
        length=num_steps
    )

    return state.params, losses

