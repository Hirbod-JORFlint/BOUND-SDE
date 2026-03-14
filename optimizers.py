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

# ============================================================
# L-BFGS Optimizer State
# ============================================================

class LBFGSState(NamedTuple):
    """
    State container for L-BFGS optimizer.

    Attributes
    ----------
    params : jnp.ndarray
        Current parameter vector.

        Shape
        -----
        (D,)

    grad : jnp.ndarray
        Current gradient.

        Shape
        -----
        (D,)

    s_history : jnp.ndarray
        Parameter displacement history.

        Shape
        -----
        (m, D)

    y_history : jnp.ndarray
        Gradient displacement history.

        Shape
        -----
        (m, D)

    rho_history : jnp.ndarray
        Reciprocal curvature scalars.

        Shape
        -----
        (m,)

    k : int
        Iteration index.
    """

    params: jnp.ndarray
    grad: jnp.ndarray
    s_history: jnp.ndarray
    y_history: jnp.ndarray
    rho_history: jnp.ndarray
    k: int

# ============================================================
# Initialize L-BFGS State
# ============================================================

def lbfgs_init(params: jnp.ndarray, grad: jnp.ndarray, memory: int = 10) -> LBFGSState:
    """
    Initialize L-BFGS optimizer state.

    Parameters
    ----------
    params : jnp.ndarray
        Shape
        -----
        (D,)

    grad : jnp.ndarray
        Initial gradient.

    memory : int
        Number of curvature pairs.

    Returns
    -------
    state : LBFGSState
    """

    D = params.shape[0]

    s_history = jnp.zeros((memory, D))
    y_history = jnp.zeros((memory, D))
    rho_history = jnp.zeros((memory,))

    return LBFGSState(
        params=params,
        grad=grad,
        s_history=s_history,
        y_history=y_history,
        rho_history=rho_history,
        k=0
    )

# ============================================================
# Two-Loop Recursion for Search Direction
# ============================================================

def lbfgs_direction(state: LBFGSState):
    """
    Compute L-BFGS search direction.

    Mathematical definition
    -----------------------

    p_k = -H_k^{-1} g_k

    using two-loop recursion.

    Returns
    -------
    direction : jnp.ndarray

        Shape
        -----
        (D,)
    """

    s = state.s_history
    y = state.y_history
    rho = state.rho_history

    q = state.grad

    m = s.shape[0]

    alpha = jnp.zeros((m,))

    def backward(carry, i):

        q, alpha = carry

        si = s[i]
        yi = y[i]
        rhoi = rho[i]

        ai = rhoi * jnp.dot(si, q)

        q = q - ai * yi

        alpha = alpha.at[i].set(ai)

        return (q, alpha), None

    (q, alpha), _ = jax.lax.scan(
        backward,
        (q, alpha),
        jnp.arange(m - 1, -1, -1)
    )

    gamma = jnp.dot(s[-1], y[-1]) / (jnp.dot(y[-1], y[-1]) + 1e-12)

    r = gamma * q

    def forward(carry, i):

        r = carry

        si = s[i]
        yi = y[i]
        rhoi = rho[i]

        beta = rhoi * jnp.dot(yi, r)

        r = r + si * (alpha[i] - beta)

        return r, None

    r, _ = jax.lax.scan(
        forward,
        r,
        jnp.arange(m)
    )

    return -r

# ============================================================
# L-BFGS Update Step
# ============================================================

def lbfgs_step(state: LBFGSState, grad_fn):
    """
    Perform one L-BFGS optimization step.

    Parameters
    ----------
    state : LBFGSState

    grad_fn : callable
        Returns
        -------
        loss, grad

    Returns
    -------
    new_state : LBFGSState
    loss : float
    """

    direction = lbfgs_direction(state)

    step_size = 1.0

    new_params = state.params + step_size * direction

    loss, new_grad = grad_fn(new_params)

    s = new_params - state.params
    y = new_grad - state.grad

    rho = 1.0 / (jnp.dot(y, s) + 1e-12)

    s_history = jnp.roll(state.s_history, shift=-1, axis=0)
    y_history = jnp.roll(state.y_history, shift=-1, axis=0)
    rho_history = jnp.roll(state.rho_history, shift=-1)

    s_history = s_history.at[-1].set(s)
    y_history = y_history.at[-1].set(y)
    rho_history = rho_history.at[-1].set(rho)

    new_state = LBFGSState(
        params=new_params,
        grad=new_grad,
        s_history=s_history,
        y_history=y_history,
        rho_history=rho_history,
        k=state.k + 1
    )

    return new_state, loss

# ============================================================
# L-BFGS Optimization Loop
# ============================================================

def run_lbfgs(
    init_params,
    grad_fn,
    num_steps=200,
    memory=10
):
    """
    Run L-BFGS optimization.

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

    memory : int

    Returns
    -------
    params : jnp.ndarray
    loss_history : jnp.ndarray
    """

    loss, grad = grad_fn(init_params)

    state = lbfgs_init(init_params, grad, memory)

    def step(state, _):

        new_state, loss = lbfgs_step(state, grad_fn)

        return new_state, loss

    state, losses = jax.lax.scan(
        step,
        state,
        xs=None,
        length=num_steps
    )

    return state.params, losses

