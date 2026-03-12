# manifolds.py
"""
Manifold utilities for S^1 and the probability simplex Δ^d.

This file provides:
- S^1 utilities: metric, distance, exp/log maps (angle coordinates), and projection.
- Simplex utilities: Fisher (information) inner product (bilinear form), tangent projection,
  retraction (positive normalized exponential), and Hellinger/Fisher geodesic distance.

Mathematical notes (LaTeX):
- For S^1 (angle θ): metric g(θ)=1 and geodesic distance d_S1(θ, φ)=\min_{k\in\mathbb Z} |θ-φ+2π k|.
- Fisher inner product on Δ^d (probabilities p with p_i>0, sum_i p_i=1):
  For tangent vectors u,v (with ∑_i u_i = ∑_i v_i = 0),
  \[
    \langle u, v \rangle_{p}^{\mathrm{Fisher}} = \sum_{i=0}^d \frac{u_i v_i}{p_i}.
  \]
- Hellinger / spherical embedding:
  ψ(p) = (\sqrt{p_0}, \dots, \sqrt{p_d}) maps Δ^d (interior) into positive orthant of S^d,
  and the Fisher metric corresponds to the sphere metric up to constant.

All functions accept and return JAX arrays. Small epsilons are used to maintain numerical stability.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


### --- S^1: angle utilities --- ###

def wrap_angle(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Wrap angle(s) to the interval [0, 2π).

    Parameters
    ----------
    theta
        (...,) angle(s) in radians (can be any real number).

    Returns
    -------
    wrapped
        (...,) angles in [0, 2π).
    """
    two_pi = 2.0 * jnp.pi
    return jnp.mod(theta, two_pi)


def s1_angle_difference(theta_a: jnp.ndarray, theta_b: jnp.ndarray) -> jnp.ndarray:
    """
    Minimal oriented angular difference (log map on S^1):
    returns the unique value in (-π, π] equal to theta_b - theta_a modulo 2π.

    \(\mathrm{Log}_{\theta_a}(\theta_b) = \mathrm{argmin}_{v \in (-\pi,\pi]} \theta_b - \theta_a - 2\pi k\).

    Parameters
    ----------
    theta_a, theta_b
        Angles shape (...,). Operates elementwise via broadcasting.

    Returns
    -------
    delta
        Oriented minimal differences in (-π, π], shape (...,).
    """
    raw = theta_b - theta_a
    wrapped = (raw + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    # Convention: map π to -π? We keep (-π, π]
    return wrapped


def s1_distance(theta_a: jnp.ndarray, theta_b: jnp.ndarray) -> jnp.ndarray:
    """
    Geodesic distance on S^1 between angles theta_a and theta_b.

    d(θ_a, θ_b) = | Log_{θ_a}(θ_b) | in [0, π].

    Parameters
    ----------
    theta_a, theta_b : (...,)

    Returns
    -------
    dist : (...,) geodesic distances
    """
    return jnp.abs(s1_angle_difference(theta_a, theta_b))


def s1_exp_map(theta: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map on S^1 with canonical metric (simply addition modulo 2π).

    Exp_{θ}(v) = θ + v (mod 2π).

    Parameters
    ----------
    theta : (...,) base angle
    v     : (...,) tangent vector (real number – angle increment)

    Returns
    -------
    theta_out : (...,) angle on [0, 2π)
    """
    return wrap_angle(theta + v)


def s1_log_map(theta_a: jnp.ndarray, theta_b: jnp.ndarray) -> jnp.ndarray:
    """
    Log map on S^1 (same as angle difference).

    Returns tangent vector at theta_a pointing to theta_b with magnitude in (-π, π].
    """
    return s1_angle_difference(theta_a, theta_b)


def s1_metric(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Metric at angle θ for S^1 (scalar): g(θ)=1 for the standard circle.

    Parameters
    ----------
    theta : (...,) unused but included for API consistency.

    Returns
    -------
    g : (...,) metric scalar (ones)
    """
    return jnp.ones_like(theta)


### --- Simplex Δ^d: Fisher metric, projections, retraction, distances --- ###

def simplex_project_tangent(v: jnp.ndarray) -> jnp.ndarray:
    """
    Project an ambient vector v ∈ R^{d+1} to the tangent space of the simplex at any point:
    tangent vectors satisfy ∑_i u_i = 0. The projection simply subtracts the mean.

    Projection: u = v - (1/(d+1)) * (sum_j v_j) * 1 .

    Parameters
    ----------
    v : (..., d+1)

    Returns
    -------
    u : (..., d+1) tangent vector with zero-sum per last axis.
    """
    s = jnp.sum(v, axis=-1, keepdims=True)   # (...,1)
    dplus1 = v.shape[-1]
    return v - s / float(dplus1)


def fisher_inner_product(p: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Fisher information inner product at p between tangent vectors u and v:
      <u, v>_p = sum_i (u_i v_i / p_i).
    The inputs u and v must satisfy sum(u)=sum(v)=0 (tangent condition); we do not enforce it here,
    but any residual mean will still be present in the formula (user should project).

    Parameters
    ----------
    p : (..., d+1) probability vectors with positive entries
    u : (..., d+1) tangent vector
    v : (..., d+1) tangent vector
    eps : float small positive regularizer to avoid divide-by-zero

    Returns
    -------
    ip : (...) scalar inner product values
    """
    p_safe = jnp.clip(p, a_min=eps)
    return jnp.sum((u * v) / p_safe, axis=-1)


def simplex_fisher_metric_matrix(p: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Return the ambient representation of the Fisher information metric at p as a diagonal matrix:
      G(p) = diag(1 / p_i),
    with the understanding that the metric is only non-degenerate on the tangent subspace sum=0.

    Parameters
    ----------
    p : (..., d+1)
    eps : float

    Returns
    -------
    G : (..., d+1, d+1) diagonal matrices (may be singular when restricted to ambient space).
    """
    p_safe = jnp.clip(p, a_min=eps)
    inv = 1.0 / p_safe                                # (..., d+1)
    # form diagonal matrices
    G = jnp.einsum('...i,...j->...ij', inv, jnp.eye(p.shape[-1]))
    return G


def simplex_retraction(p: jnp.ndarray, v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Retraction on the simplex (positive normalized exponential, commonly used in information geometry):

      Retract_p(v) = normalize( p * exp(v) ) = q / sum(q),  q_i = p_i * exp(v_i).

    This map ensures positivity and membership in Δ^d for any v in R^{d+1}. If the tangent constraint
    sum(v)=0 is enforced, this behaves like a Riemannian exponential approximation.

    Parameters
    ----------
    p : (..., d+1) base point (probabilities)
    v : (..., d+1) tangent vector or ambient displacement
    eps : small float for numerical stability

    Returns
    -------
    q : (..., d+1) new probability vector in simplex interior
    """
    p_safe = jnp.clip(p, a_min=eps)
    # Compute unnormalized q
    q = p_safe * jnp.exp(v)
    q_sum = jnp.sum(q, axis=-1, keepdims=True)
    q_norm = q / (q_sum + eps)
    return q_norm


def simplex_hellinger_distance(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Geodesic-like distance on the simplex via Hellinger / spherical embedding:
      d_H(p, q) = 2 * arccos( sum_i sqrt(p_i q_i) ).

    This equals (up to constant scaling) the Fisher-Rao distance for the multinomial manifold.

    Parameters
    ----------
    p, q : (..., d+1) probability vectors
    eps : float

    Returns
    -------
    dist : (...) distances in [0, 2π]
    """
    # inner product in the square-root embedding:
    root_inner = jnp.sum(jnp.sqrt(jnp.clip(p, a_min=eps) * jnp.clip(q, a_min=eps)), axis=-1)
    root_inner_clipped = jnp.clip(root_inner, -1.0, 1.0)
    return 2.0 * jnp.arccos(root_inner_clipped)


# Optionally: mappings between simplex and sphere (Hellinger embedding)
def simplex_to_sphere(p: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Map p in Δ^d to the positive orthant of S^d via square-root map ψ(p) = sqrt(p).

    Parameters
    ----------
    p : (..., d+1)

    Returns
    -------
    s : (..., d+1) unit vectors (norm ≈ 1)
    """
    s = jnp.sqrt(jnp.clip(p, a_min=eps))
    # Normalize numerically to unit length
    s = s / jnp.linalg.norm(s, axis=-1, keepdims=True)
    return s


def sphere_to_simplex(s: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Map a sphere vector s with non-negative entries to the simplex via squaring and normalization:
      p_i = s_i^2 / sum_j s_j^2  (this maps unit vectors to probabilities).

    Parameters
    ----------
    s : (..., d+1)

    Returns
    -------
    p : (..., d+1)
    """
    sq = jnp.square(s)
    total = jnp.sum(sq, axis=-1, keepdims=True) + eps
    return sq / total
