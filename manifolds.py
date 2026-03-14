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

from dataclasses import dataclass
from typing import Any, Callable, Tuple
import math

import jax
import jax.numpy as jnp
from kernels import (
    _multi_indices_leq,
    interval_cosine_basis,
    reflecting_diffusion_generator,
    simplex_monomial_basis,
    s1_fourier_basis,
    s1_wrapped_ou_generator,
    wright_fisher_generator,
)


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

# ============================================================
# Riemannian Differential Operators for Manifold Diffusions
# ============================================================


# ------------------------------------------------------------
# S1 Laplace–Beltrami Operator
# ------------------------------------------------------------

def s1_laplace_beltrami_operator(
    theta: jnp.ndarray,
    f_fn
) -> jnp.ndarray:
    """
    Compute Laplace–Beltrami operator on S¹.

    Mathematical definition
    -----------------------

    On the unit circle with coordinate θ,

        Δ_{S¹} f = ∂²f / ∂θ²

    Parameters
    ----------
    theta : jnp.ndarray

        Shape
        -----
        (...,)

    f_fn : Callable
        Function mapping θ -> scalar.

    Returns
    -------
    lap : jnp.ndarray

        Shape
        -----
        (...,)
    """

    def single_point(t):

        df = jax.grad(f_fn)(t)

        d2f = jax.grad(lambda z: jax.grad(f_fn)(z))(t)

        return d2f

    return jax.vmap(single_point)(theta)


# ------------------------------------------------------------
# S1 Gradient Operator
# ------------------------------------------------------------

def s1_gradient(
    theta: jnp.ndarray,
    f_fn
) -> jnp.ndarray:
    """
    Compute intrinsic gradient on S¹.

    On S¹ the gradient reduces to

        ∇f = df/dθ

    Parameters
    ----------
    theta : jnp.ndarray

        Shape
        -----
        (...,)

    f_fn : Callable

    Returns
    -------
    grad : jnp.ndarray

        Shape
        -----
        (...,)
    """

    grad_fn = jax.grad(f_fn)

    return jax.vmap(grad_fn)(theta)


# ------------------------------------------------------------
# Fisher Metric on Simplex
# ------------------------------------------------------------

def simplex_fisher_metric(
    p: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Fisher information metric on simplex.

    Metric definition

        g_ij = δ_ij / p_i

    Parameters
    ----------
    p : jnp.ndarray

        Shape
        -----
        (..., d)

    Returns
    -------
    G : jnp.ndarray

        Shape
        -----
        (..., d, d)
    """

    inv_p = 1.0 / p

    return jnp.einsum("...i,ij->...ij", inv_p, jnp.eye(p.shape[-1]))


# ------------------------------------------------------------
# Riemannian Gradient on Simplex
# ------------------------------------------------------------

def simplex_riemannian_gradient(
    p: jnp.ndarray,
    f_fn
) -> jnp.ndarray:
    """
    Compute Riemannian gradient under Fisher metric.

    Definition

        grad_f = G^{-1} ∇f

    where G is the Fisher metric.

    Parameters
    ----------
    p : jnp.ndarray

        Shape
        -----
        (..., d)

    f_fn : Callable

    Returns
    -------
    grad : jnp.ndarray

        Shape
        -----
        (..., d)
    """

    def single_point(x):

        grad_euclid = jax.grad(f_fn)(x)

        G = simplex_fisher_metric(x)

        G_inv = jnp.linalg.inv(G)

        return G_inv @ grad_euclid

    return jax.vmap(single_point)(p)


# ------------------------------------------------------------
# Laplace–Beltrami Operator on Simplex
# ------------------------------------------------------------

def simplex_laplace_beltrami(
    p: jnp.ndarray,
    f_fn
) -> jnp.ndarray:
    """
    Compute Laplace–Beltrami operator on simplex.

    Mathematical expression

        Δf =
        (1 / √|g|)
        ∂_i ( √|g| g^{ij} ∂_j f )

    Using Fisher metric.

    Parameters
    ----------
    p : jnp.ndarray

        Shape
        -----
        (..., d)

    f_fn : Callable

    Returns
    -------
    lap : jnp.ndarray

        Shape
        -----
        (...,)
    """

    def single_point(x):

        grad = jax.grad(f_fn)(x)

        hess = jax.jacfwd(jax.grad(f_fn))(x)

        G = simplex_fisher_metric(x)

        G_inv = jnp.linalg.inv(G)

        term = jnp.trace(G_inv @ hess)

        return term

    return jax.vmap(single_point)(p)


# ------------------------------------------------------------
# Generic Manifold Diffusion Generator
# ------------------------------------------------------------

def manifold_diffusion_generator(
    x: jnp.ndarray,
    f_fn,
    drift_fn,
    diffusion_fn
) -> jnp.ndarray:
    """
    Compute generator for manifold diffusion.

    Generator

        L f =
        μ·∇f
        +
        (1/2) σ² Δ_M f

    Parameters
    ----------
    x : jnp.ndarray

        Shape
        -----
        (..., d)

    f_fn : Callable

    drift_fn : Callable

    diffusion_fn : Callable

    Returns
    -------
    Lf : jnp.ndarray

        Shape
        -----
        (...,)
    """

    def single_point(z):

        grad = jax.grad(f_fn)(z)

        hess = jax.jacfwd(jax.grad(f_fn))(z)

        mu = drift_fn(z)

        sigma = diffusion_fn(z)

        drift_term = jnp.dot(mu, grad)

        diff_term = 0.5 * sigma**2 * jnp.trace(hess)

        return drift_term + diff_term

    return jax.vmap(single_point)(x)


def _pad_to_length(array, length, dtype=jnp.float32):
    """
    Pad or slice an array to a fixed length so it matches manifold dimension.

    Parameters
    ----------
    array : array_like
        Input vector.

    length : int
        Desired output length.

    dtype : dtype

    Returns
    -------
    padded : jnp.ndarray

        Shape
        -----
        (length,)
    """

    array = jnp.asarray(array, dtype=dtype)

    if length == 0:
        return jnp.zeros((0,), dtype=dtype)

    if array.size >= length:
        return array[:length]

    pad_size = length - array.size

    padding = jnp.zeros((pad_size,), dtype=dtype)

    return jnp.concatenate([array, padding], axis=0)


def _scalar_from_params(array, default=0.0):
    """
    Return a scalar value for one-dimensional drift or diffusion.

    Parameters
    ----------
    array : array_like
        Parameter vector.

    default : float
        Fallback value when `array` is empty.

    Returns
    -------
    scalar : jnp.ndarray
        Shape ()
    """

    array = jnp.asarray(array)

    if array.size == 0:
        return jnp.array(default, dtype=array.dtype)

    return array[0]


def _simplex_degree_for_M(dplus1: int, M: int) -> int:
    """
    Select the minimal total degree such that simplex monomials span at least `M` basis functions.

    Parameters
    ----------
    dplus1 : int
        Dimension of barycentric coordinates.

    M : int
        Desired number of basis functions.

    Returns
    -------
    degree : int
    """

    if M <= 0:
        return 0

    degree = 0

    landmark = dplus1 - 1

    while True:

        count = _multi_indices_leq(degree, landmark).shape[0]

        if count >= M:
            return degree

        degree += 1


@dataclass
class ManifoldSpec:
    """
    Descriptor bundling geometry-awareness for a spectral domain.
    """

    domain: str
    dimension: int
    sample_states_fn: Callable[[int, int], jnp.ndarray]
    basis_fn: Callable[[jnp.ndarray, int], jnp.ndarray]
    generator_fn: Callable[[Any, jnp.ndarray, int], jnp.ndarray]
    project_fn: Callable[[jnp.ndarray], jnp.ndarray]

    def sample_states(self, num_states: int, spectral_dim: int) -> jnp.ndarray:
        """
        Draw quadrature states for the manifold.
        """
        return self.sample_states_fn(num_states, spectral_dim)

    def evaluate_basis(self, states: jnp.ndarray, spectral_dim: int) -> jnp.ndarray:
        """
        Evaluate the manifold-specific spectral basis.
        """
        return self.basis_fn(states, spectral_dim)

    def apply_generator(self, params, states: jnp.ndarray, spectral_dim: int) -> jnp.ndarray:
        """
        Apply the infinitesimal generator to the basis.
        """
        return self.generator_fn(params, states, spectral_dim)

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Project points back to the manifold after a simulation step.
        """
        return self.project_fn(x)


def create_interval_manifold(L: float, U: float) -> ManifoldSpec:
    """
    Reflecting-boundary manifold on [L, U].
    """

    def sample_states(num_states, spectral_dim):
        """
        Sample uniform quadrature points on [L, U].

        Returns
        -------
        states : jnp.ndarray
            Shape (count,)
        """

        count = max(num_states, spectral_dim * 4, 256)

        return jnp.linspace(L, U, count)

    def basis(states, spectral_dim):
        """
        Evaluate reflecting cosine basis on the interval.

        Returns
        -------
        phi : jnp.ndarray
            Shape (..., spectral_dim)
        """

        K = max(spectral_dim - 1, 0)

        phi = interval_cosine_basis(states, K, L, U)

        return phi[..., :spectral_dim]

    def generator(params, states, spectral_dim):
        """
        Apply the reflecting interval generator to the spectral basis.

        Returns
        -------
        values : jnp.ndarray
            Shape (..., spectral_dim)
        """

        mu = _scalar_from_params(params.drift, default=0.0)

        sigma = _scalar_from_params(params.diffusion, default=1.0)

        K = max(spectral_dim - 1, 0)

        mu_fn = lambda _: mu

        sigma_fn = lambda _: sigma

        vals = reflecting_diffusion_generator(
            states,
            interval_cosine_basis,
            K,
            mu_fn,
            sigma_fn,
            L,
            U
        )

        return vals[..., :spectral_dim]

    def project_fn(x):
        """
        Project onto [L, U] via clipping.

        Returns
        -------
        clipped : jnp.ndarray
            Shape (...,)
        """

        return jnp.clip(x, L, U)

    return ManifoldSpec(
        domain="interval",
        dimension=1,
        sample_states_fn=sample_states,
        basis_fn=basis,
        generator_fn=generator,
        project_fn=project_fn
    )


def create_circle_manifold() -> ManifoldSpec:
    """
    Circular manifold using the wrapped Ornsteinâ€“Uhlenbeck generator.
    """

    def sample_states(num_states, spectral_dim):
        """
        Uniformly sample angles on [0, 2Ï€).

        Returns
        -------
        states : jnp.ndarray
            Shape (count,)
        """

        count = max(num_states, spectral_dim * 4, 256)

        return jnp.linspace(0.0, 2.0 * jnp.pi, count, endpoint=False)

    def basis(states, spectral_dim):
        """
        Evaluate Fourier basis up to frequency K.

        Returns
        -------
        phi : jnp.ndarray
            Shape (..., spectral_dim)
        """

        K = int(math.ceil(max(spectral_dim - 1, 0) / 2.0))

        phi = s1_fourier_basis(K, states)

        return phi[..., :spectral_dim]

    def generator(params, states, spectral_dim):
        """
        Apply the wrapped OU generator on the circle.

        Returns
        -------
        values : jnp.ndarray
            Shape (..., spectral_dim)
        """

        boundary = jnp.asarray(params.boundary_params)

        if boundary.size >= 2:
            kappa = boundary[0]
            preferred = boundary[1]
        elif boundary.size == 1:
            kappa = boundary[0]
            preferred = 0.0
        else:
            kappa = 0.0
            preferred = 0.0

        sigma = _scalar_from_params(params.diffusion, default=1.0)

        K = int(math.ceil(max(spectral_dim - 1, 0) / 2.0))

        vals = s1_wrapped_ou_generator(
            states,
            s1_fourier_basis,
            K,
            kappa,
            preferred,
            sigma
        )

        return vals[..., :spectral_dim]

    def project_fn(x):
        """
        Wrap angles to [0, 2Ï€).

        Returns
        -------
        wrapped : jnp.ndarray
            Shape (...,)
        """

        return wrap_angle(x)

    return ManifoldSpec(
        domain="circle",
        dimension=1,
        sample_states_fn=sample_states,
        basis_fn=basis,
        generator_fn=generator,
        project_fn=project_fn
    )


def create_simplex_manifold(dplus1: int, concentration: float = 1.0) -> ManifoldSpec:
    """
    Simplex manifold with Fisher geometry approximated via monomial basis.
    """

    def sample_states(num_states, spectral_dim):
        """
        Sample deterministic Dirichlet points inside the simplex.

        Returns
        -------
        states : jnp.ndarray
            Shape (count, dplus1)
        """

        count = max(num_states, spectral_dim * 4, 256)

        key = jax.random.PRNGKey(0)

        alpha = jnp.full((dplus1,), concentration)

        return jax.random.dirichlet(key, alpha, shape=(count,))

    def basis(states, spectral_dim):
        """
        Evaluate truncated simplex monomial basis of degree `degree`.

        Returns
        -------
        phi : jnp.ndarray
            Shape (..., spectral_dim)
        """

        degree = _simplex_degree_for_M(dplus1, spectral_dim)

        phi = simplex_monomial_basis(degree, states)

        return phi[..., :spectral_dim]

    def generator(params, states, spectral_dim):
        """
        Apply the Wrightâ€“Fisher generator to the basis.

        Returns
        -------
        values : jnp.ndarray
            Shape (..., spectral_dim)
        """

        theta = _pad_to_length(params.boundary_params, dplus1)

        theta = jnp.clip(theta, a_min=1e-6)

        basis_fn = lambda x: basis(x, spectral_dim)

        return wright_fisher_generator(
            states,
            basis_fn,
            theta
        )

    def project_fn(x):
        """
        Re-normalize to the simplex interior.

        Returns
        -------
        projected : jnp.ndarray
            Shape (..., dplus1)
        """

        clipped = jnp.clip(x, a_min=1e-12)

        summed = jnp.sum(clipped, axis=-1, keepdims=True)

        return clipped / (summed + 1e-12)

    return ManifoldSpec(
        domain="simplex",
        dimension=dplus1,
        sample_states_fn=sample_states,
        basis_fn=basis,
        generator_fn=generator,
        project_fn=project_fn
    )
