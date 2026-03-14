"""SDE kernels: basis evaluations and generator helpers for constrained domains."""

from itertools import product
from typing import Callable
import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


def interval_cosine_basis(x: jnp.ndarray, K: int, L: float, U: float) -> jnp.ndarray:
    r"""
    Cosine Neumann basis on the reflecting interval [L, U].

    Mathematical intent
    ------------------
    \[
        \phi_0(x) = \frac{1}{U-L},\quad
        \phi_k(x) = \frac{2}{U-L}\cos\left(\frac{k\pi(x-L)}{U-L}\right),\quad k\geq 1.
    \]

    Parameters
    ----------
    x : jnp.ndarray
        Points lying in [L, U].
        Shape
        -----
        (...,)

    K : int
        Highest oscillatory mode.

    L : float
        Lower reflecting boundary.

    U : float
        Upper reflecting boundary.

    Returns
    -------
    basis_vals : jnp.ndarray
        Shape
        -----
        (..., K+1)
    """

    x = jnp.asarray(x)
    width = U - L
    if width <= 0:
        raise ValueError("Upper bound U must be greater than lower bound L.")
    base = jnp.full(x.shape, 1.0 / width)
    if K == 0:
        return base[..., None]

    ks = jnp.arange(1, K + 1, dtype=x.dtype)
    angles = (x[..., None] - L) * (ks * jnp.pi / width)
    cos_terms = jnp.cos(angles) * (2.0 / width)
    return jnp.concatenate([base[..., None], cos_terms], axis=-1)


def interval_laplacian_eigenvalues(K: int, sigma: float, L: float, U: float) -> jnp.ndarray:
    r"""
    Eigenvalues of the reflecting Laplacian on [L, U].

    Mathematical intent
    ------------------
    \[
        \lambda_k = -\frac{\sigma^2}{2}\left(\frac{k\pi}{U-L}\right)^2,\quad k=0,\dots,K.
    \]

    Parameters
    ----------
    K : int
        Largest cosine index.

    sigma : float
        Diffusion amplitude.

    L : float
        Lower boundary.

    U : float
        Upper boundary.

    Returns
    -------
    eigenvalues : jnp.ndarray
        Shape
        -----
        (K+1,)
    """

    width = jnp.array(U - L, dtype=jnp.float32)
    ks = jnp.arange(0, K + 1, dtype=jnp.float32)
    return -0.5 * sigma**2 * (ks * jnp.pi / width) ** 2


def s1_fourier_basis(theta: jnp.ndarray, K: int) -> jnp.ndarray:
    r"""
    Fourier basis functions on the unit circle.

    Mathematical intent
    ------------------
    \[
        \phi_0(\theta) = \frac{1}{\sqrt{2\pi}},\quad
        \phi_{2k-1}(\theta) = \sqrt{\frac{1}{\pi}}\cos(k\theta),\quad
        \phi_{2k}(\theta) = \sqrt{\frac{1}{\pi}}\sin(k\theta).
    \]

    Parameters
    ----------
    theta : jnp.ndarray
        Angles in radians.
        Shape
        -----
        (...,)

    K : int
        Highest frequency.

    Returns
    -------
    basis_vals : jnp.ndarray
        Shape
        -----
        (..., 2K+1)
    """

    theta = jnp.asarray(theta)
    base = jnp.full(theta.shape, 1.0 / jnp.sqrt(2.0 * jnp.pi))
    if K == 0:
        return base[..., None]

    ks = jnp.arange(1, K + 1, dtype=theta.dtype)
    angles = theta[..., None] * ks
    cos_terms = jnp.cos(angles) / jnp.sqrt(jnp.pi)
    sin_terms = jnp.sin(angles) / jnp.sqrt(jnp.pi)
    stacked = jnp.stack([cos_terms, sin_terms], axis=-1)
    interleaved = stacked.reshape(theta.shape + (2 * K,))
    return jnp.concatenate([base[..., None], interleaved], axis=-1)


def s1_laplace_eigenvalues(K: int, sigma: float) -> jnp.ndarray:
    r"""
    Laplace eigenvalues on S^1 aligned with `s1_fourier_basis`.

    Mathematical intent
    ------------------
    \[
        \lambda_0=0,\quad \lambda_{\text{cos/sin}} = -\frac{\sigma^2}{2}k^2,\quad k=1,\dots,K.
    \]

    Parameters
    ----------
    K : int
        Highest sine/cosine frequency.

    sigma : float
        Diffusion amplitude.

    Returns
    -------
    eigenvalues : jnp.ndarray
        Shape
        -----
        (2K+1,)
    """

    ks = jnp.arange(1, K + 1, dtype=jnp.float32)
    pair_vals = -0.5 * sigma**2 * ks**2
    repeated = jnp.repeat(pair_vals, 2)
    zero = jnp.array([0.0], dtype=repeated.dtype)
    return jnp.concatenate([zero, repeated], axis=0)


def _multi_indices_leq(degree: int, dim: int) -> jnp.ndarray:
    r"""
    Multi-indices \(\alpha\) with \(\sum_i \alpha_i \leq \text{degree}\) in \(\mathbb{N}^{d+1}\).

    Parameters
    ----------
    degree : int
        Total degree bound.

    dim : int
        Simplex dimension \(d := \text{dim}\).

    Returns
    -------
    indices : jnp.ndarray
        Shape
        -----
        (M, dim+1)
    """

    tuples = [t for t in product(range(degree + 1), repeat=dim + 1) if sum(t) <= degree]
    return jnp.asarray(tuples, dtype=jnp.int32)


def _simplex_dirichlet_gram(
    alphas: jnp.ndarray, dplus1: int, eps: float = 1e-6, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    r"""
    Gram matrix for monomials under a uniform Dirichlet weight on Δ^d.

    Mathematical intent
    ------------------
    \[
        G_{ij} = \int_{\Delta^d} p^{\alpha_i+\alpha_j}\,dp = \frac{\prod_{k}\Gamma(1+\alpha_{i,k}+\alpha_{j,k})}{\Gamma(d+1+\sum_k(\alpha_{i,k}+\alpha_{j,k}))}.
    \]

    Parameters
    ----------
    alphas : jnp.ndarray
        Multi-indices shape (M, d+1).

    dplus1 : int
        Number of components in the simplex.

    eps : float
        Diagonal regularizer to ensure positive definiteness.

    Returns
    -------
    gram : jnp.ndarray
        Shape
        -----
        (M, M)
    """

    alphas = jnp.asarray(alphas, dtype=dtype)
    expanded = alphas[:, None, :] + alphas[None, :, :]
    totals = jnp.sum(expanded, axis=-1)
    log_num = jnp.sum(jsp.gammaln(1.0 + expanded), axis=-1)
    log_den = jsp.gammaln(dplus1 + totals)
    gram = jnp.exp(log_num - log_den)
    normalization = math.factorial(dplus1 - 1) if dplus1 > 1 else 1
    gram = gram * normalization
    diag = jnp.eye(gram.shape[0], dtype=dtype) * eps
    return gram + diag


def simplex_monomial_basis(p: jnp.ndarray, degree: int) -> jnp.ndarray:
    r"""
    Orthonormal polynomial basis on the probability simplex Δ^d for degree ≤ 2.

    Mathematical intent
    ------------------
    Polynomials are orthonormalized under the Dirichlet(1) weight using Gram–Schmidt.

    Parameters
    ----------
    p : jnp.ndarray
        Simplex points with positive entries summing to one.
        Shape
        -----
        (..., d+1)

    degree : int
        Degree bound (maximum 2).

    Returns
    -------
    basis_vals : jnp.ndarray
        Shape
        -----
        (..., M)
    """

    if degree > 2:
        raise NotImplementedError("Simplex basis currently supports degree ≤ 2.")

    p = jnp.asarray(p)
    dplus1 = p.shape[-1]
    alphas_int = _multi_indices_leq(degree, dplus1 - 1)
    monomials = jnp.prod(jnp.power(p[..., None, :], alphas_int[None, :, :]), axis=-1)
    dtype = p.dtype
    alphas_float = alphas_int.astype(dtype)
    gram = _simplex_dirichlet_gram(alphas_float, dplus1, eps=1e-6, dtype=dtype)
    eigvals, eigvecs = jnp.linalg.eigh(gram)
    safe = 1e-12
    inv_sqrt = jnp.where(eigvals > safe, 1.0 / jnp.sqrt(eigvals), 0.0)
    transform = eigvecs @ (inv_sqrt[:, None] * eigvecs.T)
    basis = jnp.matmul(monomials, transform)
    return basis


def apply_generator_batch(
    generator_fn: Callable[..., jnp.ndarray],
    states: jnp.ndarray,
    basis_fn: Callable[..., jnp.ndarray],
    *args,
) -> jnp.ndarray:
    r"""
    Vectorize the generator applied to a basis evaluated at states.

    Parameters
    ----------
    generator_fn : Callable
        Function computing \(L\phi\) per state/basis row.
    states : jnp.ndarray
        Points where the generator is evaluated.
        Shape
        -----
        (N, ...)
    basis_fn : Callable
        Basis evaluator returning shape (..., M).
    *args : Any
        Additional arguments forwarded to both basis_fn and generator_fn.

    Returns
    -------
    values : jnp.ndarray
        Shape
        -----
        (N, M)
    """

    states_array = jnp.asarray(states)
    basis_vals = basis_fn(states_array, *args)

    def _apply(state, basis_row):
        return generator_fn(state, basis_row, *args)

    return jax.vmap(_apply)(states_array, basis_vals)


def reflecting_diffusion_generator(
    x: jnp.ndarray,
    basis_fn: Callable[..., jnp.ndarray],
    K: int,
    mu_fn: Callable[[jnp.ndarray], jnp.ndarray],
    sigma_fn: Callable[[jnp.ndarray], jnp.ndarray],
    L: float,
    U: float,
) -> jnp.ndarray:
    r"""
    Evaluate the reflecting diffusion generator on the cosine basis.

    Mathematical intent
    ------------------
    \[
        L\phi_k = \mu(x)\phi_k'(x) + \frac{1}{2}\sigma(x)^2\phi_k''(x),\quad \phi_k'(L)=\phi_k'(U)=0.
    \]

    Parameters
    ----------
    x : jnp.ndarray
        Evaluation points in [L, U].
        Shape
        -----
        (...,)

    basis_fn : Callable
        Basis evaluator for the interval (expects signature basis_fn(x, K, L, U)).

    K : int
        Highest cosine index.

    mu_fn : Callable
        Drift function returning shape (...,).

    sigma_fn : Callable
        Diffusion amplitude returning shape (...,).

    L : float
        Lower boundary.

    U : float
        Upper boundary.

    Returns
    -------
    values : jnp.ndarray
        Shape
        -----
        (..., K+1)
    """

    points = jnp.asarray(x)
    width = U - L
    ks = jnp.arange(0, K + 1, dtype=points.dtype)
    prefactor = ks * jnp.pi / width
    angles = (points[..., None] - L) * prefactor
    dphi = -2.0 / width * prefactor * jnp.sin(angles)
    d2phi = -2.0 / width * prefactor**2 * jnp.cos(angles)
    mu = mu_fn(points)[..., None]
    sigma = sigma_fn(points)[..., None]
    return mu * dphi + 0.5 * sigma**2 * d2phi


def s1_wrapped_ou_generator(
    theta: jnp.ndarray,
    basis_fn: Callable[[jnp.ndarray, int], jnp.ndarray],
    K: int,
    kappa: float,
    preferred: float,
    sigma: float,
) -> jnp.ndarray:
    r"""
    Wrapped Ornstein–Uhlenbeck generator on \(S^1\).

    The generator is

        \(L f = -\kappa (\theta - \theta^*) f' + \frac{1}{2}\sigma^2 f''\).

    Parameters
    ----------
    theta : jnp.ndarray
        Angles in radians.

        Shape
        -----
        (...,)

    basis_fn : Callable
        Fourier basis evaluator.

    K : int
        Highest frequency.

    kappa : float
        Reversion strength.

    preferred : float
        Preferred angle.

    sigma : float
        Diffusion amplitude.

    Returns
    -------
    values : jnp.ndarray
        Generator applied to the basis.

        Shape
        -----
        (..., 2K+1)
    """

    def single(point):
        phi_fn = lambda z: basis_fn(z, K)
        grad = jax.jacrev(phi_fn)(point)
        hess = jax.jacfwd(jax.jacrev(phi_fn))(point)
        drift = -kappa * (point - preferred)
        return drift * grad + 0.5 * sigma**2 * hess

    return jax.vmap(single)(theta)


def wright_fisher_generator(
    p: jnp.ndarray,
    basis_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
) -> jnp.ndarray:
    r"""
    Wright–Fisher generator applied to simplex basis.

    Parameters
    ----------
    p : jnp.ndarray
        Simplex points.

        Shape
        -----
        (..., d+1)

    basis_fn : Callable
        Basis evaluator returning shape (..., M).

    theta : jnp.ndarray
        Dirichlet concentration.

        Shape
        -----
        (d+1,)

    Returns
    -------
    values : jnp.ndarray
        Generator acting on the basis.

        Shape
        -----
        (..., M)
    """

    Theta = jnp.sum(theta)

    def single(point):
        phi_fn = lambda z: basis_fn(z)
        grad = jax.jacrev(phi_fn)(point)
        hess = jax.jacfwd(jax.jacrev(phi_fn))(point)
        drift = theta - point * Theta
        drift_term = jnp.einsum("mi,i->m", grad, drift)
        cov = jnp.diag(point) - jnp.outer(point, point)
        diff_term = 0.5 * jnp.einsum("mij,ij->m", hess, cov)
        return drift_term + diff_term

    return jax.vmap(single)(p)
