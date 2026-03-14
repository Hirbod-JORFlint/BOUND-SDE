"""kernels.py: SDE infinitesimal generators and spectral basis functions.

Functions are pure, JAX-native, and document the targeted differential
operators with LaTeX in the docstrings.
"""

from itertools import product
from typing import Callable

import jax
import jax.numpy as jnp


def interval_cosine_basis(x: jnp.ndarray, K: int, L: float, U: float) -> jnp.ndarray:
    """
    Reflecting-boundary cosine basis on [L,U].

    Mathematical intent
    ------------------
    \[
        \phi_0(x) = \frac{1}{\sqrt{U-L}},\quad
        \phi_k(x) = \sqrt{\frac{2}{U-L}}\cos\left(\frac{k\pi(x-L)}{U-L}\right),
        \quad k\geq 1
    \]
    satisfies \(\partial_x \phi_k(L)=\partial_x \phi_k(U)=0\).

    Parameters
    ----------
    x : jnp.ndarray
        Points in the interval [L, U].

        Shape
        -----
        (...,)

    K : int
        Number of cosine modes beyond the constant.

    L : float
        Lower boundary.

    U : float
        Upper boundary.

    Returns
    -------
    basis_vals : jnp.ndarray

        Shape
        -----
        (..., K+1)
    """

    x = jnp.asarray(x)
    width = U - L
    phi0 = jnp.full_like(x, 1.0 / jnp.sqrt(width))

    if K == 0:
        return phi0[..., None]

    ks = jnp.arange(1, K + 1)
    angles = jnp.expand_dims(x - L, axis=-1) * (ks * jnp.pi / width)
    cos_terms = jnp.cos(angles) * jnp.sqrt(2.0 / width)
    return jnp.concatenate([phi0[..., None], cos_terms], axis=-1)


def interval_laplacian_eigenvalues(K: int, sigma: float, L: float, U: float) -> jnp.ndarray:
    """
    Eigenvalues of the reflecting Laplacian on [L,U].

    Mathematical intent
    ------------------
    \[
        \lambda_k = -\frac{\sigma^2}{2}\left(\frac{k\pi}{U-L}\right)^2,\quad k=0,\dots,K
    \]

    Parameters
    ----------
    K : int
        Highest cosine index.

    sigma : float
        Diffusion coefficient.

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

    width = U - L
    ks = jnp.arange(0, K + 1)
    return -0.5 * sigma**2 * (ks * jnp.pi / width) ** 2


def s1_fourier_basis(theta: jnp.ndarray, K: int) -> jnp.ndarray:
    """
    2K+1 Fourier basis functions on S^1 (constant + sin/cos pairs).

    Mathematical intent
    ------------------
    \[
        \phi_0(\theta) = \frac{1}{\sqrt{2\pi}},\quad
        \phi_{2k-1} = \sqrt{\frac{1}{\pi}}\cos(k\theta),\quad
        \phi_{2k} = \sqrt{\frac{1}{\pi}}\sin(k\theta).
    \]

    Parameters
    ----------
    theta : jnp.ndarray
        Angles on S^1 in radians.

        Shape
        -----
        (...,)

    K : int
        Highest sinusoidal frequency to include.

    Returns
    -------
    basis_vals : jnp.ndarray

        Shape
        -----
        (..., 2K+1)
    """

    theta = jnp.asarray(theta)
    base = jnp.full(theta.shape, 1.0 / jnp.sqrt(2 * jnp.pi))

    if K == 0:
        return base[..., None]

    ks = jnp.arange(1, K + 1)
    angles = jnp.expand_dims(theta, axis=-1) * ks
    cos_terms = jnp.cos(angles) / jnp.sqrt(jnp.pi)
    sin_terms = jnp.sin(angles) / jnp.sqrt(jnp.pi)
    interleaved = jnp.stack([cos_terms, sin_terms], axis=-1).reshape(theta.shape + (2 * K,))
    return jnp.concatenate([base[..., None], interleaved], axis=-1)


def _multi_indices_leq(degree: int, dim: int) -> jnp.ndarray:
    """
    Enumerate multi-indices α in ℕ^{d+1} satisfying sum(α) ≤ degree.

    Parameters
    ----------
    degree : int
        Total degree bound.

    dim : int
        Equivalent to d (simplex dimension d = dim).

    Returns
    -------
    indices : jnp.ndarray

        Shape
        -----
        (M, dim+1)
    """

    tuples = [t for t in product(range(degree + 1), repeat=dim + 1) if sum(t) <= degree]
    return jnp.array(tuples, dtype=jnp.int32)


def simplex_monomial_basis(p: jnp.ndarray, degree: int) -> jnp.ndarray:
    """
    Monomial basis on the probability simplex Δ^d.

    Mathematical intent
    ------------------
    \[
        \phi_\alpha(p) = \prod_{i=0}^d p_i^{\alpha_i},\quad \sum_i\alpha_i \leq \text{degree}.
    \]

    Parameters
    ----------
    p : jnp.ndarray
        Points with non-negative entries summing to one.

        Shape
        -----
        (..., d+1)

    degree : int
        Max total degree.

    Returns
    -------
    basis_vals : jnp.ndarray

        Shape
        -----
        (..., M)
    """

    p = jnp.asarray(p)
    dplus1 = p.shape[-1]
    dims = dplus1 - 1
    alphas = _multi_indices_leq(degree, dims)
    powers = jnp.power(p[..., None, :], alphas[None, :, :])
    return jnp.prod(powers, axis=-1)


def s1_laplace_beltrami(theta: jnp.ndarray, sigma: float, K: int) -> jnp.ndarray:
    """
    Apply the Laplace–Beltrami operator to the Fourier basis.

    Mathematical intent
    ------------------
    \(\mathcal{L} f = \frac{\sigma^2}{2}\frac{d^2f}{d\theta^2}\),
    so \(\mathcal{L}\phi_k = -\frac{\sigma^2}{2}k^2\phi_k\).

    Parameters
    ----------
    theta : jnp.ndarray
        Evaluation angles.

        Shape
        -----
        (...,)

    sigma : float
        Diffusion amplitude.

    K : int
        Highest frequency index.

    Returns
    -------
    values : jnp.ndarray

        Shape
        -----
        (..., 2K+1)
    """

    freqs = jnp.concatenate([jnp.array([0]), jnp.repeat(jnp.arange(1, K + 1), 2)])
    basis_vals = s1_fourier_basis(theta, K)
    eigen = -0.5 * sigma**2 * freqs**2
    return basis_vals * eigen


def wright_fisher_generator(p: jnp.ndarray, theta: jnp.ndarray, degree: int) -> jnp.ndarray:
    """
    Apply the Wright–Fisher generator to simplex monomials.

    Mathematical intent
    ------------------
    \[
        Lf(p) = \sum_i(\theta_i - p_i\Theta)\partial_i f + \frac{1}{2}\sum_{i,j}p_i(\delta_{ij}-p_j)\partial_{ij} f,
        \quad \Theta = \sum_i \theta_i.
    \]

    Parameters
    ----------
    p : jnp.ndarray
        Simplex points.

        Shape
        -----
        (..., d+1)

    theta : jnp.ndarray
        Dirichlet parameters.

        Shape
        -----
        (d+1,)

    degree : int
        Monomial degree used for the basis.

    Returns
    -------
    values : jnp.ndarray

        Shape
        -----
        (..., M)
    """

    def generator(x):
        f = lambda y: simplex_monomial_basis(y, degree)
        grad = jax.jacrev(f)(x)
        hess = jax.jacfwd(jax.jacrev(f))(x)
        Theta = jnp.sum(theta)
        drift = theta - x * Theta
        drift_term = jnp.einsum("...i,...i->...", drift, grad)
        cov = jnp.diag(x) - jnp.outer(x, x)
        diff_term = 0.5 * jnp.tensordot(hess, cov, axes=2)
        return drift_term + diff_term

    return jax.vmap(generator)(p)


def reflecting_diffusion_generator(
    x: jnp.ndarray,
    mu_fn: Callable[[jnp.ndarray], jnp.ndarray],
    sigma_fn: Callable[[jnp.ndarray], jnp.ndarray],
    degree: int,
    L: float,
    U: float,
) -> jnp.ndarray:
    """
    Apply generator for reflecting diffusion on [L,U].

    Mathematical intent
    ------------------
    \[
        Lf = \mu(x)f'(x) + \frac{1}{2}\sigma(x)^2 f''(x),\quad f'(L)=f'(U)=0.
    \]

    Parameters
    ----------
    x : jnp.ndarray
        Inputs on [L, U].

        Shape
        -----
        (...,)

    mu_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Drift function returning shape (...,).

    sigma_fn : Callable[[jnp.ndarray], jnp.ndarray]
        Diffusion amplitude function returning shape (...,).

    degree : int
        Number of cosine modes (excluding constant).

    L : float
        Lower boundary.

    U : float
        Upper boundary.

    Returns
    -------
    values : jnp.ndarray

        Shape
        -----
        (..., degree+1)
    """

    def generator(point):
        def basis_fn(z):
            return interval_cosine_basis(z, degree, L, U)

        df = jax.jacrev(basis_fn)(point)
        d2f = jax.jacfwd(jax.jacrev(basis_fn))(point)
        mu = mu_fn(point)
        sigma = sigma_fn(point)
        drift_term = mu * df
        diff_term = 0.5 * sigma**2 * d2f
        return drift_term + diff_term

    return jax.vmap(generator)(x)


if __name__ == "__main__":
    theta = jnp.array([0.0, jnp.pi / 2])
    basis = interval_cosine_basis(jnp.linspace(0.0, 1.0, 5), 3, 0.0, 1.0)
    assert basis.shape == (5, 4)

    eigs = interval_laplacian_eigenvalues(3, 1.0, 0.0, 1.0)
    assert eigs.shape == (4,)

    s1_values = s1_fourier_basis(theta, 2)
    assert s1_values.shape == (2, 5)
    s1_gen = s1_laplace_beltrami(theta, 1.0, 2)
    assert jnp.allclose(s1_gen[..., 0], 0.0)

    simplex = simplex_monomial_basis(jnp.array([[0.6, 0.4]]), 2)
    assert simplex.ndim == 2

    theta_sim = jnp.array([1.0, 2.0])
    wf = wright_fisher_generator(jnp.array([[0.8, 0.2]]), theta_sim, 2)
    assert wf.shape[1] >= 1

    def mu_fn(x):
        return jnp.zeros_like(x)

    def sigma_fn(x):
        return jnp.ones_like(x)

    refl = reflecting_diffusion_generator(jnp.array([0.25, 0.75]), mu_fn, sigma_fn, 2, 0.0, 1.0)
    assert refl.shape == (2, 3)
