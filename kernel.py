"""
JAX kernels: infinitesimal generator utilities and spectral basis functions.

This module contains:
- `euclidean_generator`: build generator action L[f] for Euclidean SDEs using autodiff.
- `s1_fourier_basis`: real-valued orthonormal Fourier basis on S^1.
- `simplex_monomial_basis`: monomial basis on the probability simplex Δ^d.
- `orthonormalize_simplex_basis`: Monte-Carlo empirical orthonormalization under Dirichlet weight.

All functions follow JAX functional style (pure functions) and are vectorization-friendly
(i.e. they are ready to be wrapped with `jax.vmap` or used inside `jax.lax.scan`).

Math/notation (LaTeX):
- SDE (Euclidean): \(\mathrm{d}X_t = \mu(X_t)\,\mathrm{d}t + \sigma(X_t)\,\mathrm{d}W_t.\)
- Infinitesimal generator: \((\mathcal{L} f)(x) = \mu(x)^\top \nabla f(x) + \tfrac{1}{2}\mathrm{trace}\!\big( a(x)\, \mathrm{Hess} f(x)\big)\),
  where \(a(x)=\sigma(x)\sigma(x)^\top\).

Notes:
- All arrays are JAX arrays (`jnp.ndarray`).
- Use `jax.jit`/`jax.vmap` externally as desired.
"""

from typing import Callable, Tuple, Sequence
import jax
import jax.numpy as jnp


### --- Infinitesimal generator utilities --- ###

def euclidean_generator(mu_fn: Callable[[jnp.ndarray], jnp.ndarray],
                        sigma_fn: Callable[[jnp.ndarray], jnp.ndarray]
                       ) -> Callable[[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """
    Build an operator that applies the infinitesimal generator L to a (possibly vector-valued)
    function f at point x in R^d.

    The returned function `L_apply(f_fn, x)` computes (for vector-valued f with shape (m,)):
    \[
      (\mathcal{L} f)_k(x) = \sum_{i} \mu_i(x) \partial_{i} f_k(x)
                          + \tfrac{1}{2} \sum_{i,j} a_{ij}(x) \partial_{i}\partial_{j} f_k(x),
    \]
    where \(a(x) = \sigma(x)\sigma(x)^\top\).

    Parameters
    ----------
    mu_fn
        Callable mu_fn(x) -> shape (d,) drift vector \(\mu(x)\).
    sigma_fn
        Callable sigma_fn(x) -> shape (d, r) diffusion matrix \(\sigma(x)\).
        The diffusion covariance is `a = sigma @ sigma.T`.

    Returns
    -------
    L_apply
        Callable L_apply(f_fn, x) -> shape (m,) (if f returns vector of length m) or scalar if f returns scalar.

    Shapes
    ------
    x : (d,)
    mu_fn(x) : (d,)
    sigma_fn(x) : (d, r)
    f_fn(x) : (m,) or scalar
    L_apply(...) : (m,) or scalar
    """
    def L_apply(f_fn: Callable[[jnp.ndarray], jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        # Evaluate mu and diffusion covariance
        mu = mu_fn(x)                     # (d,)
        sigma = sigma_fn(x)               # (d, r)
        a = sigma @ sigma.T               # (d, d)

        # Evaluate f at x to infer output shape (m,) or scalar
        f_x = f_fn(x)
        is_scalar = jnp.ndim(f_x) == 0

        # Jacobian: J[k,i] = ∂ f_k / ∂ x_i
        J = jax.jacrev(f_fn)(x)          # if f returns (m,), J has shape (m, d); if scalar then (d,)
        # Hessian: H[k,i,j] = ∂^2 f_k / ∂ x_i ∂ x_j
        # For vector-valued f: use jacfwd(jacrev(f)) -> (m, d, d). For scalar f, returns (d, d).
        H = jax.jacfwd(jax.jacrev(f_fn))(x)

        if is_scalar:
            # J shape (d,), H shape (d,d)
            drift_term = jnp.dot(mu, J)                                  # scalar
            diffusion_term = 0.5 * jnp.tensordot(H, a, axes=2)           # scalar: sum_{i,j} H_{ij} a_{ij}
            return drift_term + diffusion_term
        else:
            # J shape (m, d), H shape (m, d, d)
            # drift_term for each component k: sum_i mu_i * J[k,i]
            drift_term = jnp.einsum('i,ki->k', mu, J)                    # (m,)
            # diffusion term: 0.5 * sum_{i,j} H[k,i,j] * a[i,j] for each k
            diffusion_term = 0.5 * jnp.einsum('kij,ij->k', H, a)         # (m,)
            return drift_term + diffusion_term

    return L_apply

### --- S^1 Fourier basis (real-valued orthonormal) --- ###

def s1_fourier_basis(K: int, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Real orthonormal Fourier basis on S^1 (angles in radians on [0, 2π)).

    We use the orthonormal set:
      φ_0(θ) = 1 / sqrt(2π)
      φ_{2k-1}(θ) = sqrt(1/π) * cos(k θ),  k=1..K
      φ_{2k}(θ)   = sqrt(1/π) * sin(k θ),  k=1..K

    This yields (2K + 1) real orthonormal functions with respect to the inner product
    \(\langle f, g \rangle = \int_0^{2\pi} f(\theta) g(\theta) \, \mathrm d\theta\).

    Parameters
    ----------
    K
        Highest Fourier frequency (non-negative integer).
    theta
        angle(s) with shape (...,) representing points on S^1 in radians.

    Returns
    -------
    basis_vals
        Array of shape (..., 2K+1) where the last axis enumerates basis functions in order:
        [φ_0, cos(1·θ), sin(1·θ), cos(2·θ), sin(2·θ), ..., cos(K·θ), sin(K·θ)].

    Shapes
    ------
    theta : (...,)
    return : (..., 2K+1)
    """
    theta = jnp.asarray(theta)
    base0 = jnp.full_like(theta, 1.0) / jnp.sqrt(2.0 * jnp.pi)
    if K == 0:
        return base0[..., None]   # (...,1)

    # prepare cos and sin terms
    ks = jnp.arange(1, K + 1)
    # broadcast: theta[..., None] * ks[None, ...] -> (..., K)
    angles = jnp.expand_dims(theta, axis=-1) * ks  # (..., K)
    cos_terms = jnp.cos(angles) * (1.0 / jnp.sqrt(jnp.pi))
    sin_terms = jnp.sin(angles) * (1.0 / jnp.sqrt(jnp.pi))
    # interleave cos and sin in last axis
    # produce (..., 2K)
    interleaved = jnp.stack([cos_terms, sin_terms], axis=-1).reshape(theta.shape + (2 * K,))
    return jnp.concatenate([base0[..., None], interleaved], axis=-1)


### --- Simplex monomial basis + empirical orthonormalization --- ###

def _multi_indices_leq(degree: int, dim: int) -> jnp.ndarray:
    """
    Generate all multi-indices alpha = (alpha_0,...,alpha_dim) with sum(alpha) <= degree.
    Return as array shape (M, dim+1) of non-negative integers.

    The simplex Δ^d is treated as probabilities over (d+1) categories; we work with length (d+1).
    """
    # We implement a simple recursive generator (converted to JAX-friendly python object once).
    # For moderate degree and dim this is fine; the list is small combinatorially.
    from itertools import product
    # Generate all combinations of exponents from 0..degree and filter sum <= degree
    ranges = [range(degree + 1)] * (dim + 1)
    tuples = [t for t in product(*ranges) if sum(t) <= degree]
    return jnp.array(tuples, dtype=jnp.int32)   # (M, d+1)


def simplex_monomial_basis(degree: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    Monomial basis on the probability simplex Δ^d.

    Basis functions are φ_alpha(x) = \prod_{i=0}^d x_i^{alpha_i} for multi-indices alpha with
    sum(alpha) <= degree. We return basis values evaluated at x.

    Parameters
    ----------
    degree
        Maximum total degree (non-negative integer).
    x
        Points on simplex, shape (..., d+1). Each x must satisfy x_i >= 0 and sum_i x_i = 1 (within numerical tolerance).

    Returns
    -------
    monomials
        Array of shape (..., M) where M = number of multi-indices with sum <= degree.

    Shapes
    ------
    x : (..., d+1)
    return : (..., M)
    """
    x = jnp.asarray(x)
    assert x.ndim >= 1
    dplus1 = x.shape[-1]
    dims = dplus1 - 1
    alphas = _multi_indices_leq(degree, dims)  # (M, d+1)
    # compute monomials: for each alpha, compute prod_i x_i^{alpha_i}
    # x[..., None, :] -> (..., 1, d+1), alphas[None, :, :] -> (1, M, d+1)
    # use jnp.prod over last axis of x ** alpha
    exps = jnp.power(x[..., None, :], alphas[None, :, :])   # (..., M, d+1)
    monoms = jnp.prod(exps, axis=-1)                        # (..., M)
    return monoms


def orthonormalize_simplex_basis(degree: int,
                                 dirichlet_alpha: Sequence[float],
                                 rng_key: jax.random.KeyArray,
                                 n_samples: int = 10000
                                ) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """
    Empirically orthonormalize the simplex monomial basis under a Dirichlet weight via Monte-Carlo.

    Procedure:
      1. Generate N ~ Dirichlet(alpha) samples p^(s), s=1..N.
      2. Build matrix Φ of shape (N, M) where Φ[s, k] = φ_k(p^(s)).
      3. Estimate Gram matrix G = (1/N) Φ^T Φ.
      4. Compute Cholesky G = L L^T and define orthonormalizing transform C = L^{-T}.
         Then ψ(p) = C @ φ(p) yields orthonormal basis in the empirical inner product:
           (1/N) ∑_s ψ(p^(s)) ψ(p^(s))^T ≈ I_M.

    Returns a callable `orthonormal_basis_fn(x)` that maps x shape (..., d+1) to (..., M) orthonormal basis values,
    and also returns the transform matrix C (shape (M, M)) so the user can inspect coefficients.

    Parameters
    ----------
    degree
        Maximum total degree for monomials.
    dirichlet_alpha
        Sequence of length d+1 giving Dirichlet concentration parameters; used to draw samples.
    rng_key
        JAX PRNGKey for random sampling.
    n_samples
        Number of Monte-Carlo samples for Gram matrix estimation (default 10000).

    Returns
    -------
    orthonormal_basis_fn, C
        - orthonormal_basis_fn: function x -> (..., M) orthonormal basis values (empirical).
        - C: array (M, M) the linear transform applied to monomial vector to produce orthonormal basis.
    """
    # Sample points on simplex
    alpha = jnp.asarray(dirichlet_alpha)
    dimp1 = alpha.shape[0]
    # generate samples shape (n_samples, d+1)
    samples = jax.random.dirichlet(rng_key, alpha, shape=(n_samples,))  # (n_samples, d+1)

    # Evaluate monomial basis on samples
    Phi = simplex_monomial_basis(degree, samples)  # (n_samples, M)
    # Empirical Gram matrix
    G = (Phi.T @ Phi) / float(n_samples)           # (M, M)

    # Regularize slightly to ensure positive-definiteness numerically
    eps = 1e-10
    G_reg = G + eps * jnp.eye(G.shape[0])

    # Cholesky factorization (G_reg = L L^T)
    L = jnp.linalg.cholesky(G_reg)                 # (M, M)
    # Transform C = L^{-T} (so C @ Phi.T has identity empirical covariance)
    C = jax.scipy.linalg.solve_triangular(L, jnp.eye(L.shape[0]), trans='T', lower=False)  # L^T x = I -> x = (L^T)^{-1}
    # Above yields C = (L^T)^{-1}. Check: (C @ G @ C.T) ≈ I
    # Build basis function using closure over C and degree
    def orthonormal_basis_fn(x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate empirical orthonormal basis at points x: (..., d+1) -> (..., M)
        """
        monoms = simplex_monomial_basis(degree, x)    # (..., M)
        # Apply linear transform C to monomials: ψ = (monoms) @ C.T  -> (..., M)
        return jnp.einsum('...m,mk->...k', monoms, C.T)

    return orthonormal_basis_fn, C

### ============================================================
### Analytical Generators for Common Manifolds
### ============================================================

def interval_reflective_generator(
    mu: float,
    sigma: float,
    L: float,
    U: float
):
    """
    Construct infinitesimal generator for a reflected diffusion on interval [L,U].

    The SDE with reflection is:

    dX_t = μ dt + σ dW_t + dK_t

    where K_t enforces reflection at the boundaries.

    The generator inside the interval is

    Lf(x) = μ f'(x) + 1/2 σ² f''(x)

    with Neumann boundary conditions

    f'(L) = 0
    f'(U) = 0

    Parameters
    ----------
    mu : float
        Drift coefficient.
    sigma : float
        Diffusion coefficient.
    L : float
        Lower reflective boundary.
    U : float
        Upper reflective boundary.

    Returns
    -------
    generator
        Callable f(x) -> Lf(x)
    """

    def generator(f_fn, x):

        f_prime = jax.grad(f_fn)
        f_double = jax.grad(f_prime)

        drift = mu * f_prime(x)
        diffusion = 0.5 * sigma ** 2 * f_double(x)

        return drift + diffusion

    return generator

def s1_generator(mu: float, sigma: float):
    """
    Generator for diffusion on S¹.

    SDE
    ---
    dθ_t = μ dt + σ dW_t

    Generator
    ---------
    Lf(θ) = μ f'(θ) + 1/2 σ² f''(θ)

    Parameters
    ----------
    mu : float
    sigma : float

    Returns
    -------
    generator
        Callable applying generator to function f.
    """

    def generator(f_fn, theta):

        f_prime = jax.grad(f_fn)
        f_double = jax.grad(f_prime)

        drift = mu * f_prime(theta)
        diffusion = 0.5 * sigma ** 2 * f_double(theta)

        return drift + diffusion

    return generator

def wright_fisher_generator(alpha: jnp.ndarray):
    """
    Wright-Fisher diffusion generator on simplex Δ^d.

    Parameters
    ----------
    alpha : (d+1,)
        Dirichlet concentration parameters.

    Returns
    -------
    generator
        Callable applying generator to function f(x).
    """

    alpha = jnp.asarray(alpha)
    dim = alpha.shape[0]

    def generator(f_fn, x):

        grad_f = jax.grad(f_fn)
        hess_f = jax.hessian(f_fn)

        g = grad_f(x)
        H = hess_f(x)

        # Drift term
        alpha_sum = jnp.sum(alpha)
        drift = alpha - x * alpha_sum

        drift_term = jnp.dot(drift, g)

        # Diffusion matrix
        I = jnp.eye(dim)
        A = jnp.outer(x, x)
        diffusion = jnp.diag(x) - A

        diff_term = 0.5 * jnp.sum(diffusion * H)

        return drift_term + diff_term

    return generator

def apply_generator_to_basis(generator, basis_fn, x):
    """
    Apply generator to each basis function.

    Parameters
    ----------
    generator : callable
        Generator operator L.

    basis_fn : callable
        basis_fn(x) -> (K,) basis evaluations.

    x : (..., d)

    Returns
    -------
    Lphi : (..., K)
        Generator applied to each basis function.
    """

    phi = basis_fn(x)

    def basis_component(k):

        def f(z):
            return basis_fn(z)[k]

        return generator(f, x)

    K = phi.shape[-1]

    Lphi = jax.vmap(basis_component)(jnp.arange(K))

    return Lphi
