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
