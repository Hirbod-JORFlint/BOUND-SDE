import jax
import jax.numpy as jnp

from manifolds import create_interval_manifold
from likelihood import (
    ModelParams,
    ParamShape,
    flatten_params,
    unflatten_params,
    transform_params,
    inverse_transform_params,
    vectorized_neg_loglik,
    TreeData,
)


def test_flatten_unflatten_roundtrip():
    params = ModelParams(
        drift=jnp.array([0.1]),
        diffusion=jnp.array([0.5]),
        boundary_params=jnp.array([0.2]),
        root_prior=jnp.array([1.0]),
    )
    shape = ParamShape(1, 1, 1, 1)
    vec = flatten_params(params)
    reconstructed = unflatten_params(vec, shape)
    assert jnp.allclose(reconstructed.drift, params.drift)
    assert jnp.allclose(reconstructed.diffusion, params.diffusion)
    assert jnp.allclose(reconstructed.boundary_params, params.boundary_params)
    assert jnp.allclose(reconstructed.root_prior, params.root_prior)


def test_transform_and_inverse():
    params = ModelParams(
        drift=jnp.array([0.0]),
        diffusion=jnp.array([1.5]),
        boundary_params=jnp.array([0.0]),
        root_prior=jnp.array([0.5, 0.5]),
    )
    shape = ParamShape(1, 1, 1, 2)
    vec = flatten_params(params)
    transformed = transform_params(vec, shape)
    inverted = inverse_transform_params(transformed)
    retransformed = transform_params(inverted, shape)
    assert jnp.allclose(retransformed.drift, transformed.drift)
    assert jnp.allclose(retransformed.diffusion, transformed.diffusion)
    assert jnp.allclose(retransformed.boundary_params, transformed.boundary_params)
    assert jnp.allclose(retransformed.root_prior, transformed.root_prior)


def test_vectorized_neg_loglik_runs():
    shape = ParamShape(1, 1, 1, 1)
    model_params = ModelParams(
        drift=jnp.zeros((1,)),
        diffusion=jnp.ones((1,)),
        boundary_params=jnp.zeros((1,)),
        root_prior=jnp.array([1.0]),
    )
    vec = flatten_params(model_params)
    tree = TreeData(
        postorder_nodes=jnp.array([1, 0]),
        parent_index=jnp.array([-1, 0]),
        parent_branch=jnp.array([-1, 1]),
        branch_lengths=jnp.array([0.0, 0.1]),
        root_index=0,
    )
    node_loglik = jnp.zeros((1, 1))
    manifold = create_interval_manifold(0.0, 1.0)
    val = vectorized_neg_loglik(
        vec,
        shape,
        tree,
        node_loglik,
        manifold,
        spectral_dim=1
    )
    assert jnp.isscalar(val)
