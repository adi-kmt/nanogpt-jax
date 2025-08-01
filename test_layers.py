import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random
from typing import Optional
from typing_extensions import Annotated
from jaxtyping import Float, Int, Bool, Array

from layers import Linear, MLP
from config import GPTConfig
from nanogpt import DecoderBlock, NanoGPT
from norms import RMSNorm, norm_without_weight

Batch = 2
SeqLen = 8
DModel = 16
NHeads = 4
DHead = DModel // NHeads


@pytest.fixture
def config():
    return GPTConfig(
        activation_type="gelu",
        dropout_p=0.1,
        d_model=DModel,
        linear_d_hidden=32,
        rms_eps=1e-6,
        use_bias=True,
        use_qkNorm=False,
        tie_word_embeddings=True,
        use_rotary=False,
    )

@pytest.fixture
def large_key():
    return random.PRNGKey(7)

def tree_all_close(a, b, rtol=1e-5, atol=1e-5):
    return jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), a, b
    ))

def test_linear_shapes():
    key = random.PRNGKey(0)
    linear = Linear(in_features=128, out_features=256, key=key)

    x_2d = random.normal(key, (10, 128))  # seq_len, d_model
    x_3d = random.normal(key, (4, 10, 128))  # batch, seq_len, d_model

    assert linear(x_2d).shape == (10, 256)
    assert linear(x_3d).shape == (4, 10, 256)

def test_mlp_shapes():
    key = random.PRNGKey(0)
    key, k1 = random.split(key)

    config = GPTConfig(
        d_model=128,
        linear_d_hidden=256,
        activation_type='relu2',
        dropout_p=0.1,
        rms_eps=1e-5,
        use_bias=True,
        use_qkNorm=True,
        use_rotary=True
    )

    mlp = MLP(config, key=k1)

    seq_len, batch = 10, 4
    d_model = config.d_model

    # Test batched: (batch, seq_len, d_model)
    x_batched = random.normal(key, (batch, seq_len, d_model))
    out_batched = mlp(x_batched, key=key, inference=True)

    assert out_batched.shape == (batch, seq_len, d_model), \
        f"Batched output shape mismatch: {out_batched.shape} != {(batch, seq_len, d_model)}"

    # Test unbatched: (seq_len, d_model)
    x_unbatched = random.normal(key, (seq_len, d_model))
    out_unbatched = mlp(x_unbatched, key=key, inference=True)

    assert out_unbatched.shape == (seq_len, d_model), \
        f"Unbatched output shape mismatch: {out_unbatched.shape} != {(seq_len, d_model)}"

    # Test finite output
    assert jnp.all(jnp.isfinite(out_unbatched)), "MLP output contains NaN/inf"
    assert jnp.all(jnp.isfinite(out_batched)), "MLP output contains NaN/inf"

class TestNormWithoutWeight:
    def test_basic_rms_norm(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        eps = 1e-6
        out = norm_without_weight(x, eps)

        # Manual RMS norm
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
        expected = x / rms

        assert out.shape == x.shape
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_zero_input(self):
        x = jnp.zeros((3, 5))
        eps = 1e-6
        out = norm_without_weight(x, eps)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.allclose(out, 0.0, atol=1e-5)

    def test_gradient_flow(self):
        x = jnp.ones((2, 4)) + 0.1 * random.normal(random.PRNGKey(0), (2, 4))
        eps = 1e-6

        def loss(x):
            return jnp.sum(norm_without_weight(x, eps) ** 2)

        grad = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == x.shape

class TestRMSNorm:
    def test_initialization(self, config):
        norm = RMSNorm(config, key=None)
        assert norm.weight.shape == (config.d_model,)
        assert jnp.all(norm.weight == 1.0)
        assert norm.eps == config.rms_eps

    def test_forward_shape(self, config, large_key):
        norm = RMSNorm(config, key=None)
        x = random.normal(large_key, (Batch, SeqLen, config.d_model))
        y = norm(x)
        assert y.shape == x.shape

    def test_numerical_correctness(self, config, large_key):
        norm = RMSNorm(config, key=None)
        # Use correct d_model
        x = jnp.ones((1, 2, config.d_model)) * jnp.array([[[4.0, -4.0] + [0.0] * (config.d_model - 2),
                                                           [1.0, 1.0] + [0.0] * (config.d_model - 2)]])

        # Manual: RMS norm
        mean_sq = jnp.mean(x ** 2, axis=-1, keepdims=True)  # (1, 2, 1)
        inv_rms = jax.lax.rsqrt(mean_sq + config.rms_eps)  # (1, 2, 1)

        # Reshape weight to broadcast: (d_model,) -> (1, 1, d_model)
        weight = jnp.reshape(norm.weight, (1, 1, -1))  # (1, 1, d_model)

        expected = x * inv_rms * weight  # Now: (1,2,d) * (1,2,1) * (1,1,d) -> (1,2,d)

        y = norm(x)
        assert y.shape == x.shape
        assert jnp.allclose(y, expected, atol=1e-5)

    def test_invariance_to_scale(self, config, large_key):
        norm = RMSNorm(config, key=None)
        x = random.normal(large_key, (1, 5, config.d_model))
        y1 = norm(x)
        y2 = norm(2 * x)
        # RMSNorm makes vectors unit-RMS per token; scaled input should still normalize to similar magnitude
        assert jnp.mean(y1**2) == pytest.approx(1.0, abs=0.1)
        assert jnp.mean(y2**2) == pytest.approx(1.0, abs=0.1)

    def test_gradient_flow(self, config, large_key):
        norm = RMSNorm(config, key=None)
        x = random.normal(large_key, (3, 4, config.d_model))

        def loss(model, x):
            return jnp.sum(model(x) ** 2)

        grads = jax.grad(loss)(norm, x)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))