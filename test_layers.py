import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random
from typing import Optional
from typing_extensions import Annotated
from jaxtyping import Float, Int, Bool, Array

from attentions import MultiHeadAttention
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
        norm_eps=1e-6,
        use_bias=True,
        use_qkNorm=False,
        tie_word_embeddings=True,
        use_rotary=False,
        max_seq_len=SeqLen,
        n_heads=NHeads,
        d_head=DHead
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

class TestMultiHeadAttention:
    @pytest.mark.parametrize("use_rotary", [True, False])
    @pytest.mark.parametrize("use_qkNorm", [True, False])
    def test_forward_shape(self, config, large_key, use_rotary, use_qkNorm):
        config = config.copy(
            update={
                "use_rotary": use_rotary,
                "use_qkNorm": use_qkNorm,
            }
        )
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (Batch, SeqLen, config.d_model))
        mask = jnp.tril(jnp.ones((SeqLen, SeqLen), dtype=bool))

        key1, key2 = random.split(large_key)
        y = attn(x, key=key1, mask=mask)

        assert y.shape == (Batch, SeqLen, config.d_model)

    def test_causal_masking(self, config, large_key):
        config = config.copy(update={"use_rotary": False, "use_qkNorm": False})
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (1, 4, config.d_model))

        # Full attention mask
        full_mask = jnp.ones((4, 4), dtype=bool)
        y_full = attn(x, key=large_key, mask=full_mask)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_causal = attn(x, key=large_key, mask=causal_mask)

        # Outputs should differ due to masking
        assert not jnp.allclose(y_full, y_causal, atol=1e-5)

    def test_no_mask_broadcasting_error(self, config, large_key):
        config = config.copy(update={"use_rotary": False})
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (1, 5, config.d_model))
        mask = jnp.ones((5, 5), dtype=bool)  # Should be expanded internally
        try:
            _ = attn(x, key=large_key, mask=mask)
        except Exception as e:
            pytest.fail(f"Mask broadcasting failed: {e}")

    def test_rotary_applied_correctly(self, large_key):
        config = GPTConfig(
            activation_type="gelu",
            dropout_p=0.0,
            d_model=8,
            linear_d_hidden=16,
            use_bias=False,
            use_qkNorm=False,
            tie_word_embeddings=True,
            use_rotary=True,
            n_heads=2,
            d_head=4,
            max_seq_len=64,
            norm_eps=1e-5,
        )
        attn = MultiHeadAttention(config, key=large_key)
        x = jnp.ones((1, 3, 8))  # [B=1, T=3, D=8]
        mask = jnp.tril(jnp.ones((3, 3), dtype=bool))
        y = attn(x, key=large_key, mask=mask)

        assert y.shape == (1, 3, 8)
        assert jnp.all(jnp.isfinite(y))

        # Try with longer context (still within max_seq_len)
        x2 = jnp.ones((1, 50, 8))
        mask2 = jnp.tril(jnp.ones((50, 50), dtype=bool))
        y2 = attn(x2, key=large_key, mask=mask2)
        assert y2.shape == (1, 50, 8)

    def test_qk_norm_changes_output(self, large_key):
        config_base = GPTConfig(
            activation_type="gelu",
            dropout_p=0.0,
            d_model=8,
            linear_d_hidden=16,
            use_bias=False,
            use_qkNorm=False,
            tie_word_embeddings=True,
            use_rotary=False,
            n_heads=2,
            d_head=4,
            max_seq_len=64,
            norm_eps=1e-5,
        )

        # Without qkNorm
        attn_no_norm = MultiHeadAttention(config_base, key=large_key)
        x = random.normal(large_key, (1, 4, 8))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_no_norm = attn_no_norm(x, key=large_key, mask=mask)

        # With qkNorm
        config_with_norm = config_base.copy(update={"use_qkNorm": True})
        attn_with_norm = MultiHeadAttention(config_with_norm, key=large_key)
        y_with_norm = attn_with_norm(x, key=large_key, mask=mask)

        # Outputs should differ
        assert not jnp.allclose(y_no_norm, y_with_norm, atol=1e-5)

    def test_gradient_flow(self, config, large_key):
        config = config.copy(update={"use_rotary": True, "use_qkNorm": True})
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        def loss(module, x, mask, key):
            return jnp.sum(module(x, key=key, mask=mask) ** 2)

        grads = jax.grad(loss)(attn, x, mask, large_key)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_deterministic_output(self, config, large_key):
        config = config.copy(update={"dropout_p": 0.0, "use_rotary": False})
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (2, 6, config.d_model))
        mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

        y1 = attn(x, key=large_key, mask=mask)
        y2 = attn(x, key=large_key, mask=mask)
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_jit_compilation(self, config, large_key):
        config = config.copy(update={"use_rotary": True})
        attn = MultiHeadAttention(config, key=large_key)
        x = random.normal(large_key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        @jax.jit
        def forward(module, x, mask, key):
            return module(x, key=key, mask=mask)

        y = forward(attn, x, mask, large_key)
        assert y.shape == (2, 5, config.d_model)
        assert jnp.all(jnp.isfinite(y))