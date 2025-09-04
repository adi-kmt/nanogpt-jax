import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random

from attentions import MultiHeadAttention, GroupQueryAttention, MHLA
from config import GPTConfig


# Test configuration parameters
Batch = 2
SeqLen = 8
DModel = 16
NHeads = 4
DHead = 4
NKVHeads = 2  # For GQA tests


@pytest.fixture
def key():
    return random.PRNGKey(42)


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
        d_head=DHead,
        n_kv_heads=NKVHeads,
        n_layers=2,
        vocab_size=DModel
    )


@pytest.fixture
def mhla_config():
    return GPTConfig.MhlaConfig(
        d_c=8,
        d_c1=8,
        d_r=2
    )


class TestMultiHeadAttention:
    @pytest.mark.parametrize("use_rotary", [True, False])
    @pytest.mark.parametrize("use_qkNorm", [True, False])
    def test_forward_shape(self, config, key, use_rotary, use_qkNorm):
        config = config.model_copy(update={
            "use_rotary": use_rotary,
            "use_qkNorm": use_qkNorm,
        })
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (Batch, SeqLen, config.d_model))
        mask = jnp.tril(jnp.ones((SeqLen, SeqLen), dtype=bool))

        key1, key2 = random.split(key)
        y = attn(x, key=key1, mask=mask)

        assert y.shape == (Batch, SeqLen, config.d_model)

    def test_causal_masking(self, config, key):
        config = config.model_copy(update={"use_rotary": False, "use_qkNorm": False})
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (1, 4, config.d_model))

        # Full attention mask
        full_mask = jnp.ones((4, 4), dtype=bool)
        y_full = attn(x, key=key, mask=full_mask)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_causal = attn(x, key=key, mask=causal_mask)

        # Outputs should differ due to masking
        assert not jnp.allclose(y_full, y_causal, atol=1e-5)

    def test_no_mask_broadcasting_error(self, config, key):
        config = config.model_copy(update={"use_rotary": False})
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (1, 5, config.d_model))
        mask = jnp.ones((5, 5), dtype=bool)  # Should be expanded internally
        try:
            _ = attn(x, key=key, mask=mask)
        except Exception as e:
            pytest.fail(f"Mask broadcasting failed: {e}")

    def test_rotary_applied_correctly(self, key):
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
            n_kv_heads=2,
            max_seq_len=64,
            norm_eps=1e-5,
            n_layers=2,
            vocab_size=8
        )
        attn = MultiHeadAttention(config, key=key)
        x = jnp.ones((1, 3, 8))  # [B=1, T=3, D=8]
        mask = jnp.tril(jnp.ones((3, 3), dtype=bool))
        y = attn(x, key=key, mask=mask)

        assert y.shape == (1, 3, 8)
        assert jnp.all(jnp.isfinite(y))

        # Try with longer context (still within max_seq_len)
        x2 = jnp.ones((1, 50, 8))
        mask2 = jnp.tril(jnp.ones((50, 50), dtype=bool))
        y2 = attn(x2, key=key, mask=mask2)
        assert y2.shape == (1, 50, 8)

    def test_qk_norm_changes_output(self, key):
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
            n_kv_heads=2,
            max_seq_len=64,
            norm_eps=1e-5,
            n_layers=2,
            vocab_size=8
        )

        # Without qkNorm
        attn_no_norm = MultiHeadAttention(config_base, key=key)
        x = random.normal(key, (1, 4, 8))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_no_norm = attn_no_norm(x, key=key, mask=mask)

        # With qkNorm
        config_with_norm = config_base.model_copy(update={"use_qkNorm": True})
        attn_with_norm = MultiHeadAttention(config_with_norm, key=key)
        y_with_norm = attn_with_norm(x, key=key, mask=mask)

        # Outputs should differ
        assert not jnp.allclose(y_no_norm, y_with_norm, atol=1e-5)

    def test_gradient_flow(self, config, key):
        config = config.model_copy(update={"use_rotary": True, "use_qkNorm": True})
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        def loss(module, x, mask, key):
            return jnp.sum(module(x, key=key, mask=mask) ** 2)

        grads = jax.grad(loss)(attn, x, mask, key)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_deterministic_output(self, config, key):
        config = config.model_copy(update={"dropout_p": 0.0, "use_rotary": False})
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (2, 6, config.d_model))
        mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

        y1 = attn(x, key=key, mask=mask)
        y2 = attn(x, key=key, mask=mask)
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_jit_compilation(self, config, key):
        config = config.model_copy(update={"use_rotary": True})
        attn = MultiHeadAttention(config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        @jax.jit
        def forward(module, x, mask, key):
            return module(x, key=key, mask=mask)

        y = forward(attn, x, mask, key)
        assert y.shape == (2, 5, config.d_model)
        assert jnp.all(jnp.isfinite(y))


class TestGroupQueryAttention:
    @pytest.mark.parametrize("use_rotary", [True, False])
    @pytest.mark.parametrize("use_qkNorm", [True, False])
    def test_forward_shape(self, config, key, use_rotary, use_qkNorm):
        config = config.model_copy(update={
            "use_rotary": use_rotary,
            "use_qkNorm": use_qkNorm,
        })
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (Batch, SeqLen, config.d_model))
        mask = jnp.tril(jnp.ones((SeqLen, SeqLen), dtype=bool))

        key1, key2 = random.split(key)
        y = attn(x, key=key1, mask=mask)

        assert y.shape == (Batch, SeqLen, config.d_model)

    def test_kv_heads_repetition(self, config, key):
        """Test that KV heads are properly repeated to match Q heads"""
        config = config.model_copy(update={"use_rotary": False, "use_qkNorm": False})
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (1, 4, config.d_model))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        
        y = attn(x, key=key, mask=mask)
        assert y.shape == (1, 4, config.d_model)

    def test_causal_masking(self, config, key):
        config = config.model_copy(update={"use_rotary": False, "use_qkNorm": False})
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (1, 4, config.d_model))

        # Full attention mask
        full_mask = jnp.ones((4, 4), dtype=bool)
        y_full = attn(x, key=key, mask=full_mask)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_causal = attn(x, key=key, mask=causal_mask)

        # Outputs should differ due to masking
        assert not jnp.allclose(y_full, y_causal, atol=1e-5)

    def test_rotary_applied_correctly(self, key):
        config = GPTConfig(
            activation_type="gelu",
            dropout_p=0.0,
            d_model=8,
            linear_d_hidden=16,
            use_bias=False,
            use_qkNorm=False,
            tie_word_embeddings=True,
            use_rotary=True,
            n_heads=4,
            d_head=2,
            n_kv_heads=2,
            max_seq_len=64,
            norm_eps=1e-5,
            n_layers=2,
            vocab_size=8
        )
        attn = GroupQueryAttention(config, key=key)
        x = jnp.ones((1, 3, 8))  # [B=1, T=3, D=8]
        mask = jnp.tril(jnp.ones((3, 3), dtype=bool))
        y = attn(x, key=key, mask=mask)

        assert y.shape == (1, 3, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_qk_norm_changes_output(self, key):
        config_base = GPTConfig(
            activation_type="gelu",
            dropout_p=0.0,
            d_model=8,
            linear_d_hidden=16,
            use_bias=False,
            use_qkNorm=False,
            tie_word_embeddings=True,
            use_rotary=False,
            n_heads=4,
            d_head=2,
            n_kv_heads=2,
            max_seq_len=64,
            norm_eps=1e-5,
            n_layers=2,
            vocab_size=8
        )

        # Without qkNorm
        attn_no_norm = GroupQueryAttention(config_base, key=key)
        x = random.normal(key, (1, 4, 8))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_no_norm = attn_no_norm(x, key=key, mask=mask)

        # With qkNorm
        config_with_norm = config_base.model_copy(update={"use_qkNorm": True})
        attn_with_norm = GroupQueryAttention(config_with_norm, key=key)
        y_with_norm = attn_with_norm(x, key=key, mask=mask)

        # Outputs should differ
        assert not jnp.allclose(y_no_norm, y_with_norm, atol=1e-5)

    def test_gradient_flow(self, config, key):
        config = config.model_copy(update={"use_rotary": True, "use_qkNorm": True})
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        def loss(module, x, mask, key):
            return jnp.sum(module(x, key=key, mask=mask) ** 2)

        grads = jax.grad(loss)(attn, x, mask, key)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_deterministic_output(self, config, key):
        config = config.model_copy(update={"dropout_p": 0.0, "use_rotary": False})
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (2, 6, config.d_model))
        mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

        y1 = attn(x, key=key, mask=mask)
        y2 = attn(x, key=key, mask=mask)
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_jit_compilation(self, config, key):
        config = config.model_copy(update={"use_rotary": True})
        attn = GroupQueryAttention(config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        @jax.jit
        def forward(module, x, mask, key):
            return module(x, key=key, mask=mask)

        y = forward(attn, x, mask, key)
        assert y.shape == (2, 5, config.d_model)
        assert jnp.all(jnp.isfinite(y))


class TestMHLA:
    def test_forward_shape(self, config, mhla_config, key):
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (Batch, SeqLen, config.d_model))
        mask = jnp.tril(jnp.ones((SeqLen, SeqLen), dtype=bool))

        key1, key2 = random.split(key)
        y = attn(x, key=key1, mask=mask)

        assert y.shape == (Batch, SeqLen, config.d_model)

    @pytest.mark.parametrize("use_qkNorm", [True, False])
    def test_qk_norm_functionality(self, config, mhla_config, key, use_qkNorm):
        config = config.model_copy(update={"use_qkNorm": use_qkNorm})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (1, 4, config.d_model))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))

        y = attn(x, key=key, mask=mask)
        assert y.shape == (1, 4, config.d_model)

    def test_causal_masking(self, config, mhla_config, key):
        config = config.model_copy(update={"use_qkNorm": False})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (1, 4, config.d_model))

        # Full attention mask
        full_mask = jnp.ones((4, 4), dtype=bool)
        y_full = attn(x, key=key, mask=full_mask)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        y_causal = attn(x, key=key, mask=causal_mask)

        # Outputs should differ due to masking
        assert not jnp.allclose(y_full, y_causal, atol=1e-5)

    def test_rotary_applied_correctly(self, key, mhla_config):
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
            d_head=2,
            n_kv_heads=2,
            max_seq_len=64,
            norm_eps=1e-5,
            n_layers=2,
            vocab_size=8
        )
        attn = MHLA(config, mhla_config, key=key)
        x = jnp.ones((1, 3, 8))  # [B=1, T=3, D=8]
        mask = jnp.tril(jnp.ones((3, 3), dtype=bool))
        y = attn(x, key=key, mask=mask)

        assert y.shape == (1, 3, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_gradient_flow(self, config, mhla_config, key):
        config = config.model_copy(update={"use_rotary": True, "use_qkNorm": True})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        def loss(module, x, mask, key):
            return jnp.sum(module(x, key=key, mask=mask) ** 2)

        grads = jax.grad(loss)(attn, x, mask, key)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_deterministic_output(self, config, mhla_config, key):
        config = config.model_copy(update={"dropout_p": 0.0})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (2, 6, config.d_model))
        mask = jnp.tril(jnp.ones((6, 6), dtype=bool))

        y1 = attn(x, key=key, mask=mask)
        y2 = attn(x, key=key, mask=mask)
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_jit_compilation(self, config, mhla_config, key):
        config = config.model_copy(update={"use_rotary": True})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (2, 5, config.d_model))
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        @jax.jit
        def forward(module, x, mask, key):
            return module(x, key=key, mask=mask)

        y = forward(attn, x, mask, key)
        assert y.shape == (2, 5, config.d_model)
        assert jnp.all(jnp.isfinite(y))

    def test_concatenation_dimensions(self, config, mhla_config, key):
        """Test that state and rotary embeddings are properly concatenated"""
        config = config.model_copy(update={"use_rotary": True, "use_qkNorm": False})
        attn = MHLA(config, mhla_config, key=key)
        x = random.normal(key, (1, 4, config.d_model))
        mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
        
        y = attn(x, key=key, mask=mask)
        assert y.shape == (1, 4, config.d_model)