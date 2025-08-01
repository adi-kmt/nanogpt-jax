from functools import partial

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import filter_grad
from jax import random
import optax

from attentions import MultiHeadAttention
from layers import Linear, MLP, activations
from config import GPTConfig
from nanogpt import DecoderBlock, NanoGPT
from norms import RMSNorm, norm_without_weight
from typing import Optional, Callable
from dataclasses import replace

Batch = 2
SeqLen = 8
DModel = 16
NHeads = 4
InFeatures = 8
OutFeatures = 16
DHead = DModel // NHeads

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
        d_head=DHead
    )

@pytest.fixture
def large_key():
    return random.PRNGKey(7)

def tree_all_close(a, b, rtol=1e-5, atol=1e-5):
    return jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), a, b
    ))

class TestActivations:
    @pytest.fixture(params=list(activations.keys()))
    def activation_name(self, request):
        return request.param

    def test_activation_output_shape(self, activation_name):
        act_fn: Callable = activations[activation_name]
        x = random.normal(random.PRNGKey(0), (10, 5))
        y = act_fn(x)
        assert y.shape == x.shape

    def test_activation_finite_output(self, activation_name):
        act_fn: Callable = activations[activation_name]
        x = random.normal(random.PRNGKey(0), (20,))
        y = act_fn(x)
        assert jnp.all(jnp.isfinite(y))

    def test_relu(self):
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert jnp.allclose(activations["relu"](x), expected)

    def test_gelu(self):
        x = jnp.array([0.0, 1.0, -1.0])
        expected = 0.5 * x * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))
        assert jnp.allclose(activations["gelu"](x), expected, atol=1e-5)

    def test_silu_swish_same(self):
        x = random.normal(random.PRNGKey(0), (5,))
        assert jnp.allclose(activations["silu"](x), activations["swish"](x), atol=1e-5)

    def test_relu2(self):
        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # relu2(x) = (max(0, x))^2
        expected = jnp.array([0.0, 0.0, 0.0, 1.0, 4.0])
        assert jnp.allclose(activations["relu2"](x), expected)

    def test_identity(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(activations["identity"](x), x)

    def test_gradient_flow(self, activation_name):
        act_fn = activations[activation_name]
        x = jnp.ones((3, 4)) * 0.1

        def loss(x):
            return jnp.sum(act_fn(x) ** 2)

        grad = jax.grad(loss)(x)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == x.shape

class TestLinear:
    def test_initialization(self, key):
        linear = Linear(InFeatures, OutFeatures, key=key, use_bias=True)
        assert linear.weight.shape == (OutFeatures, InFeatures)
        assert linear.bias is not None
        assert linear.bias.shape == (OutFeatures,)

        linear_no_bias = Linear(InFeatures, OutFeatures, key=key, use_bias=False)
        assert linear_no_bias.bias is None

    def test_forward_shape(self, key):
        linear = Linear(InFeatures, OutFeatures, key=key, use_bias=True)
        x = random.normal(key, (Batch, SeqLen, InFeatures))
        y = linear(x)
        assert y.shape == (Batch, SeqLen, OutFeatures)

    def test_no_bias_forward(self, key):
        linear = Linear(InFeatures, OutFeatures, key=key, use_bias=False)
        x = random.normal(key, (3, 4, InFeatures))
        y = linear(x)
        assert y.shape == (3, 4, OutFeatures)
        assert linear.bias is None

    def test_numerical_correctness(self, key):
        # Manually compute linear layer
        key1, key2 = random.split(key)
        weight = random.uniform(key1, (2, 3), minval=-0.5, maxval=0.5)
        bias = random.uniform(key2, (2,), minval=-0.5, maxval=0.5)
        linear = Linear(3, 2, key=key, use_bias=True)
        linear = eqx.tree_at(lambda l: l.weight, linear, weight)
        linear = eqx.tree_at(lambda l: l.bias, linear, bias)

        x = jnp.array([[1.0, 2.0, 3.0]])
        expected = jnp.einsum("bi,oi->bo", x, weight) + bias
        y = linear(x)
        assert jnp.allclose(y, expected, atol=1e-5)

    def test_gradient_flow(self, key):
        linear = Linear(InFeatures, OutFeatures, key=key, use_bias=True)
        x = random.normal(key, (2, 3, InFeatures))

        def loss(model, x):
            return jnp.sum(model(x) ** 2)

        grads = jax.grad(loss)(linear, x)
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_jit_compilation(self, key):
        linear = Linear(InFeatures, OutFeatures, key=key, use_bias=False)
        x = random.normal(key, (2, 4, InFeatures))

        @jax.jit
        def forward(model, x):
            return model(x)

        y = forward(linear, x)
        assert y.shape == (2, 4, OutFeatures)
        assert jnp.all(jnp.isfinite(y))

class TestMLP:
    @pytest.mark.parametrize("activation_type", ["relu", "gelu", "silu", "swish", "relu2"])
    def test_forward_shape(self, config, key, activation_type):
        config = config.model_copy(update={"activation_type": activation_type})
        mlp = MLP(config, key=key)
        x = random.normal(key, (Batch, SeqLen, config.d_model))
        key1, _ = random.split(key)
        y = mlp(x, inference=False, key=key1)
        assert y.shape == (Batch, SeqLen, config.d_model)

    def test_inference_vs_training_dropout(self, config, key):
        config = config.model_copy(update={"dropout_p": 0.5})
        mlp = MLP(config, key=key)
        x = random.normal(key, (2, 3, config.d_model))
        key1, key2 = random.split(key)

        y1 = mlp(x, inference=True, key=key1)
        y2 = mlp(x, inference=True, key=key1)
        assert jnp.allclose(y1, y2, atol=1e-5)

        y3 = mlp(x, inference=False, key=key1)
        y4 = mlp(x, inference=False, key=key2)
        assert not jnp.allclose(y3, y4, atol=1e-5)

    def test_zero_dropout(self, config, key):
        config = config.model_copy(update={"dropout_p": 0.0})
        mlp = MLP(config, key=key)
        x = random.normal(key, (2, 4, config.d_model))
        y1 = mlp(x, inference=False, key=key)
        y2 = mlp(x, inference=False, key=key)
        assert jnp.allclose(y1, y2, atol=1e-5)

    def test_gradient_flow(self, config, key):
        mlp = MLP(config, key=key)
        x = random.normal(key, (2, 3, config.d_model))
        key1, _ = random.split(key)

        def loss(module, x, key):
            return jnp.sum(module(x, inference=False, key=key) ** 2)

        grads = filter_grad(loss)(mlp, x, key1)

        # Check gradients are finite and exist
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))

    def test_jit_compilation(self, config, key):
        mlp = MLP(config, key=key)
        x = random.normal(key, (2, 4, config.d_model))
        key1, _ = random.split(key)

        @partial(jax.jit, static_argnames=['inference'])
        def forward(module, x, key, inference):
            return module(x, inference=inference, key=key)

        y = forward(mlp, x, key1, inference=True)
        assert y.shape == (2, 4, config.d_model)
        assert jnp.all(jnp.isfinite(y))

    def test_activation_applied_correctly(self, config, key):
        config = config.model_copy(update={"activation_type": "relu"})
        mlp = MLP(config, key=key)
        x = -jnp.ones((1, 1, config.d_model))
        y = mlp(x, inference=True, key=key)

        h = activations[mlp.activation_type](mlp.layer1(x))
        expected = mlp.layer2(h)
        assert jnp.allclose(y, expected, atol=1e-5)

        # Only assert on intermediate activation
        assert jnp.all(h >= -1e-5)  # ReLU output should be non-negative

    def test_identity_activation(self, config, key):
        config = config.model_copy(update={"activation_type": "identity"})
        mlp = MLP(config, key=key)
        x = random.normal(key, (2, 3, config.d_model))
        y = mlp(x, inference=True, key=key)
        expected = mlp.layer2(mlp.layer1(x))
        assert jnp.allclose(y, expected, atol=1e-5)

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
        assert norm.eps == config.norm_eps

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
        inv_rms = jax.lax.rsqrt(mean_sq + config.norm_eps)  # (1, 2, 1)

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

class TestDecoderBlock:
    def test_initialization(self, config, key):
        block = DecoderBlock(config, key=key)
        assert isinstance(block.attn_norm, type(config.norm_eps))
        assert isinstance(block.ffn_norm, type(config.norm_eps))
        assert isinstance(block.attn, MultiHeadAttention)
        assert isinstance(block.ffn, MLP)
        assert block.config == config

    def test_forward_pass_shape(self, config, key):
        block = DecoderBlock(config, key=key)
        x = jax.random.normal(key, (10, config.d_model))
        key1, key2 = random.split(key, 2)
        mask = jnp.tril(jnp.ones((10, 10), dtype=bool))
        y = block(x, key=key2, mask=mask, inference=False)
        assert y.shape == (10, config.d_model)

    def test_residual_connections(self, config, key):
        block = DecoderBlock(config, key=key)
        x = jax.random.normal(key, (5, config.d_model))
        key1, key2 = random.split(key, 2)
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))
        y = block(x, key=key2, mask=mask, inference=True)
        # Residuals should preserve input shape and not zero it
        assert not jnp.allclose(y, x, atol=1e-7)  # Should change
        assert jnp.all(jnp.isfinite(y))  # Should be well-behaved

    def test_inference_mode(self, config, key):
        block = DecoderBlock(config, key=key)
        x = jax.random.normal(key, (5, config.d_model))
        key1, key2 = random.split(key, 2)
        mask = jnp.tril(jnp.ones((5, 5), dtype=bool))

        # Run in inference mode
        y_inf = block(x, key=key2, mask=mask, inference=True)

        # Run in training mode
        y_train = block(x, key=key2, mask=mask, inference=False)

        # Outputs might differ slightly due to dropout
        if config.dropout > 0:
            assert not jnp.allclose(y_inf, y_train, atol=1e-5)
        else:
            assert jnp.allclose(y_inf, y_train, atol=1e-5)

class TestNanoGPT:
    def test_initialization(self, config, key):
        model = NanoGPT(config, key=key)
        assert isinstance(model.wte, eqx.nn.Embedding)
        assert len(model.blocks) == config.n_layers
        assert isinstance(model.final_norm, RMSNorm)
        assert isinstance(model.lm_head, Linear)
        assert model.config == config

    def test_tied_embeddings_initialization(self, tied_config, key):
        model = NanoGPT(tied_config, key=key)
        assert model.lm_head is None
        assert isinstance(model.wte, eqx.nn.Embedding)

    def test_forward_pass_shape(self, config, key):
        model = NanoGPT(config, key=key)
        input_ids = random.randint(key, (15,), 0, config.vocab_size)
        logits = model(input_ids, key=key, inference=True)
        assert logits.shape == (15, config.vocab_size)

    def test_mask_is_applied(self, config, key):
        model = NanoGPT(config, key=key)
        input_ids = jnp.array([1, 2, 3, 4])
        T = len(input_ids)

        # Custom upper triangle mask (should block future tokens)
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        logits_with_mask = model(input_ids, key=key, mask=mask, inference=True)

        # Try with full mask (all allowed)
        full_mask = jnp.ones((T, T), dtype=bool)
        logits_full_mask = model(input_ids, key=key, mask=full_mask, inference=True)

        # They should differ because causal masking restricts attention
        assert not jnp.allclose(logits_with_mask, logits_full_mask, atol=1e-5)

    def test_tied_embeddings_forward(self, tied_config, key):
        model = NanoGPT(tied_config, key=key)
        input_ids = random.randint(key, (10,), 0, tied_config.vocab_size)
        logits = model(input_ids, key=key, inference=True)
        assert logits.shape == (10, tied_config.vocab_size)

        # Manually verify tied weights: wte.weight used in lm_head
        expected_logits = jnp.einsum("t d, v d -> t v", model.wte(input_ids), model.wte.weight)
        assert jnp.allclose(logits, expected_logits, atol=1e-5)

    def test_deterministic_vs_randomized(self, config, key):
        model = NanoGPT(config, key=key)
        input_ids = random.randint(key, (8,), 0, config.vocab_size)

        key1, key2 = random.split(key, 2)
        out1 = model(input_ids, key=key1, inference=False)
        out2 = model(input_ids, key=key2, inference=False)

        # With dropout, different keys should give different outputs
        if config.dropout > 0:
            assert not jnp.allclose(out1, out2, atol=1e-5)
        else:
            assert jnp.allclose(out1, out2, atol=1e-5)

    def test_inference_mode_stability(self, config, key):
        model = NanoGPT(config, key=key)
        input_ids = random.randint(key, (6,), 0, config.vocab_size)

        # Multiple calls in inference mode should be deterministic
        out1 = model(input_ids, key=key, inference=True)
        out2 = model(input_ids, key=key, inference=True)
        assert jnp.allclose(out1, out2, atol=1e-5)

    def test_gradient_flow(self, config, key):
        model = NanoGPT(config, key=key)
        input_ids = random.randint(key, (5,), 0, config.vocab_size)
        target = random.randint(key, (5,), 0, config.vocab_size)

        def loss_fn(model):
            logits = model(input_ids, key=key, inference=False)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))

        grads = jax.grad(loss_fn)(model)
        # Ensure gradients exist and are finite
        assert eqx.filter_grads(lambda m: m)(model) is not None
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jnp.ndarray))


# --- Optional: Smoke test for JIT compilation ---
def test_jit_compilation(config, key):
    model = NanoGPT(config, key=key)
    input_ids = random.randint(key, (7,), 0, config.vocab_size)

    @jax.jit
    def forward(model, x, key):
        return model(x, key=key, inference=True)

    logits = forward(model, input_ids, key)
    assert logits.shape == (7, config.vocab_size)
    assert jnp.all(jnp.isfinite(logits))