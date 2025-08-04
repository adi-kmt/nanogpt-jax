import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional
from jaxtyping import Array, Float, Bool, Int
from config import GPTConfig

from attentions import MultiHeadAttention
from layers import MLP, Linear
from norms import RMSNorm


class DecoderBlock(eqx.Module):
    attn_norm: RMSNorm
    attn: MultiHeadAttention
    ffn_norm: RMSNorm
    ffn: MLP
    config: GPTConfig = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        self.config = config
        key1, key2 = jax.random.split(key)

        self.attn_norm = RMSNorm(config, key=None)
        self.ffn_norm = RMSNorm(config, key=None)

        self.attn = MultiHeadAttention(config, key=key1)

        self.ffn = MLP(config, key=key2)

    def __call__(self,
                 x: Float[Array, "batch seq_len d_model"],
                 key: jax.random.PRNGKey,
                 mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
                 inference: bool = False) -> Float[Array, "batch seq_len d_model"]:
        key1, key2 = jax.random.split(key)

        h = self.attn_norm(x)  # Now x is (B, T, D)
        h = self.attn(h, key=key1, mask=mask)  # attn expects (B, T, D)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h, inference=inference, key=key2)
        x = x + h

        return x

class NanoGPT(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: list[DecoderBlock]
    final_norm: RMSNorm
    lm_head: Optional[Linear]
    config: GPTConfig = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        self.config = config
        key, *subkeys = jax.random.split(key, 1 + config.n_layers + 1)

        self.wte = eqx.nn.Embedding(
            config.vocab_size, config.d_model, key=subkeys[0]
        )
        self.blocks = [
            DecoderBlock(config, key=subkeys[i + 1]) for i in range(config.n_layers)
        ]
        self.final_norm = RMSNorm(config, key=None)

        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                config.d_model,
                config.vocab_size,
                key=subkeys[-1],
                use_bias=False
            )
        else:
            self.lm_head = None

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"],
        key: jax.random.PRNGKey,
        mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
        inference: bool = True,
    ) -> Float[Array, "batch seq_len vocab_size"]:
        x = jax.vmap(jax.vmap(self.wte))(input_ids)

        B, T = x.shape[:2]
        if mask is None:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))

        keys = jax.random.split(key, len(self.blocks))
        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, mask=mask, inference=inference)

        x = self.final_norm(x)

        if self.config.tie_word_embeddings:
            logits = jnp.einsum("b t d, v d -> b t v", x, self.wte.weight)
        else:
            logits = self.lm_head(x)

        return logits


def init_model_weights(model, key, config):
    """Initialize model weights with proper scaling for deep networks - DEEP FIXED VERSION"""
    params, others = eqx.partition(model, eqx.is_array)

    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(params)))
    key_iter = iter(keys)

    d_model = config.d_model
    n_layers = config.n_layers

    # ✅ CRITICAL: Depth-aware scaling factors
    depth_scale = 1.0 / jnp.sqrt(n_layers)  # Scale down with depth
    residual_scale = 0.5 / jnp.sqrt(n_layers)  # Even smaller for residual connections

    def init_param(path, param):
        """Initialize a single parameter based on its path."""
        shape = param.shape
        path_names = [p.name for p in path if isinstance(p, jax.tree_util.GetAttrKey)]

        print(f"Initializing {path_names} with shape {shape}")

        if 'rotary' in path_names:
            print(f"  -> Keeping precomputed RoPE values")
            return param

        k = next(key_iter)

        if 'wte' in path_names or 'wpe' in path_names:
            std = 0.01 * depth_scale  # Much smaller for deep networks
            result = jax.random.normal(k, shape) * std
            print(f"  -> Embedding: std={std:.5f}")
            return result

        elif 'bias' in path_names:
            print(f"  -> Bias: zeros")
            return jnp.zeros(shape)

        elif any(n in path_names for n in ['attn_norm', 'ffn_norm', 'final_norm']):
            if 'weight' in path_names:
                print(f"  -> Norm weight: ones")
                return jnp.ones(shape)
            return jnp.zeros(shape)

        elif any(w in path_names for w in ['w_q', 'w_k', 'w_v']):
            # Use much smaller initialization for deep networks
            fan_in = shape[-2]
            std = (0.02 / jnp.sqrt(fan_in)) * depth_scale
            result = jax.random.normal(k, shape) * std
            print(f"  -> QKV: std={std:.6f} (depth_scale={depth_scale:.3f})")
            return result

        elif 'w_o' in path_names:
            fan_in = shape[-2]
            std = (0.01 / jnp.sqrt(fan_in)) * residual_scale
            result = jax.random.normal(k, shape) * std
            print(f"  -> Output proj: std={std:.6f} (residual_scale={residual_scale:.3f})")
            return result

        elif 'layer1' in path_names:
            fan_in, fan_out = shape[-2], shape[-1]
            std = jnp.sqrt(2.0 / fan_in) * depth_scale * 0.5
            result = jax.random.normal(k, shape) * std
            print(f"  -> MLP layer1: std={std:.6f}")
            return result

        elif 'layer2' in path_names:
            fan_in = shape[-2]
            std = (0.01 / jnp.sqrt(fan_in)) * residual_scale
            result = jax.random.normal(k, shape) * std
            print(f"  -> MLP layer2: std={std:.6f} (residual)")
            return result

        elif 'lm_head' in path_names and 'weight' in path_names:
            if config.tie_word_embeddings:
                return param
            std = 0.01 * depth_scale
            result = jax.random.normal(k, shape) * std
            print(f"  -> LM head: std={std:.5f}")
            return result

        if len(shape) >= 2:
            fan_in = shape[-2]
            std = (0.01 / jnp.sqrt(fan_in)) * depth_scale
            result = jax.random.normal(k, shape) * std
            print(f"  -> Fallback: std={std:.6f}")
            return result

        print(f"  -> Zeros fallback")
        return jnp.zeros(shape)

    # Apply initialization
    new_params = jax.tree_util.tree_map_with_path(init_param, params)
    new_model = eqx.combine(new_params, others)
    return new_model


def validate_init(params):
    """Validate parameter initialization."""
    leaves = jax.tree_util.tree_leaves(params)

    # Check for numerical issues
    for i, p in enumerate(leaves):
        if jnp.any(jnp.isnan(p)):
            raise ValueError(f"NaN detected in param {i}")
        if jnp.any(jnp.isinf(p)):
            raise ValueError(f"Inf detected in param {i}")

    # Compute statistics
    max_abs = max(jnp.max(jnp.abs(p)) for p in leaves)
    mean_abs = sum(jnp.mean(jnp.abs(p)) for p in leaves) / len(leaves)
    mean_std = sum(jnp.std(p) for p in leaves) / len(leaves)

    print(f"Initialization validation:")
    print(f"  Max absolute value: {max_abs:.4f}")
    print(f"  Mean absolute value: {mean_abs:.4f}")
    print(f"  Mean standard deviation: {mean_std:.4f}")

    if max_abs > 5.0:
        print("⚠️ Warning: Some parameters have unusually large values")
    if mean_std < 0.01 or mean_std > 1.0:
        print("⚠️ Warning: Unusual mean standard deviation")


def debug_model_init(model, config):
    """Detailed debug output for model initialization."""
    print("\n🔍 Debugging Model Initialization...")
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))

    for i, leaf in enumerate(leaves):
        print(f"Param {i}, shape {leaf.shape}: "
              f"mean={jnp.mean(leaf):.6f}, "
              f"std={jnp.std(leaf):.6f}, "
              f"norm={jnp.linalg.norm(leaf):.6f}, "
              f"min={jnp.min(leaf):.6f}, "
              f"max={jnp.max(leaf):.6f}")

    total_norm = jnp.linalg.norm(jnp.array([jnp.linalg.norm(l) for l in leaves]))
    print(f"\n✅ Initialization complete.")
    print(f"📊 Total param norm: {total_norm:.4f}")
    print(f"💡 Max absolute value: {max(jnp.max(jnp.abs(l)) for l in leaves):.6f}")
    print(f"🔧 Model specs: d_model={config.d_model}, "
          f"n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, "
          f"vocab={config.vocab_size}")
