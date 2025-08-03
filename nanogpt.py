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
    """Initialize model weights with proper scaling for different layer types.

    Args:
        model: The model to initialize
        key: JAX PRNGKey
        config: Model configuration (must contain d_model, n_heads, n_layers, etc.)

    Returns:
        Model with properly initialized weights
    """
    # Partition model into parameters and other parts
    params, others = eqx.partition(model, eqx.is_array)

    # Create keys for each parameter
    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(params)))
    key_iter = iter(keys)

    # Precompute useful dimensions
    d_model = config.d_model
    d_head = d_model // config.n_heads

    def init_param(path, param):
        """Initialize a single parameter based on its path."""
        k = next(key_iter)
        shape = param.shape
        path_names = [p.name for p in path if isinstance(p, jax.tree_util.GetAttrKey)]

        print(f"DEBUG: Full path: {path}")
        print(f"DEBUG: Path names: {path_names}")
        print(f"Initializing {path_names} with shape {shape}")

        # Embeddings (token and positional)
        if 'wte' in path_names or 'wpe' in path_names:
            result = jax.random.normal(k, shape) * 0.02
            print(f"DEBUG: Embedding init - original mean: {jnp.mean(param):.6f}, new mean: {jnp.mean(result):.6f}")
            return result

        # Biases (always zero)
        elif 'bias' in path_names:
            return jnp.zeros(shape)

        # Normalization layers
        elif any(n in path_names for n in ['attn_norm', 'ffn_norm', 'final_norm']):
            if 'weight' in path_names:  # Scale parameter
                return jnp.ones(shape)
            return jnp.zeros(shape)  # Shouldn't happen

        # Rotary embeddings (leave as-is)
        elif 'rotary' in path_names:
            return param

        # Attention projections (Q, K, V)
        elif 'w_q' in path_names or 'w_k' in path_names or 'w_v' in path_names:
            scale = 1.0 / jnp.sqrt(d_head)
            return jax.random.normal(k, shape) * scale

            # Attention output projection
        elif 'w_o' in path_names:
            scale = 1.0 / jnp.sqrt(d_model)
            return jax.random.normal(k, shape) * scale

        # MLP up-projection
        elif 'layer1' in path_names:  # up-projection (usually d_model -> 4*d_model)
            scale = 1.0 / jnp.sqrt(d_model)
            return jax.random.normal(k, shape) * scale

        # MLP down-projection
        elif 'layer2' in path_names:  # down-projection (usually 4*d_model -> d_model)
            scale = 1.0 / jnp.sqrt(config.linear_d_hidden)
            return jax.random.normal(k, shape) * scale

        # LM Head
        elif 'lm_head' in path_names and 'weight' in path_names:
            if config.tie_word_embeddings:
                return param  # Will be tied to embeddings
            return jax.random.normal(k, shape) * 0.02

        # Fallback for other matrices (convolutional layers, etc.)
        if len(shape) >= 2:
            fan_in = shape[-2]
            scale = 1.0 / jnp.sqrt(fan_in)
            return jax.random.normal(k, shape) * scale

        # Fallback for scalars and other params
        return jnp.zeros(shape)

    # Apply initialization to all parameters
    # new_params = jax.tree_util.tree_map_with_path(init_param, params)
    #
    # # Combine back with non-parameter parts
    # new_model = eqx.combine(new_params, others)

    new_model = jax.tree_util.tree_map_with_path(
        lambda path, x: init_param(path, x) if eqx.is_array(x) else x,
        model
    )

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
