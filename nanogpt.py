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
    config: GPTConfig = eqx.static_field()

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
    config: GPTConfig = eqx.static_field()

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
    """Fixed weight initialization - less conservative."""
    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(model, is_leaf=eqx.is_array)))
    key_iter = iter(keys)

    def init_leaf(path, leaf):
        if not isinstance(leaf, jnp.ndarray) or leaf.ndim < 2:
            return leaf

        path_names = [p.name for p in path if isinstance(p, jax.tree_util.GetAttrKey)]
        shape = leaf.shape
        k = next(key_iter)

        # Better initialization scaling
        if 'w_out' in path_names or 'output' in path_names:
            # Output projection - small but not tiny
            scale = 0.1
        elif 'c_proj' in path_names or 'proj' in path_names:
            # Residual connection scaling
            scale = 0.02 / jnp.sqrt(config.n_layers)
        elif 'wte' in path_names or 'wpe' in path_names:
            # Embedding scaling - standard
            scale = 0.02
        elif 'ln' in path_names or 'norm' in path_names:
            # Layer norm - initialize to ones for weights, zeros for bias
            if 'weight' in path_names or 'scale' in path_names:
                return jnp.ones(shape)
            else:
                return jnp.zeros(shape)
        else:
            # General linear layers - proper Xavier/Glorot
            fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
            fan_out = shape[-1] if len(shape) >= 2 else shape[-1]
            scale = jnp.sqrt(2.0 / (fan_in + fan_out))  # Xavier initialization

        return jax.random.normal(k, shape) * scale

    return jax.tree_util.tree_map_with_path(init_leaf, model)


def debug_model_weights(model, step=0):
    """Debug function to check for problematic weights."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))

    for i, leaf in enumerate(leaves):
        if jnp.any(jnp.isnan(leaf)):
            jax.debug.print(f"⚠️  NaN found in parameter {i} at step {step}")
        if jnp.any(jnp.isinf(leaf)):
            jax.debug.print(f"⚠️  Inf found in parameter {i} at step {step}")
        if jnp.max(jnp.abs(leaf)) > 100:
            jax.debug.print(
                f"⚠️  Large weights found in parameter {i}: max={jnp.max(jnp.abs(leaf)):.3f} at step {step}")

    max_weight = max(jnp.max(jnp.abs(leaf)) for leaf in leaves)
    print(f"📊 Step {step}: Max weight magnitude: {max_weight:.4f}")