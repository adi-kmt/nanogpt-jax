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
                 x: Float[Array, "seq_len d_model"],
                 key: jax.random.PRNGKey,
                 mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
                 inference: bool = False) -> Float[Array, "seq_len d_model"]:
        key1, key2 = jax.random.split(key)

        h = self.attn_norm(x)
        h = self.attn(h, key=key1, mask=mask)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h, inference=inference, key=key2)
        x = x + h

        return x

class NanoGPT(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: list[DecoderBlock]
    final_norm: RMSNorm
    lm_head: Linear
    config: GPTConfig = eqx.static_field()

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        self.config = config
        key, *subkeys = jax.random.split(key, 1 + config.n_layers + 1)

        self.wte = eqx.nn.Embedding(config.vocab_size, config.d_model, key=subkeys[0])

        self.blocks = [
            DecoderBlock(config, key=subkeys[i+1]) for i in range(config.n_layers)
        ]

        self.final_norm = RMSNorm(config, key=None)

        if not config.tie_word_embeddings:
            self.lm_head = Linear(config.d_model, config.vocab_size, key=subkeys[-1], use_bias=False)
        else:
            self.lm_head = None

    def __call__(self,
                 input_ids: Int[Array, "seq_len"],
                 key: jax.random.PRNGKey,
                 mask: Optional[Bool[Array, "seq_len seq_len"]] = None,
                 inference: bool = True) -> Float[Array, "seq_len vocab_size"]:
        x = self.wte(input_ids)

        if mask is None:
            T = x.shape[0]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))  # [T, T]

        keys = jax.random.split(key, len(self.blocks))
        for block, k in zip(self.blocks, keys):
            x = block(x, key=k, mask=mask, inference=inference)

        x = self.final_norm(x)

        if self.config.tie_word_embeddings:
            logits = jnp.einsum("t d, v d -> t v", x, self.wte.weight)
        else:
            logits = self.lm_head(x)

        return logits

