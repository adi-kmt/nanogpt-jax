import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Bool

from config import GPTConfig
from layers import Linear
from norms import norm_without_weight

# From [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L306)
class Rotary(eqx.Module):
    cos: jax.Array
    sin: jax.Array
    dim: int = eqx.static_field()
    max_seq_len: int = eqx.static_field()

    def __init__(self, dim: int, max_seq_len: int):
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Step 1: Create angular frequencies
        # (1 / 1024) ** linspace(0, 1, dim//4)
        angular_freq = (1 / 1024) ** jnp.linspace(0, 1, dim // 4, dtype=jnp.float32)
        # Step 2: Concatenate with zeros: [angular_freq, zeros]
        angular_freq = jnp.concatenate([angular_freq, jnp.zeros((dim // 4,), dtype=jnp.float32)])  # shape: [dim//2]

        # Step 3: Outer product: t[i] * angular_freq[j]
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        theta = jnp.outer(t, angular_freq)  # [max_seq_len, dim//2]

        # Step 4: Precompute cos and sin
        self.cos = jnp.cos(theta)  # [max_seq_len, dim//2]
        self.sin = jnp.sin(theta)  # [max_seq_len, dim//2]

    def __call__(self, x_BTHD: jax.Array) -> jax.Array:
        """
        Apply rotary embedding.

        Args:
            x_BTHD: tensor of shape [Batch, Time, Heads, Dim]

        Returns:
            Rotated tensor of same shape
        """
        current_seq_len = x_BTHD.shape[1]
        assert self.cos.shape[0] >= current_seq_len, (
            f"Need {current_seq_len} positions, but only have {self.cos.shape[0]} precomputed."
        )

        # Extract cos/sin for current sequence length
        cos = self.cos[:current_seq_len]  # [T, dim//2]
        sin = self.sin[:current_seq_len]  # [T, dim//2]

        # Add broadcast axes: [1, T, 1, dim//2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        # Split input on last dimension
        x_float = x_BTHD.astype(jnp.float32)
        x1, x2 = jnp.split(x_float, 2, axis=-1)  # Each: [B, T, H, dim//2]

        # Apply rotation
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos  # Note: -sin on x1

        # Recombine
        out = jnp.concatenate([y1, y2], axis=-1)  # [B, T, H, dim]
        return out.astype(x_BTHD.dtype)


class MultiHeadAttention(eqx.Module):
    w_q: Linear
    w_k: Linear
    w_v: Linear
    w_o: Linear
    config: GPTConfig
    rotary: Rotary

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.config = config
        self.w_q = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key1)
        self.w_k = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key2)
        self.w_v = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key3)
        self.w_o = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key4)
        self.rotary = Rotary(dim=config.d_model, max_seq_len=config.max_seq_len)

    def __call__(self, x: Float[Array, "batch seq_len d_model"], key: jax.random.PRNGKey,
                 mask: Bool[Array, "seq_len seq_len"]) -> Float[Array, "batch seq_len d_model"]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(q, "b s (h k) -> b h s k", h=self.config.n_heads, k=self.config.d_head)
        k = rearrange(k, "b s (h k) -> b h s k", h=self.config.n_heads, k=self.config.d_head)
        v = rearrange(v, "b s (h k) -> b h s k", h=self.config.n_heads, k=self.config.d_head)

        if self.config.use_qkNorm:
            q = norm_without_weight(q, self.config.norm_eps)
            k = norm_without_weight(k, self.config.norm_eps)

        if self.config.use_rotary:
            q = self.rotary(q)
            k = self.rotary(k)

        attn_scores = jnp.einsum('b h s k, b h t k -> b h s t', q, k)
        attn_scores = attn_scores / jnp.sqrt(self.config.d_head)

        if mask is not None:
            mask = jnp.expand_dims(mask, (0, 1))
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        out = jnp.einsum('b h s t, b h t k -> b h s k', attn_weights, v)
        out = rearrange(out, "b h s k -> b s (h k)", h=self.config.n_heads, k=self.config.d_head)
        out = self.w_o(out)

        return out
