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
    dim: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

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
    config: GPTConfig = eqx.field(static=True)
    rotary: Rotary

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.config = config
        self.w_q = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key1)
        self.w_k = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key2)
        self.w_v = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key3)
        self.w_o = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key4)
        self.rotary = Rotary(dim=config.d_head, max_seq_len=config.max_seq_len)

    def __call__(self, x: Float[Array, "batch seq_len d_model"], key: jax.random.PRNGKey,
                 mask: Bool[Array, "seq_len seq_len"]) -> Float[Array, "batch seq_len d_model"]:

        # Project to Q, K, V
        q = self.w_q(x)  # [B, S, D]
        k = self.w_k(x)  # [B, S, D]
        v = self.w_v(x)  # [B, S, D]

        # Reshape to BTHD format (Batch, Time, Heads, Dim_head)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.config.n_heads, d=self.config.d_head)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.config.n_heads, d=self.config.d_head)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.config.n_heads, d=self.config.d_head)

        # Apply QK normalization if enabled
        if self.config.use_qkNorm:
            q = norm_without_weight(q, self.config.norm_eps)
            k = norm_without_weight(k, self.config.norm_eps)

        # Apply rotary embeddings if enabled (now expects BTHD format)
        if self.config.use_rotary:
            q = self.rotary(q)  # [B, T, H, D]
            k = self.rotary(k)  # [B, T, H, D]

        # Compute attention scores: [B, H, S, S]
        # Need to transpose for einsum: BTHD -> BHTD
        q_transposed = jnp.transpose(q, (0, 2, 1, 3))  # [B, H, T, D]
        k_transposed = jnp.transpose(k, (0, 2, 1, 3))  # [B, H, T, D]
        v_transposed = jnp.transpose(v, (0, 2, 1, 3))  # [B, H, T, D]

        attn_scores = jnp.einsum('b h s d, b h t d -> b h s t', q_transposed, k_transposed)
        attn_scores = attn_scores / jnp.sqrt(self.config.d_head)

        # Apply causal mask
        if mask is not None:
            # Expand mask to [B, H, S, S] format
            mask = jnp.expand_dims(mask, (0, 1))  # [1, 1, S, S]
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)

        # Apply softmax
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Apply attention to values: [B, H, S, D_head]
        out = jnp.einsum('b h s t, b h t d -> b h s d', attn_weights, v_transposed)

        # Transpose back to [B, S, H, D] then reshape to [B, S, D]
        out = jnp.transpose(out, (0, 2, 1, 3))  # [B, S, H, D]
        out = rearrange(out, "b s h d -> b s (h d)", h=self.config.n_heads, d=self.config.d_head)

        # Final output projection
        out = self.w_o(out)

        return out