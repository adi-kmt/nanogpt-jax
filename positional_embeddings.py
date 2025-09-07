import equinox as eqx
import jax
from jax import numpy as jnp

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

    def __call__(self, x_BTHD: jax.Array, reverse=False) -> jax.Array:
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

        if not reverse:
            # Apply rotation
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
        else:
            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos

        # Recombine
        out = jnp.concatenate([y1, y2], axis=-1)  # [B, T, H, dim]
        return out.astype(x_BTHD.dtype)
