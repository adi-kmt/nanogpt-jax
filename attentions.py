import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Bool

from config import GPTConfig
from layers import Linear
from norms import norm_without_weight
from positional_embeddings import Rotary
from jax.lax import ragged_dot_general

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

class GroupQueryAttention(eqx.Module):
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
        self.w_k = Linear(config.d_model, (config.n_kv_heads * config.d_head), use_bias=config.use_bias, key=key2)
        self.w_v = Linear(config.d_model, (config.n_kv_heads * config.d_head), use_bias=config.use_bias, key=key3)
        self.w_o = Linear(config.d_model, config.d_model, use_bias=config.use_bias, key=key4)
        self.rotary = Rotary(dim=config.d_head, max_seq_len=config.max_seq_len)

    def __call__(self, x: Float[Array, "batch seq_len d_model"], key: jax.random.PRNGKey,
                 mask: Bool[Array, "seq_len seq_len"] = None) -> Float[Array, "batch seq_len d_model"]:

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.config.n_heads, d=self.config.d_head)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.config.n_kv_heads, d=self.config.d_head)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.config.n_kv_heads, d=self.config.d_head)

        # Repeat kv_heads to match n_heads
        repeat_factor = self.config.n_heads // self.config.n_kv_heads
        k = jnp.repeat(k, repeat_factor, -2)
        v = jnp.repeat(v, repeat_factor, -2)

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

# Deeply inspired by this video from [Vizuara](https://youtu.be/m1x8vA_Tscc)
class MHLA(eqx.Module):
    w_o: Linear
    w_dq: Linear
    w_uq: Linear
    w_rq: Linear
    w_dkv: Linear
    w_rk: Linear
    w_uk: Linear
    w_uv: Linear
    rotary: Rotary
    config: GPTConfig = eqx.field(static=True)
    mhla_config: GPTConfig.MhlaConfig = eqx.field(static=True)

    def __init__(self, config: GPTConfig, mhla_config: GPTConfig.MhlaConfig, key: jax.random.PRNGKey):
        self.config = config
        self.mhla_config = mhla_config

        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        self.w_dq = Linear(config.d_model, mhla_config.d_c1, use_bias=False, key=k1)
        self.w_uq = Linear(mhla_config.d_c1, config.d_model, use_bias=False, key=k2)
        self.w_rq = Linear(mhla_config.d_c1, config.n_heads * mhla_config.d_r, use_bias=False, key=k3)
        self.w_dkv = Linear(config.d_model, mhla_config.d_c, use_bias=False, key=k4)
        self.w_rk = Linear(config.d_model, mhla_config.d_r, use_bias=False, key=k5)
        self.w_uk = Linear(mhla_config.d_c, config.d_model, use_bias=False, key=k6)
        self.w_uv = Linear(mhla_config.d_c, config.d_model, use_bias=False, key=k7)
        self.w_o = Linear(config.d_model, config.d_model, use_bias=False, key=k8)

        self.rotary = Rotary(dim=mhla_config.d_r, max_seq_len=config.max_seq_len)

    def __call__(
        self,
        x: Float[Array, "batch seq_len d_model"],
        key: jax.random.PRNGKey,
        mask: Bool[Array, "seq_len seq_len"] = None,
    ) -> Float[Array, "batch seq_len d_model"]:
        b, t, _ = x.shape

        # Compress
        c_q = self.w_dq(x)  # [b, t, d_c1]
        c_kv = self.w_dkv(x)  # [b, t, d_c]

        # State path
        q_state = self.w_uq(c_q)  # [b, t, d_model]
        k_state = self.w_uk(c_kv)  # [b, t, d_model]
        v_state = self.w_uv(c_kv)  # [b, t, d_model]

        # RoPE path
        q_rot = self.w_rq(c_q)  # [b, t, n_heads * d_r]
        k_rot = self.w_rk(x)  # [b, t, d_r]

        # Reshape to heads: [b, t, h, ...]
        q_state = rearrange(q_state, "b t (h d) -> b t h d", h=self.config.n_heads)
        k_state = rearrange(k_state, "b t (h d) -> b t h d", h=self.config.n_heads)
        v_state = rearrange(v_state, "b t (h d) -> b t h d", h=self.config.n_heads)

        q_rot = rearrange(q_rot, "b t (h r) -> b t h r", h=self.config.n_heads)
        k_rot = rearrange(k_rot, "b t r -> b t 1 r")
        k_rot = jnp.broadcast_to(k_rot, (b, t, self.config.n_heads, self.mhla_config.d_r))  # [b, t, h, r]

        # Apply RoPE: expects [b, t, h, d]
        q_rot = self.rotary(q_rot)
        k_rot = self.rotary(k_rot)

        # Concatenate: [state; rot]
        q_final = jnp.concatenate([q_state, q_rot], axis=-1)  # [b, t, h, d_head + d_r]
        k_final = jnp.concatenate([k_state, k_rot], axis=-1)  # [b, t, h, d_head + d_r]

        # Optional QK Norm
        if self.config.use_qkNorm:
            q_final = norm_without_weight(q_final, self.config.norm_eps)
            k_final = norm_without_weight(k_final, self.config.norm_eps)

        # Transpose for attention: [b, h, t, d]
        q_final = rearrange(q_final, "b t h d -> b h t d")
        k_final = rearrange(k_final, "b t h d -> b h t d")
        v_state = rearrange(v_state, "b t h d -> b h t d")

        # Attention: Q @ K^T
        attn_scores = jnp.einsum("b h s d, b h t d -> b h s t", q_final, k_final)
        attn_scores = attn_scores / jnp.sqrt(self.config.d_head + self.mhla_config.d_r)

        # Mask
        if mask is not None:
            # Expand mask to [B, H, S, S] format
            mask = jnp.expand_dims(mask, (0, 1))  # [1, 1, S, S]
            attn_scores = jnp.where(mask, attn_scores, -jnp.inf)

        # Softmax
        weights = jax.nn.softmax(attn_scores, axis=-1)

        # Output: weights @ V
        outputs = jnp.einsum("b h s t, b h t d -> b h s d", weights, v_state)

        # Back to [b, t, d_model]
        outputs = rearrange(outputs, "b h t d -> b t (h d)")

        # Final projection
        outputs = self.w_o(outputs)

        return outputs
