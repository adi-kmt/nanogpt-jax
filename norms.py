import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
from config import GPTConfig
from dtype_utils import dtype_from_name, param_dtype

class RMSNorm(eqx.Module):
    weight: Float[Array, "d_model"]
    eps: float
    compute_dtype_name: str | None = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        dtype = param_dtype(config)
        self.weight = jnp.ones(config.d_model, dtype=dtype)
        self.eps = config.norm_eps
        self.compute_dtype_name = config.compute_dtype

    def __call__(self, x: Float[Array, "batch seq_len d_model"]) -> Float[Array, "batch seq_len d_model"]:
        dtype = dtype_from_name(self.compute_dtype_name) or x.dtype
        x = x.astype(dtype)
        x_float = x.astype(jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x_float), axis=-1, keepdims=True) + self.eps)
        out = (x_float * inv_rms).astype(dtype) * self.weight.astype(dtype)
        return out.astype(dtype)

def norm_without_weight(x: Float[Array, "batch seq_len d_model"], eps: float) -> Float[Array, "batch seq_len d_model"]:
    dtype = x.dtype
    x_float = x.astype(jnp.float32)
    out = x_float * jax.lax.rsqrt(jnp.mean(jnp.square(x_float), axis=-1, keepdims=True) + eps)
    return out.astype(dtype)
