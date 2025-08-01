import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
from config import GPTConfig

class RMSNorm(eqx.Module):
    weight: Float[Array, "d_model"]
    eps: float

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        self.weight = jnp.ones(config.d_model)
        self.eps = config.norm_eps

    def __call__(self, x: Float[Array, "batch seq_len d_model"]) -> Float[Array, "batch seq_len d_model"]:
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x * inv_rms) * self.weight

def norm_without_weight(x: Float[Array, "batch seq_len d_model"], eps: float) -> Float[Array, "batch seq_len d_model"]:
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
