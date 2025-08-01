import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
from config import GPTConfig
from jax import random

activations = {
    'relu2': lambda x: jnp.square(jnp.maximum(0, x) ** 2),
    'gelu': lambda x: jnp.array(0.5 * x * (1 + jax.lax.erf(x / jnp.sqrt(2.0)))),
    'silu': lambda x: x * jax.nn.sigmoid(x),  # also called SiLU
    'swish': lambda x: x * jax.nn.sigmoid(x),
    'relu': lambda x: jnp.maximum(0, x),
    'identity': lambda x: x,
}


class Linear(eqx.Module):
    weight: Float[Array, "out_features in_features"]
    bias: Float[Array, "out_features"] | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, key: jax.random.PRNGKey, use_bias: bool = True):
        key1, key2 = jax.random.split(key)

        self.in_features = in_features
        self.out_features = out_features

        limit = 1 / jnp.power(in_features, 0.5)
        self.weight = random.uniform(key1, (out_features, in_features), minval=-limit, maxval=limit)

        if use_bias:
            self.bias = random.uniform(shape=(out_features,), key=key2)
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "batch seq_len d_model"]) -> Float[Array, "batch seq_len d_model"]:
        result = jnp.einsum("...i,ji->...j", x, self.weight)
        if self.bias is not None:
            result += self.bias
        return result

class MLP(eqx.Module):
    layer1: Linear
    layer2: Linear
    activation: eqx.field(static=True)
    dropout: eqx.nn.Dropout

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        key1, key2 = jax.random.split(key)
        self.layer1 = Linear(config.d_model, config.linear_d_hidden, key=key1)
        self.layer2 = Linear(config.linear_d_hidden, config.d_model, key=key2)
        self.activation = activations[config.activation_type]
        self.dropout = eqx.nn.Dropout(config.dropout_p)

    def __call__(self, x: Float[Array, "batch seq_len d_model"], inference: bool, key: jax.random.PRNGKey) -> Float[Array, "batch seq_len d_model"]:
        x = self.dropout(self.activation(self.layer1(x)), key=key, inference=inference)
        return self.layer2(x)
