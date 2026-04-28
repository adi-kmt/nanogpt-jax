import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
from config import GPTConfig
from dtype_utils import dtype_from_name, param_dtype
from jax import random

activations = {
    'relu2': lambda x: jnp.square(jnp.maximum(0, x)),
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
    compute_dtype_name: str | None = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        use_bias: bool = True,
        dtype=None,
        compute_dtype: str | None = None,
    ):
        key1, key2 = jax.random.split(key)

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype_name = compute_dtype

        if dtype is None:
            limit = 1 / jnp.power(in_features, 0.5)
            self.weight = random.uniform(
                key1, (out_features, in_features), minval=-limit, maxval=limit
            )
        else:
            limit = jnp.asarray(1 / jnp.power(in_features, 0.5), dtype=dtype)
            self.weight = random.uniform(
                key1,
                (out_features, in_features),
                dtype=dtype,
                minval=-limit,
                maxval=limit,
            )

        if use_bias:
            if dtype is None:
                self.bias = random.uniform(shape=(out_features,), key=key2)
            else:
                self.bias = random.uniform(shape=(out_features,), key=key2, dtype=dtype)
        else:
            self.bias = None

    def __call__(self, x: Float[Array, "batch seq_len d_model"]) -> Float[Array, "batch seq_len d_model"]:
        compute_dtype = dtype_from_name(self.compute_dtype_name)
        weight = self.weight
        bias = self.bias
        if compute_dtype is not None:
            x = x.astype(compute_dtype)
            weight = weight.astype(compute_dtype)
            if bias is not None:
                bias = bias.astype(compute_dtype)
        result = jnp.einsum("...i,ji->...j", x, weight)
        if bias is not None:
            result += bias
        return result

class MLP(eqx.Module):
    layer1: Linear
    gate: Linear | None
    layer2: Linear
    dropout: eqx.nn.Dropout
    activation_type: str = eqx.field(static=True)

    def __init__(self, config: GPTConfig, key: jax.random.PRNGKey):
        key1, key2, key3 = jax.random.split(key, 3)
        activation_type = "silu" if config.activation_type == "swilu" else config.activation_type
        if activation_type != "swiglu" and activation_type not in activations:
            raise ValueError(f"Unsupported activation_type: {config.activation_type}")
        weight_dtype = param_dtype(config)

        self.layer1 = Linear(
            config.d_model,
            config.linear_d_hidden,
            key=key1,
            use_bias=config.use_bias,
            dtype=weight_dtype,
            compute_dtype=config.compute_dtype,
        )
        self.gate = (
            Linear(
                config.d_model,
                config.linear_d_hidden,
                key=key2,
                use_bias=config.use_bias,
                dtype=weight_dtype,
                compute_dtype=config.compute_dtype,
            )
            if activation_type == "swiglu"
            else None
        )
        self.layer2 = Linear(
            config.linear_d_hidden,
            config.d_model,
            key=key3,
            use_bias=config.use_bias,
            dtype=weight_dtype,
            compute_dtype=config.compute_dtype,
        )
        self.activation_type = activation_type
        self.dropout = eqx.nn.Dropout(config.dropout_p)

    def __call__(self, x: Float[Array, "batch seq_len d_model"], inference: bool, key: jax.random.PRNGKey) -> Float[Array, "batch seq_len d_model"]:
        if self.activation_type == "swiglu":
            x = activations["silu"](self.layer1(x)) * self.gate(x)
        else:
            act_fn = activations[self.activation_type]
            x = act_fn(self.layer1(x))
        x = self.dropout(x, key=key, inference=inference)
        return self.layer2(x)
