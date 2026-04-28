from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp


DTypeName = Literal["float32", "bfloat16", "float16"]


def dtype_from_name(name: DTypeName | None):
    if name is None:
        return None
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"Unsupported dtype: {name}")


def cast_floating_to_dtype(value, dtype):
    if dtype is None:
        return value
    if eqx.is_array(value) and jnp.issubdtype(value.dtype, jnp.inexact):
        return value.astype(dtype)
    return value


def cast_floating_tree(pytree, dtype):
    if dtype is None:
        return pytree
    return jax.tree_util.tree_map(lambda value: cast_floating_to_dtype(value, dtype), pytree)


def cast_floating_tree_like(pytree, reference):
    def cast_like(value, ref):
        if (
            eqx.is_array(value)
            and eqx.is_array(ref)
            and jnp.issubdtype(value.dtype, jnp.inexact)
            and jnp.issubdtype(ref.dtype, jnp.inexact)
            and value.dtype != ref.dtype
        ):
            return value.astype(ref.dtype)
        return value

    return jax.tree_util.tree_map(
        cast_like,
        pytree,
        reference,
        is_leaf=lambda leaf: leaf is None,
    )


def compute_dtype(config):
    return dtype_from_name(config.compute_dtype or config.param_dtype)


def param_dtype(config):
    return dtype_from_name(config.param_dtype)


def logits_dtype(config):
    return dtype_from_name(config.logits_dtype)
