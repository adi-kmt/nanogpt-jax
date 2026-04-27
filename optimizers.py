from __future__ import annotations

from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from config import OptimizerGroupConfig, TrainingConfig


Schedule = Callable[[jax.Array], jax.Array]


class ScheduledWeightDecayState(NamedTuple):
    count: jax.Array


def _decay_steps(config: TrainingConfig, total_steps: int) -> int:
    if config.decay_steps is not None:
        return min(config.decay_steps, max(total_steps, 1))
    return max(1, total_steps // 5)


def _scale_schedule(schedule: Schedule, multiplier: float) -> Schedule:
    return lambda count: schedule(count) * multiplier


def create_learning_rate_schedule(config: TrainingConfig, total_steps: int) -> Schedule:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    min_lr = config.lr * config.final_lr_ratio
    warmup_steps = min(config.warmup_steps, total_steps)
    remaining_steps = max(total_steps - warmup_steps, 1)

    if config.scheduler is None:
        main_schedule = optax.constant_schedule(config.lr)
    elif config.scheduler == "cosine":
        main_schedule = optax.cosine_decay_schedule(
            config.lr,
            decay_steps=remaining_steps,
            alpha=config.final_lr_ratio,
        )
    elif config.scheduler == "linear":
        main_schedule = optax.linear_schedule(config.lr, min_lr, remaining_steps)
    elif config.scheduler == "wsd":
        decay_steps = _decay_steps(config, total_steps)
        stable_steps = max(total_steps - warmup_steps - decay_steps, 0)
        stable = optax.constant_schedule(config.lr)
        decay = optax.cosine_decay_schedule(
            config.lr,
            decay_steps=decay_steps,
            alpha=config.final_lr_ratio,
        )
        if warmup_steps == 0:
            return optax.join_schedules([stable, decay], [stable_steps])
        warmup = optax.linear_schedule(0.0, config.lr, warmup_steps)
        return optax.join_schedules(
            [warmup, stable, decay],
            [warmup_steps, warmup_steps + stable_steps],
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    if warmup_steps == 0:
        return main_schedule

    warmup = optax.linear_schedule(0.0, config.lr, warmup_steps)
    return optax.join_schedules([warmup, main_schedule], [warmup_steps])


def create_weight_decay_schedule(config: TrainingConfig, total_steps: int) -> Schedule:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    final_weight_decay = (
        config.weight_decay
        if config.final_weight_decay is None
        else config.final_weight_decay
    )
    schedule_type = config.weight_decay_schedule
    warmup_steps = min(config.warmup_steps, total_steps)
    remaining_steps = max(total_steps - warmup_steps, 1)

    if schedule_type is None or schedule_type == "constant":
        return optax.constant_schedule(config.weight_decay)
    if schedule_type == "cosine":
        alpha = final_weight_decay / config.weight_decay if config.weight_decay else 0.0
        return optax.cosine_decay_schedule(config.weight_decay, remaining_steps, alpha=alpha)
    if schedule_type == "linear":
        return optax.linear_schedule(config.weight_decay, final_weight_decay, remaining_steps)
    if schedule_type == "wsd":
        decay_steps = _decay_steps(config, total_steps)
        stable_steps = max(total_steps - warmup_steps - decay_steps, 0)
        stable = optax.constant_schedule(config.weight_decay)
        if config.weight_decay == 0:
            decay = optax.constant_schedule(0.0)
        else:
            alpha = final_weight_decay / config.weight_decay
            decay = optax.cosine_decay_schedule(
                config.weight_decay,
                decay_steps=decay_steps,
                alpha=alpha,
            )
        return optax.join_schedules([stable, decay], [stable_steps])
    raise ValueError(f"Unsupported weight_decay_schedule: {schedule_type}")


def add_scheduled_decayed_weights(weight_decay_schedule: Schedule) -> optax.GradientTransformation:
    """Add decoupled weight decay between optimizer preconditioning and LR scaling."""

    def init_fn(params):
        del params
        return ScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Scheduled weight decay requires current params")
        weight_decay = weight_decay_schedule(state.count)
        updates = jax.tree_util.tree_map(
            lambda update, param: (
                None
                if update is None
                else update + weight_decay * param
            ),
            updates,
            params,
            is_leaf=lambda leaf: leaf is None,
        )
        return updates, ScheduledWeightDecayState(
            count=optax.safe_int32_increment(state.count)
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _path_names(path) -> list[str]:
    names: list[str] = []
    for item in path:
        if isinstance(item, jax.tree_util.GetAttrKey):
            names.append(item.name)
        elif isinstance(item, jax.tree_util.SequenceKey):
            names.append(str(item.idx))
        elif isinstance(item, jax.tree_util.DictKey):
            names.append(str(item.key))
    return names


def _param_traits(path, leaf) -> set[str]:
    path_names = _path_names(path)
    traits = {"all"}
    if "rotary" in path_names:
        traits.add("rotary")
    if "wte" in path_names:
        traits.add("embedding")
    if "lm_head" in path_names:
        traits.add("head")
    if any("norm" in name for name in path_names):
        traits.add("norm")
    if path_names and path_names[-1] == "bias":
        traits.add("bias")
    if leaf.ndim == 2:
        traits.add("matrix")
    else:
        traits.add("non_matrix")

    no_decay = (
        leaf.ndim < 2
        or "embedding" in traits
        or "head" in traits
        or "norm" in traits
        or "rotary" in traits
    )
    traits.add("no_decay" if no_decay else "decay")
    return traits


def optimizer_labels(model, config: TrainingConfig):
    params = eqx.filter(model, eqx.is_array)

    def label(path, leaf):
        traits = _param_traits(path, leaf)
        for group in config.optimizer_groups:
            if "default" in group.match or any(rule in traits for rule in group.match):
                return group.name
        raise ValueError(f"No optimizer group matched parameter path {_path_names(path)}")

    return jax.tree_util.tree_map_with_path(label, params)


def _adam_chain(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule | None,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if weight_decay_schedule is not None:
        transforms.append(add_scheduled_decayed_weights(weight_decay_schedule))
    transforms.extend([
        optax.scale_by_adam(
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
        ),
        optax.scale_by_learning_rate(lr_schedule),
    ])
    return optax.chain(*transforms)


def _adamw_chain(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule | None,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = [
        optax.scale_by_adam(
            b1=config.adam_b1,
            b2=config.adam_b2,
            eps=config.adam_eps,
        )
    ]
    if weight_decay_schedule is not None:
        transforms.append(add_scheduled_decayed_weights(weight_decay_schedule))
    transforms.append(optax.scale_by_learning_rate(lr_schedule))
    return optax.chain(*transforms)


def _muon_chain(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule | None,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = [
        optax.contrib.scale_by_muon(
            ns_steps=config.muon_ns_steps,
            beta=config.muon_beta,
            eps=config.adam_eps,
            nesterov=config.muon_nesterov,
            adaptive=config.muon_adaptive,
        )
    ]
    if weight_decay_schedule is not None:
        transforms.append(add_scheduled_decayed_weights(weight_decay_schedule))
    transforms.append(optax.scale_by_learning_rate(lr_schedule))
    return optax.chain(*transforms)


def _dion_chain(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule | None,
) -> optax.GradientTransformation:
    del config, lr_schedule, weight_decay_schedule
    if not hasattr(optax.contrib, "dion"):
        raise ValueError("optimizer='dion' is not available in this Optax version")
    raise ValueError("Dion support needs wiring for this Optax version's API")


def _group_weight_decay_schedule(
    base_schedule: Schedule,
    group: OptimizerGroupConfig,
) -> Schedule | None:
    if not group.weight_decay or group.optimizer == "frozen":
        return None
    return _scale_schedule(base_schedule, group.weight_decay_multiplier)


def _group_transform(
    config: TrainingConfig,
    group: OptimizerGroupConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule,
) -> optax.GradientTransformation:
    group_lr = _scale_schedule(lr_schedule, group.lr_multiplier)
    group_wd = _group_weight_decay_schedule(weight_decay_schedule, group)
    if group.optimizer == "frozen":
        return optax.set_to_zero()
    if group.optimizer == "adam":
        return _adam_chain(config, group_lr, group_wd)
    if group.optimizer == "adamw":
        return _adamw_chain(config, group_lr, group_wd)
    if group.optimizer == "muon":
        return _muon_chain(config, group_lr, group_wd)
    if group.optimizer == "dion":
        return _dion_chain(config, group_lr, group_wd)
    raise ValueError(f"Unsupported optimizer: {group.optimizer}")


def optimizer_group_metrics(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule,
    step: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group in config.optimizer_groups:
        metrics[f"optimizer/groups/{group.name}/lr"] = float(
            jax.device_get(lr_schedule(step) * group.lr_multiplier)
        )
        group_wd = 0.0
        if group.weight_decay and group.optimizer != "frozen":
            group_wd = float(
                jax.device_get(weight_decay_schedule(step) * group.weight_decay_multiplier)
            )
        metrics[f"optimizer/groups/{group.name}/weight_decay"] = group_wd
    return metrics


def create_optimizer(
    config: TrainingConfig,
    lr_schedule: Schedule,
    weight_decay_schedule: Schedule | None = None,
    model=None,
) -> optax.GradientTransformation:
    if model is None:
        raise ValueError("create_optimizer requires a model for optimizer_groups")
    if weight_decay_schedule is None:
        weight_decay_schedule = optax.constant_schedule(config.weight_decay)

    labels = optimizer_labels(model, config)
    group_by_name = {group.name: group for group in config.optimizer_groups}
    freeze_labels = jax.tree_util.tree_map(
        lambda label: (
            "frozen"
            if group_by_name[label].optimizer == "frozen"
            else "trainable"
        ),
        labels,
    )
    freeze_before_clip = optax.multi_transform(
        {
            "frozen": optax.set_to_zero(),
            "trainable": optax.identity(),
        },
        lambda _: freeze_labels,
    )
    transforms = {
        group.name: _group_transform(config, group, lr_schedule, weight_decay_schedule)
        for group in config.optimizer_groups
    }
    return optax.chain(
        freeze_before_clip,
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.multi_transform(transforms, lambda _: labels),
    )
