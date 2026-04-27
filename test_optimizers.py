import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from checkpoint_utils import (
    load_checkpoint_metadata,
    load_model_checkpoint,
    load_opt_state_checkpoint,
    save_checkpoint,
)
from config import DataConfig, GPTConfig, TrainingConfig
from data_utils import SlowRunDataLoader, describe_data_artifacts
from nanogpt import NanoGPT
from optimizers import (
    create_learning_rate_schedule,
    create_optimizer,
    create_weight_decay_schedule,
    optimizer_group_metrics,
    optimizer_labels,
)


def tiny_model_config(**overrides):
    config = {
        "activation_type": "gelu",
        "dropout_p": 0.0,
        "d_model": 16,
        "linear_d_hidden": 32,
        "use_bias": True,
        "use_qkNorm": False,
        "tie_word_embeddings": False,
        "use_rotary": True,
        "n_heads": 4,
        "d_head": 4,
        "n_kv_heads": None,
        "max_seq_len": 8,
        "norm_eps": 1e-5,
        "n_layers": 1,
        "vocab_size": 32,
    }
    config.update(overrides)
    return GPTConfig(**config)


def tiny_train_config(**overrides):
    config = {
        "batch_size": 2,
        "micro_batch_size": 1,
        "eval_batch_size": 1,
        "epochs": 1,
        "lr": 1.0,
        "weight_decay": 0.1,
        "warmup_steps": 2,
        "max_grad_norm": 10.0,
        "scheduler": "wsd",
        "optimizer_groups": [
            {
                "name": "frozen",
                "optimizer": "frozen",
                "match": ["rotary"],
                "lr_multiplier": 0.0,
                "weight_decay": False,
            },
            {
                "name": "adamw_no_decay",
                "optimizer": "adamw",
                "match": ["no_decay"],
                "lr_multiplier": 1.0,
                "weight_decay": False,
            },
            {
                "name": "adamw_decay",
                "optimizer": "adamw",
                "match": ["default"],
                "lr_multiplier": 1.0,
                "weight_decay": True,
            },
        ],
        "grad_accum_steps": 2,
        "eval_on_start": False,
        "eval_on_end": False,
    }
    config.update(overrides)
    return TrainingConfig(**config)


def path_label_map(labels):
    result = {}

    def visit(path, leaf):
        parts = []
        for item in path:
            if isinstance(item, jax.tree_util.GetAttrKey):
                parts.append(f".{item.name}")
            elif isinstance(item, jax.tree_util.SequenceKey):
                parts.append(f"[{item.idx}]")
        result["".join(parts).lstrip(".")] = leaf
        return leaf

    jax.tree_util.tree_map_with_path(visit, labels)
    return result


def test_wsd_learning_rate_schedule_has_warmup_stable_decay():
    config = tiny_train_config(lr=1.0, warmup_steps=2, decay_steps=3, final_lr_ratio=0.1)
    schedule = create_learning_rate_schedule(config, total_steps=10)

    values = jnp.asarray([schedule(i) for i in range(10)])
    assert float(values[0]) == pytest.approx(0.0)
    assert float(values[2]) == pytest.approx(1.0)
    assert float(values[6]) == pytest.approx(1.0)
    assert float(values[-1]) < float(values[6])
    assert float(values[-1]) >= 0.1


def test_wsd_weight_decay_schedule_can_warm_down_to_zero():
    config = tiny_train_config(
        weight_decay=0.2,
        final_weight_decay=0.0,
        weight_decay_schedule="wsd",
        warmup_steps=2,
        decay_steps=3,
    )
    schedule = create_weight_decay_schedule(config, total_steps=10)

    assert float(schedule(0)) == pytest.approx(0.2)
    assert float(schedule(4)) == pytest.approx(0.2)
    assert float(schedule(9)) < float(schedule(4))


def test_optimizer_labels_keep_embedding_head_norm_and_rope_out_of_decay():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    config = tiny_train_config(optimizer_groups=[
        {
            "name": "frozen",
            "optimizer": "frozen",
            "match": ["rotary"],
            "weight_decay": False,
        },
        {
            "name": "adamw_no_decay",
            "optimizer": "adamw",
            "match": ["no_decay"],
            "weight_decay": False,
        },
        {
            "name": "muon_matrix",
            "optimizer": "muon",
            "match": ["matrix"],
            "weight_decay": True,
        },
        {
            "name": "adamw_default",
            "optimizer": "adamw",
            "match": ["default"],
            "weight_decay": True,
        },
    ])
    labels = path_label_map(optimizer_labels(model, config))

    assert labels["wte.weight"] == "adamw_no_decay"
    assert labels["lm_head.weight"] == "adamw_no_decay"
    assert labels["blocks[0].attn_norm.weight"] == "adamw_no_decay"
    assert labels["blocks[0].attn.rotary.cos"] == "frozen"
    assert labels["blocks[0].attn.w_q.weight"] == "muon_matrix"
    assert labels["blocks[0].ffn.layer1.weight"] == "muon_matrix"


def test_adamw_grouping_decays_linear_weights_but_not_embeddings_or_rope():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    params_before = eqx.filter(model, eqx.is_array)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params_before)

    config = tiny_train_config(
        scheduler=None,
        warmup_steps=0,
        lr=0.1,
        weight_decay=0.1,
        weight_decay_schedule="constant",
    )
    optimizer = create_optimizer(
        config,
        create_learning_rate_schedule(config, total_steps=2),
        create_weight_decay_schedule(config, total_steps=2),
        model,
    )
    updates, _ = optimizer.update(grads, optimizer.init(params_before), params_before)
    model_after = eqx.apply_updates(model, updates)

    assert jnp.allclose(model_after.wte.weight, model.wte.weight)
    assert jnp.allclose(model_after.lm_head.weight, model.lm_head.weight)
    assert jnp.allclose(model_after.blocks[0].attn.rotary.cos, model.blocks[0].attn.rotary.cos)
    assert not jnp.allclose(model_after.blocks[0].attn.w_q.weight, model.blocks[0].attn.w_q.weight)


def test_muon_grouping_update_runs_and_keeps_frozen_arrays_fixed():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    params = eqx.filter(model, eqx.is_array)
    grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)

    config = tiny_train_config(
        optimizer_groups=[
            {
                "name": "frozen",
                "optimizer": "frozen",
                "match": ["rotary"],
                "weight_decay": False,
            },
            {
                "name": "adamw_no_decay",
                "optimizer": "adamw",
                "match": ["no_decay"],
                "weight_decay": False,
            },
            {
                "name": "muon_matrix",
                "optimizer": "muon",
                "match": ["matrix"],
                "weight_decay": True,
            },
            {
                "name": "adamw_default",
                "optimizer": "adamw",
                "match": ["default"],
                "weight_decay": True,
            },
        ],
        scheduler=None,
        warmup_steps=0,
        lr=0.01,
        weight_decay=0.01,
    )
    optimizer = create_optimizer(
        config,
        create_learning_rate_schedule(config, total_steps=2),
        create_weight_decay_schedule(config, total_steps=2),
        model,
    )
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    model_after = eqx.apply_updates(model, updates)

    leaves = jax.tree_util.tree_leaves(eqx.filter(model_after, eqx.is_array))
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert jnp.allclose(model_after.blocks[0].attn.rotary.sin, model.blocks[0].attn.rotary.sin)


def test_frozen_group_is_zeroed_before_global_clipping():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    params = eqx.filter(model, eqx.is_array)
    base_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    base_grads = eqx.tree_at(
        lambda tree: tree.blocks[0].attn.w_q.weight,
        base_grads,
        jnp.ones_like(base_grads.blocks[0].attn.w_q.weight),
    )
    large_frozen_grads = eqx.tree_at(
        lambda tree: tree.blocks[0].attn.rotary.cos,
        base_grads,
        jnp.ones_like(base_grads.blocks[0].attn.rotary.cos) * 1e6,
    )
    config = tiny_train_config(
        scheduler=None,
        warmup_steps=0,
        lr=0.01,
        weight_decay=0.0,
        max_grad_norm=0.1,
    )
    optimizer = create_optimizer(
        config,
        create_learning_rate_schedule(config, total_steps=2),
        create_weight_decay_schedule(config, total_steps=2),
        model,
    )
    opt_state = optimizer.init(params)

    base_updates, _ = optimizer.update(base_grads, opt_state, params)
    frozen_updates, _ = optimizer.update(large_frozen_grads, opt_state, params)

    assert jnp.allclose(
        base_updates.blocks[0].attn.w_q.weight,
        frozen_updates.blocks[0].attn.w_q.weight,
    )


def test_group_lr_and_weight_decay_multipliers_are_reported():
    config = tiny_train_config(
        lr=1.0,
        weight_decay=0.2,
        warmup_steps=0,
        scheduler=None,
        weight_decay_schedule="constant",
        optimizer_groups=[
            {
                "name": "muon_matrix",
                "optimizer": "muon",
                "match": ["matrix"],
                "lr_multiplier": 3.0,
                "weight_decay": True,
                "weight_decay_multiplier": 0.5,
            },
            {
                "name": "adamw_default",
                "optimizer": "adamw",
                "match": ["default"],
                "lr_multiplier": 0.25,
                "weight_decay": False,
            },
        ],
    )
    metrics = optimizer_group_metrics(
        config,
        create_learning_rate_schedule(config, total_steps=2),
        create_weight_decay_schedule(config, total_steps=2),
        step=0,
    )

    assert metrics["optimizer/groups/muon_matrix/lr"] == pytest.approx(3.0)
    assert metrics["optimizer/groups/muon_matrix/weight_decay"] == pytest.approx(0.1)
    assert metrics["optimizer/groups/adamw_default/lr"] == pytest.approx(0.25)
    assert metrics["optimizer/groups/adamw_default/weight_decay"] == pytest.approx(0.0)


def test_dion_group_fails_clearly_until_optax_supports_it():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    config = tiny_train_config(optimizer_groups=[
        {
            "name": "dion_matrix",
            "optimizer": "dion",
            "match": ["matrix"],
        },
        {
            "name": "adamw_default",
            "optimizer": "adamw",
            "match": ["default"],
            "weight_decay": False,
        },
    ])

    with pytest.raises(ValueError, match="dion"):
        create_optimizer(
            config,
            create_learning_rate_schedule(config, total_steps=2),
            create_weight_decay_schedule(config, total_steps=2),
            model,
        )


def test_rotary_tables_are_true_constants_for_gradients():
    model = NanoGPT(tiny_model_config(), key=jax.random.PRNGKey(0))
    input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)

    def loss_fn(candidate):
        return jnp.sum(candidate(input_ids, key=jax.random.PRNGKey(1), inference=True))

    grads = eqx.filter_grad(loss_fn)(model)

    assert jnp.allclose(grads.blocks[0].attn.rotary.cos, 0.0)
    assert jnp.allclose(grads.blocks[0].attn.rotary.sin, 0.0)


def test_checkpoint_round_trip(tmp_path):
    config = tiny_model_config()
    model = NanoGPT(config, key=jax.random.PRNGKey(0))
    input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
    logits_before = model(input_ids, key=jax.random.PRNGKey(1), inference=True)
    train_config = tiny_train_config(scheduler=None, warmup_steps=0)
    optimizer = create_optimizer(
        train_config,
        create_learning_rate_schedule(train_config, total_steps=2),
        create_weight_decay_schedule(train_config, total_steps=2),
        model,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metadata = {
        "step": 3,
        "tokens_seen": 128,
        "metrics": {"eval/loss": 1.23},
    }
    checkpoint_path = save_checkpoint(tmp_path / "ckpt", model=model, opt_state=opt_state, metadata=metadata)
    loaded = load_model_checkpoint(checkpoint_path, NanoGPT(config, key=jax.random.PRNGKey(0)))
    loaded_opt_state = load_opt_state_checkpoint(checkpoint_path, opt_state)
    logits_after = loaded(input_ids, key=jax.random.PRNGKey(1), inference=True)

    assert jnp.allclose(logits_before, logits_after)
    assert len(jax.tree_util.tree_leaves(loaded_opt_state)) == len(jax.tree_util.tree_leaves(opt_state))
    loaded_metadata = load_checkpoint_metadata(checkpoint_path)
    assert loaded_metadata["step"] == 3
    assert len(loaded_metadata["checkpoint_files"]["model.eqx"]["sha256"]) == 64
    assert len(loaded_metadata["checkpoint_files"]["opt_state.eqx"]["sha256"]) == 64


def test_slowrun_npz_loader_and_artifact_hash(tmp_path):
    bos_id = 50256
    tokens = np.asarray([
        bos_id, 1, 2, 3, 4,
        bos_id, 5, 6, 7, 8,
    ], dtype=np.uint16)
    doc_starts = np.asarray([0, 5], dtype=np.int64)
    data_path = tmp_path / "fineweb_val.npz"
    np.savez(
        data_path,
        tokens=tokens,
        doc_starts=doc_starts,
        bos_id=np.asarray(bos_id, dtype=np.int64),
        seq_shuffle_seed=np.asarray(42, dtype=np.int64),
        seq_size=np.asarray(5, dtype=np.int64),
    )

    loader = SlowRunDataLoader(str(data_path), batch_size=2, seq_len=4)
    inputs, targets = next(loader)
    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)

    artifacts = describe_data_artifacts(
        DataConfig(dataset="slowrun", data_dir=str(tmp_path), data_format="npz"),
        splits=("val",),
    )
    assert artifacts["files"]["val"]["exists"]
    assert len(artifacts["files"]["val"]["sha256"]) == 64
