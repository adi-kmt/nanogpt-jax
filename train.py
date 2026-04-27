import time
import wandb
import yaml
import os
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from jaxtyping import PRNGKeyArray
from config import DataConfig, GPTConfig, TrainingConfig, WandbConfig
from nanogpt import init_model_weights, NanoGPT, debug_model_init
from data_utils import create_dataloader, gpt2_token_bytes, setup_sharding


def load_config_from_yaml(config_path: str):
    """Load model and training configurations from YAML file."""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Handle MHLA config if present
    model_data = config_data['model']
    if 'mhla_config' in model_data and model_data['mhla_config'] is not None:
        model_data['mhla_config'] = GPTConfig.MhlaConfig(**model_data['mhla_config'])
    elif 'mhla_config' in model_data:
        # If mhla_config is None or not provided, set it to None
        model_data['mhla_config'] = None
    
    model_config = GPTConfig(**model_data)
    train_config = TrainingConfig(**config_data['training'])
    data_config = DataConfig(**config_data.get("data", {}))
    wandb_config = WandbConfig(**config_data.get("logging", {}))
    
    return model_config, train_config, data_config, wandb_config


def flatten_dict(data: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, name))
        else:
            flattened[name] = value
    return flattened


def count_parameters(model) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss) if loss < 80 else float("inf")


def setup_wandb_run(
    model_config: GPTConfig,
    train_config: TrainingConfig,
    data_config: DataConfig,
    wandb_config: WandbConfig,
    model,
):
    if not wandb_config.enabled:
        return None

    run_name = wandb_config.name or f"nanogpt_jax_{data_config.dataset}_{int(time.time())}"
    config_payload = flatten_dict({
        "model": model_config.model_dump(),
        "training": train_config.model_dump(),
        "data": data_config.model_dump(),
        "logging": wandb_config.model_dump(),
        "runtime": {
            "jax_version": jax.__version__,
            "device_count": jax.device_count(),
            "devices": [str(device) for device in jax.devices()],
            "model_parameters": count_parameters(model),
        },
    })

    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=run_name,
        group=wandb_config.group,
        job_type=wandb_config.job_type,
        tags=wandb_config.tags,
        notes=wandb_config.notes,
        mode=wandb_config.mode,
        resume=wandb_config.resume,
        id=wandb_config.id,
        config=config_payload,
    )

    if wandb_config.define_metrics:
        wandb.define_metric("step")
        wandb.define_metric("tokens_seen")
        for namespace in ("train", "eval", "system", "optimizer", "data"):
            wandb.define_metric(f"{namespace}/*", step_metric="step")
    if wandb_config.save_code:
        run.log_code(
            ".",
            include_fn=lambda path: path.endswith((".py", ".yaml", ".toml", ".md")),
            exclude_fn=lambda path: "/.venv/" in path or "/.git/" in path,
        )

    return run


def create_safer_model(config, key: PRNGKeyArray):
    """Create and initialize model with proper weights."""

    model = NanoGPT(config, key=key)
    key_model, _ = jr.split(key, 2)
    model = init_model_weights(model, key=key_model, config=config)
    model = eqx.nn.inference_mode(model, value=False)
    return model


def fetch_batch_data(data_loader, grad_accum_steps):
    """Fetch raw batches (host-side)."""
    batch_data = []
    for _ in range(grad_accum_steps):
        try:
            inputs, targets = next(data_loader)
            batch_data.append((inputs, targets))
        except StopIteration:
            return None, True
    return batch_data, False


def create_lr_schedule(config: TrainingConfig, total_steps: int):
    """Create the configured learning-rate schedule."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    decay_steps = max(total_steps - config.warmup_steps, 1)
    if config.scheduler == "cosine":
        main_schedule = optax.cosine_decay_schedule(config.lr, decay_steps, alpha=0.1)
    elif config.scheduler == "linear":
        main_schedule = optax.linear_schedule(config.lr, 0.0, decay_steps)
    elif config.scheduler is None:
        main_schedule = optax.constant_schedule(config.lr)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    if config.warmup_steps == 0:
        return main_schedule

    warmup = optax.linear_schedule(0.0, config.lr, config.warmup_steps)
    return optax.join_schedules([warmup, main_schedule], [config.warmup_steps])


def create_optimizer(config: TrainingConfig, lr_schedule):
    """Create the configured optimizer."""
    transforms = [optax.clip_by_global_norm(config.max_grad_norm)]

    if config.optimizer == "adamw":
        transforms.append(optax.adamw(learning_rate=lr_schedule, weight_decay=config.weight_decay))
    elif config.optimizer == "adam":
        if config.weight_decay:
            transforms.append(optax.add_decayed_weights(config.weight_decay))
        transforms.append(optax.adam(learning_rate=lr_schedule))
    elif config.optimizer == "muon":
        transforms.append(optax.contrib.muon(learning_rate=lr_schedule, weight_decay=config.weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    return optax.chain(*transforms)


@eqx.filter_jit
def compute_loss_and_grads_safe(model, inputs, targets, key):
    """Fixed loss computation - JAX-compatible version without Python control flow."""

    def loss_fn(model):
        # Forward pass
        logits = model(inputs, key=key, inference=False)  # [B, T, V]

        # Basic numerical stability for logits
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        logits = jnp.where(jnp.isinf(logits), jnp.sign(logits) * 10.0, logits)
        logits = jnp.clip(logits, -10.0, 10.0)

        # Reshape for cross-entropy: flatten batch and sequence dimensions
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)  # [B*T, V]
        targets_flat = targets.reshape(-1)  # [B*T]

        valid = (targets_flat >= 0) & (targets_flat < V)
        safe_targets = jnp.where(valid, targets_flat, 0)

        # Standard cross-entropy loss (no label smoothing)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, safe_targets)
        loss = jnp.sum(jnp.where(valid, loss, 0.0)) / jnp.maximum(jnp.sum(valid), 1)

        # Prevent NaN
        loss = jnp.where(jnp.isnan(loss), 1000.0, loss)

        return loss

    # Compute gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    # Conservative gradient clipping
    grads = jax.tree_util.tree_map(
        lambda g: jnp.clip(g, -1.0, 1.0),
        grads
    )

    return loss, grads


@eqx.filter_jit
def eval_batch_metrics(model, inputs, targets, token_bytes, key):
    logits = model(inputs, key=key, inference=True)
    logits = jnp.where(jnp.isnan(logits), 0.0, logits)
    logits = jnp.where(jnp.isinf(logits), jnp.sign(logits) * 10.0, logits)
    logits = jnp.clip(logits, -10.0, 10.0)

    _, _, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    valid = (targets_flat >= 0) & (targets_flat < vocab_size)
    safe_targets = jnp.where(valid, targets_flat, 0)

    losses = optax.softmax_cross_entropy_with_integer_labels(logits_flat, safe_targets)
    loss_sum = jnp.sum(jnp.where(valid, losses, 0.0))
    token_count = jnp.sum(valid)
    predictions = jnp.argmax(logits_flat, axis=-1)
    correct = jnp.sum((predictions == safe_targets) & valid)

    bytes_per_token = token_bytes[safe_targets]
    byte_mask = valid & (bytes_per_token > 0)
    nats_for_bpb = jnp.sum(jnp.where(byte_mask, losses, 0.0))
    byte_count = jnp.sum(jnp.where(valid, bytes_per_token, 0))

    return loss_sum, token_count, correct, nats_for_bpb, byte_count


def evaluate_model(model, eval_loader, eval_steps, data_sharding, key, token_bytes):
    if hasattr(eval_loader, "reset"):
        eval_loader.reset()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_nats = 0.0
    total_bytes = 0

    eval_iter = iter(eval_loader)
    for _ in range(eval_steps):
        inputs, targets = next(eval_iter)
        inputs = eqx.filter_shard(inputs, data_sharding)
        targets = eqx.filter_shard(targets, data_sharding)
        key, step_key = jax.random.split(key)
        loss_sum, token_count, correct, nats, byte_count = eval_batch_metrics(
            model, inputs, targets, token_bytes, step_key
        )

        total_loss += float(jax.device_get(loss_sum))
        total_tokens += int(jax.device_get(token_count))
        total_correct += int(jax.device_get(correct))
        total_nats += float(jax.device_get(nats))
        total_bytes += int(jax.device_get(byte_count))

    loss = total_loss / total_tokens if total_tokens else float("inf")
    metrics = {
        "eval/loss": loss,
        "eval/perplexity": safe_perplexity(loss),
        "eval/accuracy": total_correct / total_tokens if total_tokens else 0.0,
        "eval/tokens": total_tokens,
        "eval/steps": eval_steps,
    }
    if total_bytes:
        metrics["eval/bpb"] = total_nats / (math.log(2) * total_bytes)
        metrics["eval/bytes"] = total_bytes
    return metrics, key


@eqx.filter_jit
def training_step_jit_safe(model, batch_data, data_sharding, optimizer, opt_state, key):
    keys = jax.random.split(key, len(batch_data) + 1)
    new_key = keys[0]
    accumulated_grads = None
    total_loss = 0.0

    for i, (inputs, targets) in enumerate(batch_data):
        inputs = eqx.filter_shard(inputs, data_sharding)
        targets = eqx.filter_shard(targets, data_sharding)
        step_key = keys[i + 1]

        loss, grads = compute_loss_and_grads_safe(model, inputs, targets, step_key)
        total_loss += loss
        accumulated_grads = grads if accumulated_grads is None else \
            jax.tree_util.tree_map(jnp.add, accumulated_grads, grads)

    # Average gradients
    accumulated_grads = jax.tree_util.tree_map(
        lambda g: g / len(batch_data), accumulated_grads
    )
    avg_loss = total_loss / len(batch_data)

    # Safety checks
    grad_norm = optax.global_norm(accumulated_grads)
    should_update = (grad_norm < 1e4) & jnp.isfinite(grad_norm)

    # Zero out grads if invalid. The final select below also preserves the
    # original model/optimizer state, so decoupled weight decay cannot move
    # parameters on skipped steps.
    safe_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(should_update, g, jnp.zeros_like(g)),
        accumulated_grads
    )

    params = eqx.filter(model, eqx.is_array)
    updates, candidate_opt_state = optimizer.update(safe_grads, opt_state, params)
    candidate_model = eqx.apply_updates(model, updates)

    def select_if_update(new, old):
        if eqx.is_array(new):
            return jnp.where(should_update, new, old)
        return old

    new_model = jax.tree_util.tree_map(select_if_update, candidate_model, model)
    new_opt_state = jax.tree_util.tree_map(select_if_update, candidate_opt_state, opt_state)

    return new_model, new_opt_state, avg_loss, accumulated_grads, new_key


def resolve_eval_steps(config: TrainingConfig, data_config: DataConfig, eval_loader, seq_len: int) -> int:
    if data_config.eval_tokens is not None:
        requested = max(1, data_config.eval_tokens // (config.eval_batch_size * seq_len))
    else:
        requested = config.eval_steps or getattr(eval_loader, "num_steps", 1)
    return max(1, min(int(requested), getattr(eval_loader, "num_steps", int(requested))))


def log_metrics(run, metrics: dict, step: int):
    if run is not None:
        wandb.log(metrics, step=step)


def train_distributed_safe(
    model_config: GPTConfig,
    config: TrainingConfig,
    data_config: DataConfig | None = None,
    wandb_config: WandbConfig | None = None,
):
    """Fixed training with better hyperparameters."""
    data_config = data_config or DataConfig()
    wandb_config = wandb_config or WandbConfig()
    print(f"Starting training on {jax.device_count()} devices")

    key = jax.random.PRNGKey(42)
    key_model, key_train, key_eval = jax.random.split(key, 3)

    data_sharding = setup_sharding()

    # Create model with safer initialization
    model = create_safer_model(model_config, key_model)
    debug_model_init(model, model_config)

    # Rest of training setup...
    num_devices = jax.device_count()
    if config.micro_batch_size % num_devices != 0:
        raise ValueError(f"Micro batch size {config.micro_batch_size} must be divisible by device count {num_devices}")

    # Training data
    train_loader = create_dataloader(
        seq_len=model_config.max_seq_len,
        batch_size=config.micro_batch_size,
        split="train",
        data_config=data_config,
    )
    eval_loader = create_dataloader(
        seq_len=model_config.max_seq_len,
        batch_size=config.eval_batch_size,
        split="val",
        data_config=data_config,
    )

    total_steps = config.epochs * max(1, getattr(train_loader, "num_steps", 1) // config.grad_accum_steps)
    lr_schedule = create_lr_schedule(config, total_steps)
    optimizer = create_optimizer(config, lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    token_bytes = gpt2_token_bytes(model_config.vocab_size)
    eval_steps = resolve_eval_steps(config, data_config, eval_loader, model_config.max_seq_len)

    run = setup_wandb_run(model_config, config, data_config, wandb_config, model)
    if run is not None:
        run.summary["model/parameters"] = count_parameters(model)
        run.summary["data/train_tokens_per_epoch"] = getattr(train_loader, "total_tokens", None)
        run.summary["data/eval_steps"] = eval_steps

    train_iter = iter(train_loader)
    step = 0
    tokens_seen = 0
    smooth_train_loss = 0.0
    ema_beta = 0.9

    print(
        f"Training for {total_steps} steps on {data_config.dataset}; "
        f"eval_steps={eval_steps}, train_tokens/epoch={getattr(train_loader, 'total_tokens', 'unknown')}"
    )

    if config.eval_on_start:
        eval_metrics, key_eval = evaluate_model(
            model,
            eval_loader,
            eval_steps,
            data_sharding,
            key_eval,
            token_bytes,
        )
        eval_metrics.update({"step": 0, "tokens_seen": 0})
        log_metrics(run, eval_metrics, step=0)
        print(f"Eval step 0 | loss: {eval_metrics['eval/loss']:.4f} | bpb: {eval_metrics.get('eval/bpb', float('nan')):.4f}")

    # Training loop
    while step < total_steps:
        start_time = time.time()

        # Check model health less frequently
        if step % 500 == 0:
            debug_model_init(model, model_config)

        batch_data, iterator_exhausted = fetch_batch_data(train_iter, config.grad_accum_steps)

        if iterator_exhausted:
            train_iter = iter(create_dataloader(
                seq_len=model_config.max_seq_len,
                batch_size=config.micro_batch_size,
                split="train",
                data_config=data_config,
            ))
            continue

        # Training step
        model, opt_state, loss, grads, key_train = training_step_jit_safe(
            model, batch_data, data_sharding, optimizer, opt_state, key_train
        )

        step += 1
        tokens_seen += config.batch_size * model_config.max_seq_len

        if step % config.log_every == 0:
            loss_val = float(jax.device_get(loss))
            grad_norm = float(optax.global_norm(jax.device_get(grads)))
            step_time = time.time() - start_time
            current_lr = float(jax.device_get(lr_schedule(step)))
            tokens_per_sec = (config.batch_size * model_config.max_seq_len) / max(step_time, 1e-9)

            # Check for problems
            if not math.isfinite(loss_val):
                print(f"Step {step}: NaN/Inf loss detected. Loss: {loss_val}")
                break

            if grad_norm > 5.0:  # Less restrictive warning
                print(f"Step {step}: high gradient norm: {grad_norm:.3f}")

            perplexity = safe_perplexity(loss_val)
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_val
            debiased_loss = smooth_train_loss / (1 - ema_beta**step)
            log_metrics(run, {
                "step": step,
                "tokens_seen": tokens_seen,
                "train/loss": loss_val,
                "train/loss_ema": debiased_loss,
                "train/perplexity": perplexity,
                "optimizer/grad_norm": grad_norm,
                "optimizer/lr": current_lr,
                "system/step_time": step_time,
                "system/tokens_per_sec": tokens_per_sec,
                "data/epoch": getattr(train_loader, "epoch", 1),
            }, step=step)

            print(f"Step {step:5d} | Loss: {loss_val:.4f} | PPL: {perplexity:.1f} | "
                  f"Grad norm: {grad_norm:.3f} | LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}")

        should_eval = config.eval_every is not None and step % config.eval_every == 0
        if should_eval or (config.eval_on_end and step == total_steps):
            eval_metrics, key_eval = evaluate_model(
                model,
                eval_loader,
                eval_steps,
                data_sharding,
                key_eval,
                token_bytes,
            )
            eval_metrics.update({"step": step, "tokens_seen": tokens_seen})
            log_metrics(run, eval_metrics, step=step)
            if run is not None:
                best_loss = run.summary.get("eval/best_loss")
                if best_loss is None or eval_metrics["eval/loss"] < best_loss:
                    run.summary["eval/best_loss"] = eval_metrics["eval/loss"]
                    if "eval/bpb" in eval_metrics:
                        run.summary["eval/best_bpb"] = eval_metrics["eval/bpb"]
                    run.summary["eval/best_step"] = step
            print(
                f"Eval step {step} | loss: {eval_metrics['eval/loss']:.4f} | "
                f"ppl: {eval_metrics['eval/perplexity']:.1f} | "
                f"bpb: {eval_metrics.get('eval/bpb', float('nan')):.4f}"
            )

    if run is not None:
        run.summary["train/final_step"] = step
        run.summary["train/tokens_seen"] = tokens_seen
        wandb.finish()
    return model, opt_state

if __name__ == "__main__":
    # Load configurations from YAML file
    config_path = os.environ.get("TRAIN_CONFIG_PATH", "config/model_config.yaml")
    model_config, train_config, data_config, wandb_config = load_config_from_yaml(config_path)

    train_distributed_safe(
        model_config=model_config,
        config=train_config,
        data_config=data_config,
        wandb_config=wandb_config,
    )
