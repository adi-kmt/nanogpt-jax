import time
import wandb

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from jaxtyping import PRNGKeyArray
from config import GPTConfig, TrainingConfig
from nanogpt import init_model_weights, NanoGPT, debug_model_init
from data_utils import create_dataloader, setup_sharding


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

        # JAX-compatible way to clamp targets to valid range
        # No Python if statements - use jnp.clip directly
        targets_flat = jnp.clip(targets_flat, 0, V - 1)

        # Standard cross-entropy loss (no label smoothing)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits_flat, targets_flat)
        loss = jnp.mean(loss)  # Average over all tokens

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

    # Zero out grads if invalid (avoid cond)
    safe_grads = jax.tree_util.tree_map(
        lambda g: jnp.where(should_update, g, jnp.zeros_like(g)),
        accumulated_grads
    )

    # Always update
    updates, new_opt_state = optimizer.update(safe_grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    return new_model, new_opt_state, avg_loss, accumulated_grads, new_key


def train_distributed_safe(model_config: GPTConfig, config: TrainingConfig):
    """Fixed training with better hyperparameters."""
    print(f"🚀 Starting SAFE distributed training on {jax.device_count()} devices")

    key = jax.random.PRNGKey(42)
    key_model, key_train = jax.random.split(key, 2)

    data_sharding = setup_sharding()

    # Create model with safer initialization
    model = create_safer_model(model_config, key_model)
    debug_model_init(model, model_config)

    # Rest of training setup...
    num_devices = jax.device_count()
    if config.micro_batch_size % num_devices != 0:
        raise ValueError(f"Micro batch size {config.micro_batch_size} must be divisible by device count {num_devices}")

    estimated_samples = 100_000
    total_steps = config.epochs * (estimated_samples // config.batch_size)

    # Better learning rate schedule
    warmup = optax.linear_schedule(0.0, config.lr, config.warmup_steps)
    decay = optax.cosine_decay_schedule(config.lr, total_steps - config.warmup_steps, alpha=0.1)
    lr_schedule = optax.join_schedules([warmup, decay], [config.warmup_steps])

    # Better optimizer settings
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.add_decayed_weights(config.weight_decay),  # Move this before Adam
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0)  # Add this for gradient descent
    )

    opt_state = optimizer.init(model)

    # Training data
    train_loader = create_dataloader(
        seq_len=model_config.max_seq_len,
        batch_size=config.micro_batch_size,
        split="train"
    )

    # Setup logging
    run_name = f"nanogpt_fixed_{int(time.time())}"

    wandb.init(project="nanogpt-equinox", name=run_name, config={
        "model_size": sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))),
        "devices": jax.device_count(),
        **vars(model_config),
        **vars(train_config),
    })

    train_iter = iter(train_loader)
    step = 0
    log_every = 10

    print(f"🎯 Training for {total_steps} steps with FIXED mode...")

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
                split="train"
            ))
            continue

        # Training step
        model, opt_state, loss, grads, key_train = training_step_jit_safe(
            model, batch_data, data_sharding, optimizer, opt_state, key_train
        )

        step += 1

        if step % log_every == 0:
            loss_val = float(jax.device_get(loss))
            grad_norm = float(optax.global_norm(jax.device_get(grads)))
            step_time = time.time() - start_time
            current_lr = lr_schedule(step)

            # Check for problems
            if jnp.isnan(loss_val) or jnp.isinf(loss_val):
                print(f"❌ Step {step}: NaN/Inf loss detected! Loss: {loss_val}")
                break

            if grad_norm > 5.0:  # Less restrictive warning
                print(f"⚠️  Step {step}: High gradient norm: {grad_norm:.3f}")

            wandb.log({
                "loss": loss_val,
                "grad_norm": grad_norm,
                "step_time": step_time,
                "lr": current_lr,
                "step": step,
                "perplexity": jnp.exp(loss_val)  # Add perplexity for easier interpretation
            })

            print(f"Step {step:5d} | Loss: {loss_val:.4f} | PPL: {jnp.exp(loss_val):.1f} | "
                  f"Grad norm: {grad_norm:.3f} | LR: {current_lr:.2e} | Time: {step_time:.3f}s")

    wandb.finish()
    return model, opt_state

if __name__ == "__main__":
    model_config = GPTConfig(
        activation_type="gelu",
        dropout_p=0.1,  # Add some dropout
        d_model=512,
        linear_d_hidden=1024,
        norm_eps=1e-5,
        use_bias=True,
        use_qkNorm=True,
        tie_word_embeddings=False,
        use_rotary=True,
        max_seq_len=64,
        n_heads=16,
        d_head=32,
        n_layers=12,
        vocab_size=50257,
    )

    train_config = TrainingConfig(
        batch_size=8,
        micro_batch_size=8,
        eval_batch_size=1,
        grad_accum_steps=2,
        epochs=1,
        lr=3e-4,
        weight_decay=0.1,
        max_grad_norm=1.0,
        warmup_steps=100,
        optimizer="adamw",
        scheduler="cosine"
    )

    train_distributed_safe(
        model_config=model_config, config=train_config
    )