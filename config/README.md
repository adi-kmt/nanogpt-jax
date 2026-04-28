# Configuration System

This project uses YAML files for configuration instead of hardcoded values. This makes it easy to experiment with different model architectures and training settings.

## Configuration Files

The configuration files are located in the `config/` directory:

1. `model_config.yaml` - Standard model configuration with MHA attention
2. `model_config_mhla.yaml` - Model configuration with MHLA attention
3. `model_config_small.yaml` - Small model configuration for smoke tests
4. `model_config_slowrun.yaml` - FineWeb/Slowrun competition data configuration
5. `model_config_template.yaml` - Fully annotated template with dtype, optimizer groups, checkpoints, data, and W&B settings

## Running Training with Different Configurations

To run training with a specific configuration, set the `TRAIN_CONFIG_PATH` environment variable:

```bash
# Run with standard MHA configuration
TRAIN_CONFIG_PATH=config/model_config.yaml python train.py

# Run with MHLA configuration
TRAIN_CONFIG_PATH=config/model_config_mhla.yaml python train.py

# Run with small model for testing
TRAIN_CONFIG_PATH=config/model_config_small.yaml python train.py

# Run with Slowrun FineWeb data
uv sync --extra slowrun
uv run python prepare_slowrun_data.py
TRAIN_CONFIG_PATH=config/model_config_slowrun.yaml uv run python train.py

# Start a new run from the full template
cp config/model_config_template.yaml config/my_run.yaml
TRAIN_CONFIG_PATH=config/my_run.yaml uv run python train.py

# Run exact full Slowrun validation from a checkpoint
uv run python scripts/eval_slowrun.py \
  --config config/model_config_slowrun.yaml \
  --checkpoint checkpoints/slowrun/best_eval \
  --full
```

## Creating Custom Configurations

You can create your own configuration files by copying and modifying existing ones:

```bash
cp config/model_config.yaml config/my_custom_config.yaml
# Edit config/my_custom_config.yaml with your desired settings
TRAIN_CONFIG_PATH=config/my_custom_config.yaml python train.py
```

## Configuration Structure

Each YAML file can define four sections:

### Model Configuration
- `activation_type`: Activation function (`gelu`, `relu`, `relu2`, `silu`, `swish`, `identity`, or `swiglu`)
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `attention_type`: Attention mechanism (`mha`, `gqa`, `mhla`, or `vo-mhla`)
- `mhla_config`: MHLA-specific parameters (only needed for MHLA attention)
- `param_dtype`: Stored floating parameter dtype (`float32`, `bfloat16`, or `float16`)
- `compute_dtype`: Forward activation/matmul dtype; set `null` to follow `param_dtype`
- `logits_dtype`: Dtype returned by the model head; training/eval loss still upcasts logits to float32

### Training Configuration
- `batch_size`: Effective training batch size; must equal `micro_batch_size * grad_accum_steps`
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `scheduler`: Learning-rate schedule (`cosine`, `linear`, `wsd`, or `null`)
- `final_lr_ratio`: End LR as a ratio of peak LR for decay schedules
- `decay_steps`: Final warmdown length; defaults to the last 20% of the run
- `weight_decay_schedule`: Weight-decay schedule (`constant`, `cosine`, `linear`, `wsd`, or `null`)
- `final_weight_decay`: End weight decay for scheduled decay
- `optimizer_groups`: Ordered first-match optimizer groups. Each group has `name`, `optimizer`, `match`, `lr_multiplier`, `weight_decay`, and `weight_decay_multiplier`.
- Optimizer names: `adam`, `adamw`, `muon`, `frozen`; `dion` is accepted for future Optax support and currently fails clearly if selected.
- Match rules: `rotary`, `embedding`, `head`, `norm`, `bias`, `matrix`, `non_matrix`, `decay`, `no_decay`, `default`, or `all`
- `optimizer_state_dtype`: Adam/Muon accumulator dtype; `float32` is recommended even when `param_dtype` is `bfloat16`
- `log_every`: Number of optimizer steps between train logs
- `eval_every`: Number of optimizer steps between validation runs
- `eval_steps`: Fixed number of eval batches; set to `null` to use `data.eval_tokens` or the full eval loader
- `checkpoint_dir`: Directory for `best_eval`, `last`, and periodic checkpoints
- `save_every`: Optional number of steps between periodic checkpoints
- `save_best`, `save_last`: Toggle best/final checkpoint writes

### Data Configuration
- `dataset`: `tinyshakespeare` or `slowrun`
- `data_dir`: Directory containing Slowrun `fineweb_train.npz`/`fineweb_val.npz` or `.pt` files
- `data_format`: `auto`, `npz`, or `pt`; `auto` prefers `.npz`
- `doc_shuffle`: Enables Slowrun-style document shuffling on the training split
- `eval_tokens`: Validation token budget for Slowrun-style evaluation

### Logging Configuration
- `enabled`: Enables or disables Weights & Biases logging
- `project`, `group`, `tags`, `notes`: WandB run metadata
- `save_code`: Uploads the current source tree to the WandB run
- `log_checkpoints`: Uploads checkpoint directories as WandB artifacts when enabled

## Attention Types

The system supports four attention mechanisms:

1. **MHA** (Multi-Head Attention) - Standard attention mechanism
2. **GQA** (Grouped-Query Attention) - Reduces memory usage with grouped key/value heads
3. **MHLA** (Multi-Head Latent Attention) - Latent attention with compression
4. **VoMHLA** - Value-rotated MHLA variant

To switch between attention types, simply change the `attention_type` field in your configuration file.
