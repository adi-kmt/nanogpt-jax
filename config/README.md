# Configuration System

This project uses YAML files for configuration instead of hardcoded values. This makes it easy to experiment with different model architectures and training settings.

## Configuration Files

The configuration files are located in the `config/` directory:

1. `model_config.yaml` - Standard model configuration with MHA attention
2. `model_config_mhla.yaml` - Model configuration with MHLA attention
3. `model_config_small.yaml` - Small model configuration for smoke tests
4. `model_config_slowrun.yaml` - FineWeb/Slowrun competition data configuration

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

### Training Configuration
- `batch_size`: Effective training batch size; must equal `micro_batch_size * grad_accum_steps`
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `optimizer`: Optimizer type ("adamw", etc.)
- `log_every`: Number of optimizer steps between train logs
- `eval_every`: Number of optimizer steps between validation runs
- `eval_steps`: Fixed number of eval batches; set to `null` to use `data.eval_tokens` or the full eval loader

### Data Configuration
- `dataset`: `tinyshakespeare` or `slowrun`
- `data_dir`: Directory containing Slowrun `fineweb_train.pt` and `fineweb_val.pt`
- `doc_shuffle`: Enables Slowrun-style document shuffling on the training split
- `eval_tokens`: Validation token budget for Slowrun-style evaluation

### Logging Configuration
- `enabled`: Enables or disables Weights & Biases logging
- `project`, `group`, `tags`, `notes`: WandB run metadata
- `save_code`: Uploads the current source tree to the WandB run

## Attention Types

The system supports four attention mechanisms:

1. **MHA** (Multi-Head Attention) - Standard attention mechanism
2. **GQA** (Grouped-Query Attention) - Reduces memory usage with grouped key/value heads
3. **MHLA** (Multi-Head Latent Attention) - Latent attention with compression
4. **VoMHLA** - Value-rotated MHLA variant

To switch between attention types, simply change the `attention_type` field in your configuration file.
