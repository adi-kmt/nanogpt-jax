# Configuration System

This project uses YAML files for configuration instead of hardcoded values. This makes it easy to experiment with different model architectures and training settings.

## Configuration Files

The configuration files are located in the `config/` directory:

1. `model_config.yaml` - Standard model configuration with MHA attention
2. `model_config_mhla.yaml` - Model configuration with MHLA attention
3. `model_config_small.yaml` - Small model configuration for testing

## Running Training with Different Configurations

To run training with a specific configuration, set the `TRAIN_CONFIG_PATH` environment variable:

```bash
# Run with standard MHA configuration
TRAIN_CONFIG_PATH=config/model_config.yaml python train.py

# Run with MHLA configuration
TRAIN_CONFIG_PATH=config/model_config_mhla.yaml python train.py

# Run with small model for testing
TRAIN_CONFIG_PATH=config/model_config_small.yaml python train.py
```

## Creating Custom Configurations

You can create your own configuration files by copying and modifying existing ones:

```bash
cp config/model_config.yaml config/my_custom_config.yaml
# Edit config/my_custom_config.yaml with your desired settings
TRAIN_CONFIG_PATH=config/my_custom_config.yaml python train.py
```

## Configuration Structure

Each YAML file has two main sections:

### Model Configuration
- `activation_type`: Activation function ("gelu", "relu", etc.)
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `attention_type`: Attention mechanism ("mha", "gqa", "mhla")
- `mhla_config`: MHLA-specific parameters (only needed for MHLA attention)

### Training Configuration
- `batch_size`: Training batch size
- `lr`: Learning rate
- `epochs`: Number of training epochs
- `optimizer`: Optimizer type ("adamw", etc.)

## Attention Types

The system supports three attention mechanisms:

1. **MHA** (Multi-Head Attention) - Standard attention mechanism
2. **GQA** (Grouped-Query Attention) - Reduces memory usage with grouped key/value heads
3. **MHLA** (Multi-Head Latent Attention) - Latent attention with compression

To switch between attention types, simply change the `attention_type` field in your configuration file.