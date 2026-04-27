# NanoGPT (but in Equinox)

> If it quacks like your code, waddles like your code, and uses your exact puns in the comments… yeah, it’s yours. Thanks for the duck!

## Some Libraries
- [Equinox](https://docs.kidger.site/equinox)
- [JAX](https://jax.readthedocs.io/en/latest/index.html)
- [Optax](https://optax.readthedocs.io/en/latest/index.html)
- [JAXtyping](https://github.com/patrick-kidger/jaxtyping)
- [Orbax-Checkpoints](https://github.com/google/orbax)

## Slowrun FineWeb Data

The Slowrun/FineWeb path is optional and selected through config:

```bash
uv sync --extra slowrun
uv run python prepare_slowrun_data.py
TRAIN_CONFIG_PATH=config/model_config_slowrun.yaml uv run python train.py
```

The training loop logs train loss, validation loss, BPB, accuracy, throughput, LR, gradient norm, tokens seen, and best eval summaries to Weights & Biases when `logging.enabled` is true.
