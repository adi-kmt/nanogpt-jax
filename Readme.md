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

`prepare_slowrun_data.py` writes JAX-friendly `.npz` files by default. Use `--format pt` or `--format both` only when you need compatibility with the original Slowrun PyTorch artifact format.

The Slowrun config uses WSD learning-rate decay, scheduled weight decay, and ordered optimizer groups: precomputed RoPE tables are frozen, no-decay parameters use AdamW, and matrix weights use Muon. Each group can set its own LR and weight-decay multiplier.

Training writes `best_eval` and `last` checkpoints under `training.checkpoint_dir`. Exact validation can be run later with:

```bash
uv run python scripts/eval_slowrun.py \
  --config config/model_config_slowrun.yaml \
  --checkpoint checkpoints/slowrun/best_eval \
  --full
```

The training loop logs train loss, validation loss, BPB, accuracy, throughput, LR, weight decay, gradient norm, tokens seen, and best eval summaries to Weights & Biases when `logging.enabled` is true.
