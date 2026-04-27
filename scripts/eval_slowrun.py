from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import equinox as eqx
import jax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint_utils import load_model_checkpoint, sha256_file, stable_hash
from data_utils import create_dataloader, describe_data_artifacts, gpt2_token_bytes, setup_sharding
from train import create_safer_model, evaluate_model, load_config_from_yaml, resolve_eval_steps


def main():
    parser = argparse.ArgumentParser(description="Run an exact Slowrun validation pass")
    parser.add_argument("--config", default=os.environ.get("TRAIN_CONFIG_PATH", "config/model_config_slowrun.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or model.eqx file")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval steps")
    parser.add_argument("--full", action="store_true", help="Evaluate every validation batch")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    model_config, train_config, data_config, wandb_config = load_config_from_yaml(args.config)
    if data_config.dataset != "slowrun":
        raise SystemExit("eval_slowrun.py expects data.dataset='slowrun'")

    key_model, key_eval = jax.random.split(jax.random.PRNGKey(42))
    model = create_safer_model(model_config, key_model)
    model = load_model_checkpoint(args.checkpoint, model)
    model = eqx.nn.inference_mode(model, value=True)

    eval_loader = create_dataloader(
        seq_len=model_config.max_seq_len,
        batch_size=train_config.eval_batch_size,
        split="val",
        data_config=data_config,
    )
    if args.full:
        eval_steps = getattr(eval_loader, "num_steps")
    elif args.eval_steps is not None:
        eval_steps = min(args.eval_steps, getattr(eval_loader, "num_steps", args.eval_steps))
    else:
        eval_steps = resolve_eval_steps(train_config, data_config, eval_loader, model_config.max_seq_len)

    token_bytes = gpt2_token_bytes(model_config.vocab_size)
    metrics, _ = evaluate_model(
        model,
        eval_loader,
        eval_steps,
        setup_sharding(),
        key_eval,
        token_bytes,
    )

    artifacts = describe_data_artifacts(data_config, splits=("val",))
    checkpoint_path = Path(args.checkpoint)
    model_checkpoint_path = checkpoint_path / "model.eqx" if checkpoint_path.is_dir() else checkpoint_path
    result = {
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "checkpoint_model_sha256": sha256_file(model_checkpoint_path),
        "config": args.config,
        "config_sha256": stable_hash({
            "model": model_config.model_dump(),
            "training": train_config.model_dump(),
            "data": data_config.model_dump(),
            "logging": wandb_config.model_dump(),
        }),
        "data_artifacts": artifacts,
        "data_artifacts_sha256": stable_hash(artifacts),
    }
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(text)
            file.write("\n")


if __name__ == "__main__":
    main()
