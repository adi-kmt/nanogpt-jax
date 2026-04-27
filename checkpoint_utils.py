from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import equinox as eqx

from config import DataConfig, GPTConfig, TrainingConfig, WandbConfig
from data_utils import describe_data_artifacts


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def config_payload(
    model_config: GPTConfig,
    train_config: TrainingConfig,
    data_config: DataConfig,
    wandb_config: WandbConfig,
) -> dict[str, Any]:
    return {
        "model": model_config.model_dump(),
        "training": train_config.model_dump(),
        "data": data_config.model_dump(),
        "logging": wandb_config.model_dump(),
    }


def checkpoint_metadata(
    *,
    step: int,
    tokens_seen: int,
    metrics: dict[str, Any],
    model_config: GPTConfig,
    train_config: TrainingConfig,
    data_config: DataConfig,
    wandb_config: WandbConfig,
    data_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = config_payload(model_config, train_config, data_config, wandb_config)
    artifacts = data_artifacts or describe_data_artifacts(data_config)
    return {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "metrics": metrics,
        "config": payload,
        "config_sha256": stable_hash(payload),
        "git_sha": git_sha(),
        "data_artifacts": artifacts,
        "data_artifacts_sha256": stable_hash(artifacts),
    }


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model,
    opt_state=None,
    metadata: dict[str, Any],
) -> Path:
    path = Path(checkpoint_dir)
    tmp_path = path.with_name(f".{path.name}.tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    model_path = tmp_path / "model.eqx"
    eqx.tree_serialise_leaves(model_path, model)
    checkpoint_files = {
        "model.eqx": {
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }
    }
    if opt_state is not None:
        opt_state_path = tmp_path / "opt_state.eqx"
        eqx.tree_serialise_leaves(opt_state_path, opt_state)
        checkpoint_files["opt_state.eqx"] = {
            "size_bytes": opt_state_path.stat().st_size,
            "sha256": sha256_file(opt_state_path),
        }
    metadata = {**metadata, "checkpoint_files": checkpoint_files}
    with open(tmp_path / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True, default=str)
        file.write("\n")

    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp_path, path)
    return path


def load_model_checkpoint(checkpoint_path: str | Path, model):
    path = Path(checkpoint_path)
    model_path = path / "model.eqx" if path.is_dir() else path
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint model file not found: {model_path}")
    return eqx.tree_deserialise_leaves(model_path, model)


def load_opt_state_checkpoint(checkpoint_path: str | Path, opt_state):
    path = Path(checkpoint_path)
    opt_state_path = path / "opt_state.eqx" if path.is_dir() else path
    if not opt_state_path.exists():
        raise FileNotFoundError(f"Checkpoint optimizer file not found: {opt_state_path}")
    return eqx.tree_deserialise_leaves(opt_state_path, opt_state)


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    metadata_path = path / "metadata.json" if path.is_dir() else path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def prune_step_checkpoints(root_dir: str | Path, keep: int | None):
    if keep is None:
        return
    root = Path(root_dir)
    if not root.exists():
        return
    step_dirs = sorted(
        [
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith("step_")
        ],
        key=lambda path: path.name,
    )
    for path in step_dirs[:-keep]:
        shutil.rmtree(path)
