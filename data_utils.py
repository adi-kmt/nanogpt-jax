import os
import hashlib
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import requests
import tiktoken

from config import DataConfig


def _as_jax_batch(batch: np.ndarray):
    batch = batch.astype(np.int32, copy=False)
    return jnp.asarray(batch[:, :-1]), jnp.asarray(batch[:, 1:])


class TinyShakespeareDataLoader:
    def __init__(self, seq_len: int, batch_size: int, split: str, config: DataConfig):
        response = requests.get(config.tinyshakespeare_url, timeout=30)
        response.raise_for_status()

        text = response.text.strip()
        split_idx = int(len(text) * config.train_split)
        text = text[:split_idx] if split == "train" else text[split_idx:]

        encoder = tiktoken.get_encoding(config.tokenizer)
        tokens = np.asarray(encoder.encode_ordinary(text), dtype=np.int64)
        seq_size = seq_len + 1
        num_sequences = len(tokens) // seq_size
        if num_sequences < batch_size:
            raise ValueError(
                f"{split} split has only {num_sequences} sequences, less than batch_size={batch_size}"
            )

        self.sequences = tokens[:num_sequences * seq_size].reshape(num_sequences, seq_size)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.seed = config.seed + (0 if split == "train" else 100_000)
        self.epoch = 1
        self.pos = 0
        self.num_steps = num_sequences // batch_size
        self.total_tokens = self.num_steps * batch_size * seq_len
        self._shuffle_epoch()

    def _shuffle_epoch(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.indices = rng.permutation(len(self.sequences))

    def __iter__(self):
        return self

    def reset(self):
        self.epoch = 1
        self.pos = 0
        self._shuffle_epoch()

    def __next__(self):
        if self.pos >= self.num_steps:
            self.epoch += 1
            self.pos = 0
            self._shuffle_epoch()

        start = self.pos * self.batch_size
        end = start + self.batch_size
        self.pos += 1
        return _as_jax_batch(self.sequences[self.indices[start:end]])


class SlowRunDataLoader:
    """Numpy/JAX loader for Slowrun FineWeb `.npz` or `.pt` files.

    The expected file format is produced by `prepare_slowrun_data.py` and mirrors
    qlabs-eng/slowrun: flat GPT-2 token ids plus document starts, BOS id, sequence
    size, and stored sequence shuffle seed.
    """

    def __init__(self, filepath: str, batch_size: int, seq_len: int, doc_shuffle: bool = False):
        self.filepath = filepath
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_size = seq_len + 1
        self.doc_shuffle = doc_shuffle
        self.epoch = 1

        data = self._load_file(filepath)
        tokens = np.asarray(_to_numpy(data["tokens"]), dtype=np.int64)
        doc_starts = np.asarray(_to_numpy(data["doc_starts"]), dtype=np.int64)
        self.bos_id = int(data["bos_id"])
        self.default_shuffle_seed = int(data["seq_shuffle_seed"])
        file_seq_size = int(data.get("seq_size", self.seq_size))
        if file_seq_size != self.seq_size:
            raise ValueError(
                f"{filepath} was prepared with seq_size={file_seq_size}, "
                f"but this run needs seq_size={self.seq_size}"
            )
        if doc_starts.size == 0 or doc_starts[0] != 0:
            raise ValueError(f"{filepath} has invalid document starts")
        if not np.all(tokens[doc_starts] == self.bos_id):
            raise ValueError(f"{filepath} document starts do not point at BOS tokens")

        doc_ends = np.concatenate([doc_starts[1:], np.asarray([tokens.size], dtype=np.int64)])
        self.doc_tokens = [tokens[start:end] for start, end in zip(doc_starts, doc_ends)]
        self._build_batches()

    @staticmethod
    def _load_file(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Slowrun data file not found: {filepath}. "
                "Run `python prepare_slowrun_data.py` first."
            )
        if filepath.endswith(".npz"):
            npz = np.load(filepath)
            return {key: npz[key] for key in npz.files}
        if not filepath.endswith(".pt"):
            raise ValueError(f"Unsupported Slowrun data file format: {filepath}")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Slowrun `.pt` data loading requires torch. Install the optional "
                "Slowrun dependencies, e.g. `uv sync --extra slowrun`."
            ) from exc
        return torch.load(filepath, map_location="cpu", weights_only=True)

    def _build_batches(self):
        tokens = np.concatenate(self.doc_tokens)
        num_sequences = tokens.size // self.seq_size
        if num_sequences < self.batch_size:
            raise ValueError(
                f"{self.filepath} has only {num_sequences} sequences, less than batch_size={self.batch_size}"
            )

        all_sequences = tokens[:num_sequences * self.seq_size].reshape(num_sequences, self.seq_size)
        if self.doc_shuffle:
            rng = np.random.RandomState(self.epoch + 1000)
            all_sequences = all_sequences[rng.permutation(num_sequences)]
        else:
            rng = np.random.RandomState(self.default_shuffle_seed)
            all_sequences = all_sequences[rng.permutation(num_sequences)]

        self.num_steps = num_sequences // self.batch_size
        usable = self.num_steps * self.batch_size
        self.batches = all_sequences[:usable].reshape(self.num_steps, self.batch_size, self.seq_size)
        self.total_tokens = usable * self.seq_len
        self.pos = 0

    def __iter__(self):
        return self

    def reset(self):
        self.epoch = 1
        self._build_batches()

    def _next_epoch(self):
        self.epoch += 1
        if self.doc_shuffle:
            rng = np.random.RandomState(self.epoch)
            doc_perm = rng.permutation(len(self.doc_tokens))
            self.doc_tokens = [self.doc_tokens[i] for i in doc_perm]
            self._build_batches()
        else:
            self.pos = 0
            rng = np.random.RandomState(self.epoch)
            self.batches = self.batches[rng.permutation(self.num_steps)]

    def __next__(self):
        if self.pos >= self.num_steps:
            self._next_epoch()

        batch = self.batches[self.pos]
        self.pos += 1
        return _as_jax_batch(batch)


def resolve_slowrun_path(config: DataConfig, split: str) -> str:
    explicit = config.train_path if split == "train" else config.val_path
    if explicit:
        return explicit

    stem = "fineweb_train" if split == "train" else "fineweb_val"
    if config.data_format == "auto":
        candidates = [
            os.path.join(config.data_dir, f"{stem}.npz"),
            os.path.join(config.data_dir, f"{stem}.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]
    return os.path.join(config.data_dir, f"{stem}.{config.data_format}")


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return value


def sha256_file(filepath: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_data_artifacts(data_config: DataConfig, splits: tuple[str, ...] = ("train", "val")) -> dict:
    if data_config.dataset != "slowrun":
        return {
            "dataset": data_config.dataset,
            "tokenizer": data_config.tokenizer,
            "tinyshakespeare_url": data_config.tinyshakespeare_url,
        }

    artifacts = {
        "dataset": data_config.dataset,
        "tokenizer": data_config.tokenizer,
        "doc_shuffle": data_config.doc_shuffle,
        "files": {},
    }
    for split in splits:
        path = resolve_slowrun_path(data_config, split)
        file_info = {
            "path": path,
            "exists": os.path.exists(path),
        }
        if os.path.exists(path):
            file_info["size_bytes"] = os.path.getsize(path)
            file_info["sha256"] = sha256_file(path)
        artifacts["files"][split] = file_info
    return artifacts


def create_dataloader(
    seq_len: int,
    batch_size: int,
    split: str = "train",
    data_config: Optional[DataConfig] = None,
):
    config = data_config or DataConfig()
    if config.dataset == "tinyshakespeare":
        return TinyShakespeareDataLoader(seq_len, batch_size, split, config)
    if config.dataset == "slowrun":
        return SlowRunDataLoader(
            filepath=resolve_slowrun_path(config, split),
            batch_size=batch_size,
            seq_len=seq_len,
            doc_shuffle=(config.doc_shuffle and split == "train"),
        )
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def gpt2_token_bytes(vocab_size: int) -> jax.Array:
    encoder = tiktoken.get_encoding("gpt2")
    if vocab_size > encoder.n_vocab:
        raise ValueError(f"Cannot compute GPT-2 token bytes for vocab_size={vocab_size}")

    eot_id = encoder._special_tokens["<|endoftext|>"]
    token_bytes = []
    for token_id in range(vocab_size):
        if token_id == eot_id:
            token_bytes.append(0)
        else:
            token_bytes.append(len(encoder.decode_single_token_bytes(token_id)))
    return jnp.asarray(token_bytes, dtype=jnp.int32)


def setup_sharding():
    """Setup data sharding for distributed training."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, axis_names=("data",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
