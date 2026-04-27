"""Prepare the Slowrun FineWeb train/validation token files.

This writes the same file schema used by qlabs-eng/slowrun:
`tokens`, `doc_starts`, `bos_id`, `seq_shuffle_seed`, and `seq_size`.
"""

import argparse
import hashlib
import os

import numpy as np
import tiktoken
from datasets import load_dataset


SEQUENCE_LENGTH = 2048
VAL_SHUFFLE_SEED = 42
TRAIN_SHUFFLE_SEED = 43

EXPECTED_HASHES = {
    "fineweb_val.pt": "6868ed375b289a89c72c2f9df1ecbdcff700c4b9478ca806435d2dbfad8573b1",
    "fineweb_train.pt": "36e7c95c1e7f6ed952fb002d76a03044e8617fea7e696a68d7dc1ce78465dcaf",
}


def _require_optional_deps():
    try:
        import torch
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Slowrun data preparation requires optional dependencies. "
            "Install them with `uv sync --extra slowrun`."
        ) from exc
    return torch, tqdm


def tokenize_documents(dataset_iter, encoder, token_budget: int, tqdm):
    bos_id = encoder._special_tokens["<|endoftext|>"]
    tokens = []
    doc_starts = []
    progress = tqdm(total=token_budget, unit="tok")
    for document in dataset_iter:
        doc_tokens = [bos_id] + encoder.encode_ordinary(document["text"])
        doc_starts.append(len(tokens))
        keep = min(len(doc_tokens), token_budget - len(tokens))
        tokens.extend(doc_tokens[:keep])
        progress.update(keep)
        if len(tokens) >= token_budget:
            break
    progress.close()
    return np.asarray(tokens[:token_budget], dtype=np.uint16), np.asarray(doc_starts, dtype=np.int64)


def write_datafile(torch, filepath: str, tokens: np.ndarray, doc_starts: np.ndarray, bos_id: int, shuffle_seed: int):
    if tokens.size == 0:
        raise ValueError(f"refusing to write empty token stream to {filepath}")
    if doc_starts.size == 0 or doc_starts[0] != 0:
        raise ValueError("document starts must begin at 0")
    if not np.all(tokens[doc_starts] == bos_id):
        raise ValueError("document starts must point at BOS tokens")

    data = {
        "tokens": torch.from_numpy(tokens.copy()),
        "doc_starts": torch.from_numpy(doc_starts.copy()),
        "bos_id": int(bos_id),
        "seq_shuffle_seed": int(shuffle_seed),
        "seq_size": int(SEQUENCE_LENGTH + 1),
    }
    torch.save(data, filepath)
    print(f"Wrote {filepath}: {tokens.size:,} tokens, {doc_starts.size:,} docs")


def sha256_file(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(filepath: str):
    basename = os.path.basename(filepath)
    expected = EXPECTED_HASHES.get(basename)
    actual = sha256_file(filepath)
    if expected is None:
        print(f"{basename} sha256: {actual}")
    elif actual != expected:
        raise ValueError(f"{basename} hash mismatch: expected {expected}, got {actual}")
    else:
        print(f"{basename} hash OK: {actual}")


def preprocess(train_tokens: int, val_tokens: int, local_dir: str, verify: bool):
    torch, tqdm = _require_optional_deps()
    encoder = tiktoken.get_encoding("gpt2")
    bos_id = encoder._special_tokens["<|endoftext|>"]
    os.makedirs(local_dir, exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    dataset_iter = iter(dataset)

    print(f"Tokenizing Slowrun validation split: {val_tokens:,} tokens")
    val_tokens_array, val_doc_starts = tokenize_documents(dataset_iter, encoder, val_tokens, tqdm)

    print(f"Tokenizing Slowrun train split: {train_tokens:,} tokens")
    train_tokens_array, train_doc_starts = tokenize_documents(dataset_iter, encoder, train_tokens, tqdm)
    del dataset_iter, dataset

    val_path = os.path.join(local_dir, "fineweb_val.pt")
    train_path = os.path.join(local_dir, "fineweb_train.pt")
    write_datafile(torch, val_path, val_tokens_array, val_doc_starts, bos_id, VAL_SHUFFLE_SEED)
    write_datafile(torch, train_path, train_tokens_array, train_doc_starts, bos_id, TRAIN_SHUFFLE_SEED)

    if verify:
        verify_hash(val_path)
        verify_hash(train_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Slowrun FineWeb data")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--local-dir", type=str, default="fineweb_data")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    preprocess(
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
        local_dir=args.local_dir,
        verify=not args.no_verify,
    )
