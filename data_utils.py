import jax
import jax.numpy as jnp
import numpy as np
import requests
import tiktoken


def create_dataloader(seq_len: int, batch_size: int, split: str = "train"):
    """Create dataloader for TinyShakespeare dataset without EOT insertion."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text.strip()  # Remove extra whitespace

    train_ratio = 0.8
    split_idx = int(len(text) * train_ratio)
    text = text[:split_idx] if split == "train" else text[split_idx:]

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)

    # Keep as one continuous stream — no EOT insertion
    tokens = ids

    # Reshape into sequences of length (seq_len + 1)
    total_len = len(tokens)
    num_sequences = total_len // (seq_len + 1)
    sequences = np.array(tokens[:num_sequences * (seq_len + 1)]).reshape(num_sequences, seq_len + 1)

    def batch_generator():
        indices = np.arange(num_sequences)
        while True:
            np.random.shuffle(indices)
            for i in range(0, num_sequences, batch_size):
                if i + batch_size > num_sequences:
                    continue
                batch = sequences[indices[i:i + batch_size]]
                inputs = jnp.array(batch[:, :-1], dtype=jnp.int32)  # [B, T]
                targets = jnp.array(batch[:, 1:], dtype=jnp.int32)   # [B, T]
                yield inputs, targets

    return batch_generator()


def setup_sharding():
    """Setup data sharding for distributed training."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, axis_names=("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    return data_sharding