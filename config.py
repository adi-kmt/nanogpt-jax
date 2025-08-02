from typing import Literal

from pydantic import BaseModel

class GPTConfig(BaseModel):
    activation_type: Literal["relu", "gelu", "relu2", "swish", "swiglu", "swilu"]
    dropout_p: float

    d_model: int

    linear_d_hidden: int

    use_bias: bool
    use_qkNorm: bool
    tie_word_embeddings: bool = True
    use_rotary:bool

    n_heads: int
    d_head: int
    max_seq_len: int
    norm_eps: float

    n_layers: int

    vocab_size: int