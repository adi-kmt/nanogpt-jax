from typing import Literal, Optional

from pydantic import BaseModel


class GPTConfig(BaseModel):
    activation_type: Literal["relu", "gelu", "relu2", "swish", "swiglu", "swilu"]
    dropout_p: float

    d_model: int

    linear_d_hidden: int

    use_bias: bool
    use_qkNorm: bool
    tie_word_embeddings: bool = True
    use_rotary: bool

    n_heads: int
    d_head: int
    n_kv_heads: Optional[int] = None
    max_seq_len: int
    norm_eps: float

    n_layers: int

    vocab_size: int
    
    # Attention type configuration
    attention_type: Literal["mha", "gqa", "mhla"] = "mha"
    mhla_config: Optional["GPTConfig.MhlaConfig"] = None

    class MhlaConfig(BaseModel):
        d_c: int
        d_c1: int
        d_r: int


class TrainingConfig(BaseModel):
    batch_size: int
    micro_batch_size: int
    eval_batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    optimizer: Literal["adam", "adamw", "muon"]
    scheduler: Literal["cosine", "linear"] | None
    grad_accum_steps: int