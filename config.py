from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator


ActivationType = Literal["relu", "gelu", "relu2", "silu", "swish", "identity", "swiglu"]


class GPTConfig(BaseModel):
    activation_type: ActivationType
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
    attention_type: Literal["mha", "gqa", "mhla", "vo-mhla"] = "mha"
    mhla_config: Optional["GPTConfig.MhlaConfig"] = None

    class MhlaConfig(BaseModel):
        d_c: int
        d_c1: int
        d_r: int

        @model_validator(mode="after")
        def validate_mhla_dims(self):
            if self.d_c <= 0 or self.d_c1 <= 0 or self.d_r <= 0:
                raise ValueError("MHLA dimensions must be positive")
            if self.d_r % 2 != 0:
                raise ValueError("mhla_config.d_r must be even for rotary embeddings")
            return self

    @field_validator("activation_type", mode="before")
    @classmethod
    def normalize_activation_type(cls, value):
        if value == "swilu":
            return "silu"
        return value

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.d_model <= 0 or self.linear_d_hidden <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.n_heads <= 0 or self.d_head <= 0:
            raise ValueError("n_heads and d_head must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.max_seq_len <= 0 or self.vocab_size <= 0:
            raise ValueError("max_seq_len and vocab_size must be positive")
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if self.d_head % 2 != 0:
            raise ValueError("d_head must be even for rotary embeddings")

        if self.n_kv_heads is not None:
            if self.n_kv_heads <= 0:
                raise ValueError("n_kv_heads must be positive when provided")
            if self.n_heads % self.n_kv_heads != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads")

        if self.attention_type == "gqa" and self.n_kv_heads is None:
            raise ValueError("GQA attention requires n_kv_heads")

        if self.attention_type in {"mhla", "vo-mhla"} and self.mhla_config is None:
            raise ValueError(f"{self.attention_type} attention requires mhla_config")

        return self


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
    log_every: int = 10
    eval_every: int | None = 500
    eval_steps: int | None = 50
    eval_on_start: bool = True
    eval_on_end: bool = True

    @model_validator(mode="after")
    def validate_training_config(self):
        if self.batch_size <= 0 or self.micro_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.batch_size != self.micro_batch_size * self.grad_accum_steps:
            raise ValueError("batch_size must equal micro_batch_size * grad_accum_steps")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.eval_every is not None and self.eval_every <= 0:
            raise ValueError("eval_every must be positive when provided")
        if self.eval_steps is not None and self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive when provided")
        return self


class DataConfig(BaseModel):
    dataset: Literal["tinyshakespeare", "slowrun"] = "tinyshakespeare"
    data_dir: str = "fineweb_data"
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    tokenizer: Literal["gpt2"] = "gpt2"
    tinyshakespeare_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    train_split: float = 0.8
    seed: int = 42
    doc_shuffle: bool = False
    eval_tokens: Optional[int] = None

    @model_validator(mode="after")
    def validate_data_config(self):
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if self.eval_tokens is not None and self.eval_tokens <= 0:
            raise ValueError("eval_tokens must be positive when provided")
        return self


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str = "nanogpt-equinox"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    mode: Literal["online", "offline", "disabled"] | None = None
    tags: list[str] = []
    notes: Optional[str] = None
    resume: Literal["allow", "must", "never", "auto"] | None = "allow"
    id: Optional[str] = None
    save_code: bool = True
    define_metrics: bool = True
