from typing import Literal

from pydantic import BaseModel

class GPTConfig(BaseModel):
    activation_type: Literal["relu", "gelu", "relu2", "swish", "swiglu", "swilu"]
    dropout_p: float

    d_model: int

    linear_d_hidden: int

    rms_eps: float

    use_bias: bool
    use_qkNorm: bool
    tie_word_embeddings: bool = True
    use_rotary:bool