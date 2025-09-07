import jax
from jaxtyping import Array, Float, Bool
from config import GPTConfig
from attentions import MultiHeadAttention, GroupQueryAttention, MHLA, VoMHLA


def create_attention(config: GPTConfig, key: jax.random.PRNGKey):
    """
    Factory function to create attention mechanisms based on config.
    
    Args:
        config: GPTConfig with attention_type field
        key: PRNGKey for initialization
        
    Returns:
        Appropriate attention module based on config.attention_type
    """
    if config.attention_type == "mha":
        return MultiHeadAttention(config, key=key)
    elif config.attention_type == "gqa":
        return GroupQueryAttention(config, key=key)
    elif config.attention_type == "mhla":
        # MHLA requires additional mhla_config
        if config.mhla_config is None:
            raise ValueError("MHLA attention type requires mhla_config to be set in GPTConfig")
        return MHLA(config, config.mhla_config, key=key)
    elif config.attention_type == "vo-mhla":
        # VoMHLA requires additional mhla_config
        if config.mhla_config is None:
            raise ValueError("MHLA attention type requires mhla_config to be set in GPTConfig")
        return VoMHLA(config, config.mhla_config, key=key)
    else:
        raise ValueError(f"Unknown attention type: {config.attention_type}")


def get_attention_output_shape(config: GPTConfig) -> int:
    """
    Get the output dimension for the attention mechanism.
    
    Args:
        config: GPTConfig
        
    Returns:
        Output dimension (d_model)
    """
    return config.d_model


def get_attention_input_shape(config: GPTConfig) -> int:
    """
    Get the input dimension for the attention mechanism.
    
    Args:
        config: GPTConfig
        
    Returns:
        Input dimension (d_model)
    """
    return config.d_model