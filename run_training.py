#!/usr/bin/env python3
"""
Example demonstrating how to run training with different YAML configurations.
"""

import os
import sys

def run_training_with_config(config_name: str = "model_config.yaml"):
    """Run training with a specific configuration file."""
    config_path = f"config/{config_name}"
    
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found!")
        return False
    
    # Set the config path as an environment variable that train.py can read
    os.environ["TRAIN_CONFIG_PATH"] = config_path
    
    # Import and run the training script
    try:
        # Add current directory to Python path
        sys.path.insert(0, '.')
        
        # Run training
        from train import load_config_from_yaml, train_distributed_safe
        model_config, train_config = load_config_from_yaml(config_path)
        print(f"Starting training with configuration: {config_name}")
        print(f"Model attention type: {model_config.attention_type}")
        train_distributed_safe(model_config, train_config)
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("Training Configuration Examples:")
    print("1. Standard MHA model: python run_training.py model_config.yaml")
    print("2. MHLA model: python run_training.py model_config_mhla.yaml")
    print("3. Small test model: python run_training.py model_config_small.yaml")
    
    # If a config name is provided as command line argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        run_training_with_config(config_name)
    else:
        # Default to standard config
        run_training_with_config("model_config.yaml")