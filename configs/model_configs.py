"""
Model configurations for mini-model training.
Based on outline.md specifications.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Base model configuration."""
    vocab_size: int = 50257  # GPT-2 BPE tokenizer
    num_layers: int = 12
    dims: int = 768
    num_heads: int = 12
    ff_mult: float = 4.0  # Standard is 4x, SwiGLU uses 3.5x
    max_seq_len: int = 512
    use_rope: bool = True
    use_checkpoint: bool = True
    dropout: float = 0.0  # No dropout for small models per research
    
    @property
    def num_params_millions(self) -> float:
        """Estimate total parameters in millions."""
        # Embeddings
        embed_params = self.vocab_size * self.dims
        
        # Per transformer layer:
        # - Attention: 4 * dims^2 (Q, K, V, O projections)
        # - FFN: 2 * dims * (dims * ff_mult) for standard, 3x for SwiGLU
        # - Layer norms: 2 * dims
        attn_params = 4 * self.dims * self.dims
        ffn_dim = int(self.dims * self.ff_mult)
        ffn_params = 2 * self.dims * ffn_dim  # Simplified
        ln_params = 4 * self.dims  # 2 layer norms, 2 params each
        layer_params = attn_params + ffn_params + ln_params
        total_layer_params = self.num_layers * layer_params
        
        # Output projection (shared with embedding usually)
        out_params = 0  # Assume weight tying
        
        total = embed_params + total_layer_params + out_params
        return total / 1_000_000


# Predefined configurations
CONFIGS = {
    # Tiny config for pipeline validation
    "10M": ModelConfig(
        vocab_size=50257,
        num_layers=4,
        dims=256,
        num_heads=4,
        ff_mult=4.0,
        max_seq_len=256,
        use_checkpoint=False,
    ),
    
    # Small config for quick experiments
    "50M": ModelConfig(
        vocab_size=50257,
        num_layers=8,
        dims=512,
        num_heads=8,
        ff_mult=4.0,
        max_seq_len=512,
        use_checkpoint=False,
    ),
    
    # Target config from outline.md
    "150M": ModelConfig(
        vocab_size=50257,
        num_layers=12,
        dims=768,
        num_heads=12,
        ff_mult=3.5,  # SwiGLU efficiency
        max_seq_len=512,
        use_checkpoint=True,
    ),
    
    # Stretch goal
    "200M": ModelConfig(
        vocab_size=50257,
        num_layers=16,
        dims=768,
        num_heads=12,
        ff_mult=3.5,
        max_seq_len=512,
        use_checkpoint=True,
    ),
}


@dataclass 
class TrainingConfig:
    """Training hyperparameters from outline.md."""
    # Optimizer
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Schedule
    warmup_steps: int = 500
    total_steps: int = 50000
    min_lr: float = 1e-6
    
    # Batch
    micro_batch_size: int = 8
    grad_accum_steps: int = 4  # Effective BS = 32
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


# Training configs for different phases
TRAINING_CONFIGS = {
    "baseline_test": TrainingConfig(
        learning_rate=3e-4,
        warmup_steps=100,
        total_steps=1000,
        micro_batch_size=4,
        grad_accum_steps=1,
        log_interval=10,
        eval_interval=200,
        save_interval=500,
    ),
    
    "full_train": TrainingConfig(
        learning_rate=5e-4,
        warmup_steps=500,
        total_steps=50000,
        micro_batch_size=8,
        grad_accum_steps=4,
        log_interval=10,
        eval_interval=500,
        save_interval=2500,
    ),
}


def get_config(model_size: str) -> ModelConfig:
    """Get model config by size string."""
    if model_size not in CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(CONFIGS.keys())}")
    return CONFIGS[model_size]


def get_training_config(name: str) -> TrainingConfig:
    """Get training config by name."""
    if name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {name}. Choose from {list(TRAINING_CONFIGS.keys())}")
    return TRAINING_CONFIGS[name]


if __name__ == "__main__":
    # Print all configs
    print("Model Configurations:")
    print("-" * 50)
    for name, config in CONFIGS.items():
        print(f"{name}: ~{config.num_params_millions:.1f}M params")
        print(f"  layers={config.num_layers}, dims={config.dims}, heads={config.num_heads}")
        print(f"  seq_len={config.max_seq_len}, checkpoint={config.use_checkpoint}")
        print()
