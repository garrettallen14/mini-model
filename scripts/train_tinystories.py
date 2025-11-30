#!/usr/bin/env python3
"""
Phase 2 Validation: Train on TinyStories only.

This is the "validate pipeline first" path from datas.md:
- 476M tokens (~8 hours on M4)
- 150M model 
- Validates full training loop before 3B token run

Usage:
    ./venv/bin/python scripts/train_tinystories.py
    ./venv/bin/python scripts/train_tinystories.py --small  # 10M model, quick test
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from data.loader import TinyStoriesLoader
from data.tokenizer import get_tokenizer


class TransformerLM(nn.Module):
    """Transformer Language Model with configurable size."""
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        
        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def count_params(model) -> int:
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def create_model(size: str = "150M") -> TransformerLM:
    """Create model based on size string."""
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    config = configs[size]
    config["vocab_size"] = 50257  # GPT-2
    
    return TransformerLM(**config)


def train(
    model_size: str = "150M",
    batch_size: int = 8,
    seq_len: int = 512,
    learning_rate: float = 5e-4,
    warmup_steps: int = 500,
    log_interval: int = 100,
    eval_interval: int = 1000,
    save_interval: int = 5000,
    max_tokens: int = None,  # None = full dataset
    output_dir: str = "checkpoints",
):
    """Main training loop."""
    
    print("=" * 60)
    print(f"TinyStories Training - {model_size} Model")
    print("=" * 60)
    
    # Setup
    mx.set_default_device(mx.gpu)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Tokenizer
    tokenizer = get_tokenizer()
    print(f"\nTokenizer: {tokenizer.vocab_size} vocab")
    
    # Model
    print(f"\nCreating {model_size} model...")
    model = create_model(model_size)
    mx.eval(model.parameters())
    
    num_params = count_params(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Optimizer with cosine schedule
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    # Data loader
    print(f"\nLoading TinyStories (batch_size={batch_size}, seq_len={seq_len})...")
    loader = TinyStoriesLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    
    # Loss function
    def loss_fn(model, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    
    # Compiled step
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch):
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        return loss
    
    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    start_time = time.perf_counter()
    step_times = []
    losses = []
    best_loss = float("inf")
    
    for step_num, batch in enumerate(loader.iterate_batches()):
        # Convert to MLX
        batch_mx = mx.array(batch)
        
        # Learning rate schedule (linear warmup + cosine decay)
        if step_num < warmup_steps:
            lr = learning_rate * (step_num + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step_num - warmup_steps) / max(1, 50000 - warmup_steps)
            lr = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
            lr = max(lr, 1e-6)
        optimizer.learning_rate = lr
        
        # Training step
        step_start = time.perf_counter()
        loss = step(batch_mx)
        mx.eval(loss)
        step_time = time.perf_counter() - step_start
        
        loss_val = loss.item()
        losses.append(loss_val)
        step_times.append(step_time)
        
        # Logging
        if (step_num + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            avg_step_time = sum(step_times[-log_interval:]) / log_interval
            tokens_per_sec = batch_size * seq_len / avg_step_time
            
            elapsed = time.perf_counter() - start_time
            total_tokens = loader.total_tokens
            
            print(
                f"Step {step_num+1:6d} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {math.exp(avg_loss):7.2f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"Total: {total_tokens/1e6:.1f}M"
            )
        
        # Save checkpoint
        if (step_num + 1) % save_interval == 0:
            ckpt_path = output_path / f"step_{step_num+1}.safetensors"
            avg_loss = sum(losses[-save_interval:]) / save_interval
            
            # Save model weights
            model.save_weights(str(ckpt_path))
            
            # Save training state
            state_path = output_path / f"step_{step_num+1}_state.json"
            with open(state_path, "w") as f:
                json.dump({
                    "step": step_num + 1,
                    "total_tokens": loader.total_tokens,
                    "loss": avg_loss,
                    "learning_rate": lr,
                    "model_size": model_size,
                }, f, indent=2)
            
            print(f"  → Saved checkpoint to {ckpt_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_path / "best.safetensors"
                model.save_weights(str(best_path))
                print(f"  → New best model (loss: {best_loss:.4f})")
        
        # Check token limit
        if max_tokens and loader.total_tokens >= max_tokens:
            print(f"\nReached token limit: {max_tokens/1e6:.1f}M")
            break
    
    # Final stats
    total_time = time.perf_counter() - start_time
    total_tokens = loader.total_tokens
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    print(f"  Tokens/sec: {total_tokens/total_time:,.0f}")
    print(f"  Final loss: {sum(losses[-100:])/100:.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train on TinyStories")
    parser.add_argument("--small", action="store_true", help="Use 10M model for quick test")
    parser.add_argument("--medium", action="store_true", help="Use 50M model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-tokens", type=int, default=None, 
                       help="Stop after this many tokens (e.g., 100000000 for 100M)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    if args.small:
        model_size = "10M"
    elif args.medium:
        model_size = "50M"
    else:
        model_size = "150M"
    
    train(
        model_size=model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
