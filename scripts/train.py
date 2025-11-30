#!/usr/bin/env python3
"""
Full training script with 3-phase curriculum.

From datas.md:
- Phase 1 (0-1B): TinyStories + Cosmopedia (foundation)
- Phase 2 (1B-2B): Math + Code ramp
- Phase 3 (2B-3B): Full mix

Usage:
    ./venv/bin/python scripts/train.py                    # Full 3B training
    ./venv/bin/python scripts/train.py --tinystories-only # TinyStories only (476M)
    ./venv/bin/python scripts/train.py --max-tokens 1e9   # Stop at 1B tokens
"""

import argparse
import json
import math
import os
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

from data.loader import CurriculumDataLoader, TinyStoriesLoader
from data.tokenizer import get_tokenizer
from eval.metrics import TrainingMetrics, InductionHeadProbe, compute_loss


# Use simpler model for now (proven to work in test)
class TransformerLM(nn.Module):
    """Transformer Language Model."""
    
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


def create_model(size: str = "150M", vocab_size: int = 50257) -> TransformerLM:
    """Create model by size string."""
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
        "200M": {"num_layers": 16, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}")
    
    config = configs[size]
    config["vocab_size"] = vocab_size
    return TransformerLM(**config)


def count_params(model) -> int:
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def train(
    model_size: str = "150M",
    batch_size: int = 8,
    seq_len: int = 512,
    learning_rate: float = 5e-4,
    min_lr: float = 1e-6,
    warmup_steps: int = 500,
    max_tokens: int = 3_000_000_000,  # 3B default
    log_interval: int = 100,
    eval_interval: int = 2000,
    save_interval: int = 10000,
    output_dir: str = "checkpoints",
    tinystories_only: bool = False,
    resume_from: str = None,
):
    """Main training loop with curriculum."""
    
    run_name = f"{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"Mini-Model Training: {model_size}")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Target: {max_tokens/1e9:.1f}B tokens")
    print(f"Mode: {'TinyStories only' if tinystories_only else '3-phase curriculum'}")
    
    # Setup
    mx.set_default_device(mx.gpu)
    
    # Tokenizer
    tokenizer = get_tokenizer()
    print(f"\nTokenizer: {tokenizer.vocab_size} vocab")
    
    # Model
    print(f"\nCreating {model_size} model...")
    model = create_model(model_size, tokenizer.vocab_size)
    mx.eval(model.parameters())
    
    num_params = count_params(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    
    # Data loader
    if tinystories_only:
        print(f"\nLoading TinyStories (batch_size={batch_size}, seq_len={seq_len})...")
        loader = TinyStoriesLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    else:
        print(f"\nInitializing curriculum loader...")
        loader = CurriculumDataLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    
    # Metrics
    metrics = TrainingMetrics()
    
    # Resume from checkpoint
    start_step = 0
    start_tokens = 0
    if resume_from:
        print(f"\nResuming from {resume_from}...")
        model.load_weights(resume_from)
        state_file = resume_from.replace(".safetensors", "_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
                start_step = state.get("step", 0)
                start_tokens = state.get("total_tokens", 0)
        print(f"  Resumed at step {start_step}, {start_tokens/1e6:.1f}M tokens")
    
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
    
    # Estimate max steps
    tokens_per_step = batch_size * seq_len
    max_steps = max_tokens // tokens_per_step
    
    # Save config
    config = {
        "model_size": model_size,
        "num_params": num_params,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_tokens": max_tokens,
        "tinystories_only": tinystories_only,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    start_time = time.perf_counter()
    step_times = []
    best_loss = float("inf")
    
    for step_num, batch in enumerate(loader.iterate_batches()):
        if step_num < start_step:
            continue
        
        # Convert to MLX
        batch_mx = mx.array(batch)
        
        # Learning rate schedule
        lr = get_lr(step_num, warmup_steps, max_steps, learning_rate, min_lr)
        optimizer.learning_rate = lr
        
        # Training step
        step_start = time.perf_counter()
        loss = step(batch_mx)
        mx.eval(loss)
        step_time = time.perf_counter() - step_start
        
        loss_val = loss.item()
        total_tokens = loader.total_tokens
        
        # Track metrics
        metrics.add_train_loss(loss_val, total_tokens, lr)
        step_times.append(step_time)
        
        # Logging
        if (step_num + 1) % log_interval == 0:
            avg_loss = metrics.get_recent_avg()
            avg_step_time = sum(step_times[-log_interval:]) / min(len(step_times), log_interval)
            tokens_per_sec = tokens_per_step / avg_step_time
            
            elapsed = time.perf_counter() - start_time
            progress = total_tokens / max_tokens * 100
            
            # Get phase info for curriculum
            if hasattr(loader, 'current_phase'):
                phase_info = f" | Phase: {loader.current_phase}"
            else:
                phase_info = ""
            
            print(
                f"Step {step_num+1:6d} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {math.exp(min(avg_loss, 10)):7.2f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"Progress: {progress:.1f}%{phase_info}"
            )
        
        # Evaluation
        if (step_num + 1) % eval_interval == 0:
            print("\n  Running evaluation...")
            
            # Health check
            health = metrics.check_health()
            for name, (passed, msg) in health.items():
                status = "✓" if passed else "✗"
                print(f"    {status} {name}: {msg}")
            
            # Early stopping check
            should_stop, reason = metrics.should_stop_early()
            if should_stop:
                print(f"\n  ⚠️ Early stopping: {reason}")
                break
            
            print()
        
        # Save checkpoint
        if (step_num + 1) % save_interval == 0:
            ckpt_name = f"step_{step_num+1}"
            ckpt_path = output_path / f"{ckpt_name}.safetensors"
            
            model.save_weights(str(ckpt_path))
            
            # Save state
            with open(output_path / f"{ckpt_name}_state.json", "w") as f:
                json.dump({
                    "step": step_num + 1,
                    "total_tokens": total_tokens,
                    "loss": metrics.get_recent_avg(),
                    "learning_rate": lr,
                }, f, indent=2)
            
            print(f"  → Saved checkpoint: {ckpt_path}")
            
            # Track best
            current_loss = metrics.get_recent_avg()
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = output_path / "best.safetensors"
                model.save_weights(str(best_path))
                print(f"  → New best model (loss: {best_loss:.4f})")
        
        # Check token limit
        if total_tokens >= max_tokens:
            print(f"\nReached token target: {max_tokens/1e9:.1f}B")
            break
    
    # Final save
    final_path = output_path / "final.safetensors"
    model.save_weights(str(final_path))
    
    # Final stats
    total_time = time.perf_counter() - start_time
    total_tokens = loader.total_tokens
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Avg tokens/sec: {total_tokens/total_time:,.0f}")
    print(f"  Final loss: {metrics.get_recent_avg():.4f}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Output: {output_path}")
    
    # Save final summary
    summary = {
        "total_time_hours": total_time / 3600,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_time,
        "final_loss": metrics.get_recent_avg(),
        "best_loss": best_loss,
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train mini-model with curriculum")
    
    # Model
    parser.add_argument("--size", type=str, default="150M", 
                       choices=["10M", "50M", "150M", "200M"])
    
    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max-tokens", type=float, default=3e9,
                       help="Max tokens (e.g., 1e9 for 1B)")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=10000)
    
    # Data
    parser.add_argument("--tinystories-only", action="store_true",
                       help="Train on TinyStories only (pipeline validation)")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    train(
        model_size=args.size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_tokens=int(args.max_tokens),
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        tinystories_only=args.tinystories_only,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
