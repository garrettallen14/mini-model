#!/usr/bin/env python3
"""
Find optimal training configuration with the fast data loader.

Tests different batch sizes and sequence lengths to maximize throughput.
"""

import gc
import sys
import time
from functools import partial
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, dims, num_heads, checkpoint=False):
        super().__init__()
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


def count_params(model):
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def test_config(
    model_size: str,
    batch_size: int, 
    seq_len: int,
    num_steps: int = 20,
    warmup_steps: int = 5,
):
    """Test a single configuration."""
    
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    
    config = configs[model_size]
    
    try:
        # Create model
        model = TransformerLM(vocab_size=50257, **config)
        mx.eval(model.parameters())
        
        optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=0.01)
        
        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")
        
        state = [model.state, optimizer.state]
        
        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, y):
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            return loss
        
        # Create dummy data
        dummy_batch = mx.random.randint(0, 50257, (batch_size, seq_len + 1))
        x, y = dummy_batch[:, :-1], dummy_batch[:, 1:]
        
        # Warmup
        for _ in range(warmup_steps):
            loss = step(x, y)
            mx.eval(loss)
        
        # Profile
        step_times = []
        start = time.perf_counter()
        
        for _ in range(num_steps):
            loss = step(x, y)
            mx.eval(loss)
            
            step_time = time.perf_counter() - start
            step_times.append(step_time * 1000)
            start = time.perf_counter()
        
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_step = batch_size * seq_len
        tokens_per_sec = tokens_per_step / (avg_step_time / 1000)
        
        # Cleanup
        del model, optimizer
        gc.collect()
        
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "step_time_ms": avg_step_time,
            "tokens_per_sec": tokens_per_sec,
            "tokens_per_step": tokens_per_step,
            "error": None,
        }
        
    except Exception as e:
        gc.collect()
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "step_time_ms": 0,
            "tokens_per_sec": 0,
            "tokens_per_step": 0,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("OPTIMAL CONFIGURATION FINDER")
    print("=" * 70)
    
    mx.set_default_device(mx.gpu)
    
    model_size = "150M"
    print(f"\nModel: {model_size}")
    print("-" * 70)
    
    # Test configurations
    batch_sizes = [2, 4, 6, 8, 10, 12, 16]
    seq_lens = [256, 384, 512, 640, 768]
    
    results = []
    
    print(f"\n{'BS':<4} {'SL':<6} {'Step (ms)':<12} {'Tok/step':<10} {'Tok/s':<12} {'Status'}")
    print("-" * 70)
    
    for bs in batch_sizes:
        for sl in seq_lens:
            result = test_config(model_size, bs, sl)
            results.append(result)
            
            if result["error"]:
                status = f"❌ {result['error'][:30]}"
            else:
                status = "✓"
            
            print(f"{bs:<4} {sl:<6} {result['step_time_ms']:<12.1f} {result['tokens_per_step']:<10} {result['tokens_per_sec']:<12,.0f} {status}")
    
    # Filter successful results
    valid = [r for r in results if not r["error"]]
    
    if not valid:
        print("\n❌ All configurations failed!")
        return
    
    # Find best
    best = max(valid, key=lambda r: r["tokens_per_sec"])
    
    # Find pareto-optimal (best for each batch size)
    pareto = {}
    for r in valid:
        bs = r["batch_size"]
        if bs not in pareto or r["tokens_per_sec"] > pareto[bs]["tokens_per_sec"]:
            pareto[bs] = r
    
    print("\n" + "=" * 70)
    print("TOP CONFIGURATIONS")
    print("=" * 70)
    
    top5 = sorted(valid, key=lambda r: -r["tokens_per_sec"])[:5]
    
    print(f"\n{'Rank':<6} {'BS':<4} {'SL':<6} {'Tok/s':<12} {'Step (ms)':<12} {'Est Hours'}")
    print("-" * 70)
    
    total_tokens = 476_000_000
    
    for i, r in enumerate(top5, 1):
        hours = total_tokens / r["tokens_per_sec"] / 3600
        print(f"{i:<6} {r['batch_size']:<4} {r['seq_len']:<6} {r['tokens_per_sec']:<12,.0f} {r['step_time_ms']:<12.1f} {hours:.2f}h")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    hours = total_tokens / best["tokens_per_sec"] / 3600
    
    print(f"\n  Best config: BS={best['batch_size']}, SeqLen={best['seq_len']}")
    print(f"  Throughput: {best['tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Step time: {best['step_time_ms']:.1f}ms")
    print(f"  Estimated time: {hours:.2f} hours")
    
    print(f"\n  Command:")
    print(f"  ./venv/bin/python scripts/train.py --size 150M --tinystories-only \\")
    print(f"      --batch-size {best['batch_size']} --seq-len {best['seq_len']}")
    
    # Also test without checkpointing to see if it helps
    print("\n" + "=" * 70)
    print("TESTING WITHOUT GRADIENT CHECKPOINTING")
    print("=" * 70)
    
    # Modify config to disable checkpointing
    test_no_ckpt = test_config_no_checkpoint(best["batch_size"], best["seq_len"])
    if test_no_ckpt and not test_no_ckpt.get("error"):
        hours_no_ckpt = total_tokens / test_no_ckpt["tokens_per_sec"] / 3600
        print(f"\n  Without checkpoint: {test_no_ckpt['tokens_per_sec']:,.0f} tok/s ({hours_no_ckpt:.2f}h)")
        print(f"  With checkpoint: {best['tokens_per_sec']:,.0f} tok/s ({hours:.2f}h)")
        
        if test_no_ckpt["tokens_per_sec"] > best["tokens_per_sec"]:
            print(f"\n  ⚠️  Checkpointing is slowing things down!")
            print(f"  Consider disabling it if memory allows.")


def test_config_no_checkpoint(batch_size: int, seq_len: int, num_steps: int = 20):
    """Test without gradient checkpointing."""
    try:
        model = TransformerLM(
            vocab_size=50257, 
            num_layers=12, 
            dims=768, 
            num_heads=12, 
            checkpoint=False  # Disabled!
        )
        mx.eval(model.parameters())
        
        optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=0.01)
        
        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")
        
        state = [model.state, optimizer.state]
        
        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, y):
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            return loss
        
        dummy_batch = mx.random.randint(0, 50257, (batch_size, seq_len + 1))
        x, y = dummy_batch[:, :-1], dummy_batch[:, 1:]
        
        # Warmup
        for _ in range(5):
            loss = step(x, y)
            mx.eval(loss)
        
        # Profile
        step_times = []
        start = time.perf_counter()
        
        for _ in range(num_steps):
            loss = step(x, y)
            mx.eval(loss)
            step_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
        
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_sec = (batch_size * seq_len) / (avg_step_time / 1000)
        
        del model, optimizer
        gc.collect()
        
        return {"tokens_per_sec": tokens_per_sec, "step_time_ms": avg_step_time}
        
    except Exception as e:
        gc.collect()
        return {"error": str(e)}


if __name__ == "__main__":
    main()
