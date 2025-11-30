#!/usr/bin/env python3
"""
Deep profiling of training pipeline.

Profiles:
1. Memory usage (model, optimizer, activations)
2. Throughput at different batch sizes
3. Data loading vs compute time
4. GPU utilization patterns
5. Optimal configuration finder

Run before long training to catch issues early.
"""

import gc
import json
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Try to get memory info
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class ProfileResult:
    """Single profiling result."""
    batch_size: int
    seq_len: int
    model_size: str
    
    # Memory (MB)
    model_memory_mb: float
    optimizer_memory_mb: float
    peak_memory_mb: float
    available_memory_mb: float
    
    # Speed
    tokens_per_sec: float
    steps_per_sec: float
    step_time_ms: float
    
    # Breakdown
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    data_load_time_ms: float
    
    # Efficiency
    compute_efficiency: float  # compute / total time
    memory_efficiency: float   # used / available
    
    # Status
    oom: bool = False
    error: str = None


class TransformerLM(nn.Module):
    """Same model as train.py."""
    
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


def get_model_config(size: str) -> dict:
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
        "200M": {"num_layers": 16, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    return configs[size]


def count_params(model) -> int:
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def estimate_memory_mb(num_params: int, batch_size: int, seq_len: int, dims: int, num_layers: int) -> Dict[str, float]:
    """Estimate memory usage in MB."""
    bytes_per_param = 2  # BF16
    
    # Model weights
    model_mb = (num_params * bytes_per_param) / (1024**2)
    
    # Optimizer states (AdamW: m + v + params)
    optimizer_mb = model_mb * 3
    
    # Activations (rough estimate with checkpointing)
    # Per layer: batch * seq * hidden * 4 (attention intermediate)
    act_per_layer = batch_size * seq_len * dims * 4 * bytes_per_param
    # With checkpointing, we only store layer outputs
    activations_mb = (act_per_layer * num_layers) / (1024**2)
    
    # Gradients
    gradients_mb = model_mb
    
    return {
        "model_mb": model_mb,
        "optimizer_mb": optimizer_mb,
        "activations_mb": activations_mb,
        "gradients_mb": gradients_mb,
        "total_mb": model_mb + optimizer_mb + activations_mb + gradients_mb,
    }


def get_system_memory() -> Dict[str, float]:
    """Get system memory info."""
    if not HAS_PSUTIL:
        return {"total_gb": 24.0, "available_gb": 12.0}  # Assume M4 24GB
    
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent,
    }


def profile_single_config(
    model_size: str,
    batch_size: int,
    seq_len: int,
    num_steps: int = 20,
    warmup_steps: int = 5,
) -> ProfileResult:
    """Profile a single configuration."""
    
    config = get_model_config(model_size)
    vocab_size = 50257
    
    # Get initial memory
    gc.collect()
    initial_mem = get_system_memory()
    
    try:
        # Create model
        model = TransformerLM(vocab_size=vocab_size, **config)
        mx.eval(model.parameters())
        
        num_params = count_params(model)
        
        # Memory after model creation
        post_model_mem = get_system_memory()
        model_memory = initial_mem["available_gb"] - post_model_mem["available_gb"]
        
        # Create optimizer
        optimizer = optim.AdamW(learning_rate=5e-4, weight_decay=0.01)
        
        # Loss function
        def loss_fn(model, x, y):
            logits = model(x)
            return nn.losses.cross_entropy(logits, y, reduction="mean")
        
        # Compiled step
        state = [model.state, optimizer.state]
        
        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, y):
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            return loss
        
        # Create dummy data
        dummy_x = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        dummy_y = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Timing breakdown
        forward_times = []
        backward_times = []
        optimizer_times = []
        step_times = []
        
        # Warmup
        for _ in range(warmup_steps):
            loss = step(dummy_x, dummy_y)
            mx.eval(loss)
        
        # Profile steps
        for i in range(num_steps):
            # Full step timing
            step_start = time.perf_counter()
            
            # Forward pass
            fwd_start = time.perf_counter()
            logits = model(dummy_x)
            mx.eval(logits)
            fwd_end = time.perf_counter()
            forward_times.append((fwd_end - fwd_start) * 1000)
            
            # Backward + optimizer (combined in compiled step)
            bwd_start = time.perf_counter()
            loss = step(dummy_x, dummy_y)
            mx.eval(loss)
            bwd_end = time.perf_counter()
            backward_times.append((bwd_end - bwd_start) * 1000)
            
            step_end = time.perf_counter()
            step_times.append((step_end - step_start) * 1000)
        
        # Memory after training
        post_train_mem = get_system_memory()
        peak_memory = initial_mem["available_gb"] - post_train_mem["available_gb"]
        
        # Calculate metrics
        avg_step_time = sum(step_times) / len(step_times)
        tokens_per_step = batch_size * seq_len
        tokens_per_sec = tokens_per_step / (avg_step_time / 1000)
        
        # Memory estimates
        mem_est = estimate_memory_mb(num_params, batch_size, seq_len, 
                                     config["dims"], config["num_layers"])
        
        result = ProfileResult(
            batch_size=batch_size,
            seq_len=seq_len,
            model_size=model_size,
            model_memory_mb=mem_est["model_mb"],
            optimizer_memory_mb=mem_est["optimizer_mb"],
            peak_memory_mb=peak_memory * 1024,  # Convert to MB
            available_memory_mb=initial_mem["available_gb"] * 1024,
            tokens_per_sec=tokens_per_sec,
            steps_per_sec=1000 / avg_step_time,
            step_time_ms=avg_step_time,
            forward_time_ms=sum(forward_times) / len(forward_times),
            backward_time_ms=sum(backward_times) / len(backward_times),
            optimizer_time_ms=0,  # Included in backward
            data_load_time_ms=0,  # Synthetic data
            compute_efficiency=1.0,  # No data loading overhead here
            memory_efficiency=peak_memory / initial_mem["available_gb"],
        )
        
        # Cleanup
        del model, optimizer
        gc.collect()
        
        return result
        
    except Exception as e:
        gc.collect()
        return ProfileResult(
            batch_size=batch_size,
            seq_len=seq_len,
            model_size=model_size,
            model_memory_mb=0,
            optimizer_memory_mb=0,
            peak_memory_mb=0,
            available_memory_mb=initial_mem["available_gb"] * 1024,
            tokens_per_sec=0,
            steps_per_sec=0,
            step_time_ms=0,
            forward_time_ms=0,
            backward_time_ms=0,
            optimizer_time_ms=0,
            data_load_time_ms=0,
            compute_efficiency=0,
            memory_efficiency=0,
            oom="memory" in str(e).lower(),
            error=str(e),
        )


def profile_data_loading(batch_size: int, seq_len: int, num_batches: int = 50) -> Dict[str, float]:
    """Profile data loading overhead."""
    from data.loader import TinyStoriesLoader
    from data.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    loader = TinyStoriesLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    
    load_times = []
    
    start = time.perf_counter()
    for i, batch in enumerate(loader.iterate_batches()):
        load_time = time.perf_counter() - start
        load_times.append(load_time * 1000)
        
        if i >= num_batches:
            break
        
        start = time.perf_counter()
    
    return {
        "avg_load_time_ms": sum(load_times) / len(load_times),
        "min_load_time_ms": min(load_times),
        "max_load_time_ms": max(load_times),
        "total_tokens": loader.total_tokens,
    }


def find_optimal_config(model_size: str, target_memory_pct: float = 0.7) -> Dict:
    """Find optimal batch size and sequence length."""
    
    print(f"\n{'='*60}")
    print(f"Finding optimal config for {model_size}")
    print(f"{'='*60}")
    
    sys_mem = get_system_memory()
    print(f"System memory: {sys_mem['available_gb']:.1f}GB available / {sys_mem['total_gb']:.1f}GB total")
    
    # Test configurations
    batch_sizes = [2, 4, 8, 12, 16]
    seq_lens = [256, 512, 768, 1024]
    
    results = []
    
    for bs in batch_sizes:
        for sl in seq_lens:
            print(f"\n  Testing BS={bs}, SeqLen={sl}...", end=" ", flush=True)
            
            result = profile_single_config(model_size, bs, sl, num_steps=10, warmup_steps=3)
            
            if result.error:
                print(f"❌ {result.error[:50]}")
            elif result.oom:
                print(f"❌ OOM")
            else:
                print(f"✓ {result.tokens_per_sec:,.0f} tok/s, {result.peak_memory_mb:.0f}MB")
                results.append(result)
    
    if not results:
        return {"error": "All configurations failed"}
    
    # Find best by tokens/sec
    best = max(results, key=lambda r: r.tokens_per_sec)
    
    # Find most memory efficient that's still fast
    fast_threshold = best.tokens_per_sec * 0.9
    efficient = [r for r in results if r.tokens_per_sec >= fast_threshold]
    most_efficient = min(efficient, key=lambda r: r.memory_efficiency)
    
    return {
        "fastest": {
            "batch_size": best.batch_size,
            "seq_len": best.seq_len,
            "tokens_per_sec": best.tokens_per_sec,
            "memory_mb": best.peak_memory_mb,
        },
        "most_efficient": {
            "batch_size": most_efficient.batch_size,
            "seq_len": most_efficient.seq_len,
            "tokens_per_sec": most_efficient.tokens_per_sec,
            "memory_mb": most_efficient.peak_memory_mb,
        },
        "all_results": [
            {
                "bs": r.batch_size,
                "sl": r.seq_len,
                "tok_s": r.tokens_per_sec,
                "mem_mb": r.peak_memory_mb,
            }
            for r in sorted(results, key=lambda r: -r.tokens_per_sec)
        ],
    }


def run_full_profile(model_size: str = "150M"):
    """Run comprehensive profiling."""
    
    print("="*70)
    print(f"DEEP PROFILING: {model_size} Model")
    print("="*70)
    
    mx.set_default_device(mx.gpu)
    
    # 1. System info
    print("\n[1/6] System Information")
    print("-"*40)
    sys_mem = get_system_memory()
    print(f"  Total Memory: {sys_mem['total_gb']:.1f} GB")
    print(f"  Available: {sys_mem['available_gb']:.1f} GB")
    print(f"  Used: {sys_mem.get('used_gb', 0):.1f} GB ({sys_mem.get('percent', 0):.0f}%)")
    print(f"  MLX Device: {mx.default_device()}")
    
    # 2. Model info
    print("\n[2/6] Model Architecture")
    print("-"*40)
    config = get_model_config(model_size)
    model = TransformerLM(vocab_size=50257, **config)
    mx.eval(model.parameters())
    num_params = count_params(model)
    print(f"  Layers: {config['num_layers']}")
    print(f"  Dims: {config['dims']}")
    print(f"  Heads: {config['num_heads']}")
    print(f"  Checkpoint: {config['checkpoint']}")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    del model
    gc.collect()
    
    # 3. Memory estimates
    print("\n[3/6] Memory Estimates (BS=8, SeqLen=512)")
    print("-"*40)
    mem_est = estimate_memory_mb(num_params, 8, 512, config["dims"], config["num_layers"])
    print(f"  Model weights: {mem_est['model_mb']:.0f} MB")
    print(f"  Optimizer states: {mem_est['optimizer_mb']:.0f} MB")
    print(f"  Activations: {mem_est['activations_mb']:.0f} MB")
    print(f"  Gradients: {mem_est['gradients_mb']:.0f} MB")
    print(f"  Estimated total: {mem_est['total_mb']:.0f} MB ({mem_est['total_mb']/1024:.1f} GB)")
    
    # 4. Data loading
    print("\n[4/6] Data Loading Speed")
    print("-"*40)
    data_profile = profile_data_loading(8, 512, num_batches=30)
    print(f"  Avg load time: {data_profile['avg_load_time_ms']:.2f} ms/batch")
    print(f"  Min/Max: {data_profile['min_load_time_ms']:.2f} / {data_profile['max_load_time_ms']:.2f} ms")
    
    # 5. Training speed at default config
    print("\n[5/6] Training Speed (BS=8, SeqLen=512)")
    print("-"*40)
    result = profile_single_config(model_size, 8, 512, num_steps=20, warmup_steps=5)
    
    if result.error:
        print(f"  ❌ Error: {result.error}")
    else:
        print(f"  Step time: {result.step_time_ms:.1f} ms")
        print(f"  Forward: {result.forward_time_ms:.1f} ms")
        print(f"  Backward+Optim: {result.backward_time_ms:.1f} ms")
        print(f"  Tokens/sec: {result.tokens_per_sec:,.0f}")
        print(f"  Steps/sec: {result.steps_per_sec:.2f}")
        print(f"  Peak memory: {result.peak_memory_mb:.0f} MB")
    
    # 6. Find optimal config
    print("\n[6/6] Finding Optimal Configuration")
    optimal = find_optimal_config(model_size)
    
    # Summary
    print("\n" + "="*70)
    print("PROFILING SUMMARY")
    print("="*70)
    
    if "error" not in optimal:
        print(f"\n  Fastest Configuration:")
        print(f"    Batch Size: {optimal['fastest']['batch_size']}")
        print(f"    Seq Length: {optimal['fastest']['seq_len']}")
        print(f"    Throughput: {optimal['fastest']['tokens_per_sec']:,.0f} tokens/sec")
        print(f"    Memory: {optimal['fastest']['memory_mb']:.0f} MB")
        
        print(f"\n  Recommended (90% speed, better efficiency):")
        print(f"    Batch Size: {optimal['most_efficient']['batch_size']}")
        print(f"    Seq Length: {optimal['most_efficient']['seq_len']}")
        print(f"    Throughput: {optimal['most_efficient']['tokens_per_sec']:,.0f} tokens/sec")
        
        # Time estimate
        tokens_per_sec = optimal['most_efficient']['tokens_per_sec']
        total_tokens = 476_000_000  # TinyStories
        hours = total_tokens / tokens_per_sec / 3600
        
        print(f"\n  Time Estimate (TinyStories 476M tokens):")
        print(f"    {hours:.1f} hours at {tokens_per_sec:,.0f} tok/s")
        
        # Recommended command
        bs = optimal['most_efficient']['batch_size']
        sl = optimal['most_efficient']['seq_len']
        print(f"\n  Recommended Command:")
        print(f"    ./venv/bin/python scripts/train.py --size {model_size} --tinystories-only \\")
        print(f"        --batch-size {bs} --seq-len {sl}")
    
    print("\n" + "="*70)
    
    # Save results
    output_path = project_root / "profiling_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model_size": model_size,
            "system_memory": sys_mem,
            "memory_estimates": mem_est,
            "data_loading": data_profile,
            "optimal_config": optimal,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    
    return optimal


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile training pipeline")
    parser.add_argument("--size", type=str, default="150M",
                       choices=["10M", "50M", "150M", "200M"])
    parser.add_argument("--quick", action="store_true",
                       help="Quick profile (fewer configs)")
    
    args = parser.parse_args()
    
    run_full_profile(args.size)
