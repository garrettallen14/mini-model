#!/usr/bin/env python3
"""
Profile and compare data loading strategies.

Compares:
1. Original TinyStoriesLoader (streaming, slow)
2. FastDataLoader (cached, memory-mapped)
3. InMemoryLoader (everything in RAM, fastest)
"""

import gc
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def profile_original_loader(batch_size: int = 8, seq_len: int = 512, num_batches: int = 100):
    """Profile the original streaming loader."""
    print("\n[1/3] Original TinyStoriesLoader (Streaming)")
    print("-" * 50)
    
    from data.loader import TinyStoriesLoader
    from data.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    loader = TinyStoriesLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    
    batch_times = []
    first_batch_time = None
    
    start = time.perf_counter()
    total_start = start
    
    for i, batch in enumerate(loader.iterate_batches()):
        batch_time = time.perf_counter() - start
        batch_times.append(batch_time * 1000)
        
        if i == 0:
            first_batch_time = batch_time
            print(f"  First batch: {first_batch_time*1000:.1f}ms (includes dataset load)")
        
        if i >= num_batches:
            break
        
        start = time.perf_counter()
    
    total_time = time.perf_counter() - total_start
    
    # Exclude first batch from avg (it includes dataset load)
    avg_time = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else batch_times[0]
    
    print(f"  Batches: {len(batch_times)}")
    print(f"  Avg batch time (excl first): {avg_time:.2f}ms")
    print(f"  Min/Max: {min(batch_times[1:]):.2f} / {max(batch_times[1:]):.2f}ms")
    print(f"  Total time: {total_time:.2f}s")
    
    tokens_per_sec = (num_batches * batch_size * seq_len) / total_time
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    return {
        "name": "Original (Streaming)",
        "avg_batch_ms": avg_time,
        "first_batch_ms": first_batch_time * 1000,
        "tokens_per_sec": tokens_per_sec,
    }


def profile_fast_loader(batch_size: int = 8, seq_len: int = 512, num_batches: int = 100):
    """Profile the fast cached loader."""
    print("\n[2/3] FastDataLoader (Cached + Prefetch)")
    print("-" * 50)
    
    from data.fast_loader import FastDataLoader
    from data.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    loader = FastDataLoader(
        seq_len=seq_len, 
        batch_size=batch_size, 
        tokenizer=tokenizer,
        prefetch=True,
        prefetch_buffer=16,
    )
    
    # Pre-load (separate from batch timing)
    load_start = time.perf_counter()
    loader.load_tinystories()
    load_time = time.perf_counter() - load_start
    print(f"  Cache load time: {load_time:.2f}s")
    
    batch_times = []
    start = time.perf_counter()
    total_start = start
    
    for i, batch in enumerate(loader.iterate_batches()):
        batch_time = time.perf_counter() - start
        batch_times.append(batch_time * 1000)
        
        if i >= num_batches:
            break
        
        start = time.perf_counter()
    
    total_time = time.perf_counter() - total_start
    avg_time = sum(batch_times) / len(batch_times)
    
    print(f"  Batches: {len(batch_times)}")
    print(f"  Avg batch time: {avg_time:.2f}ms")
    print(f"  Min/Max: {min(batch_times):.2f} / {max(batch_times):.2f}ms")
    print(f"  Total time: {total_time:.2f}s")
    
    tokens_per_sec = (num_batches * batch_size * seq_len) / total_time
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    return {
        "name": "Fast (Cached+Prefetch)",
        "avg_batch_ms": avg_time,
        "load_time_s": load_time,
        "tokens_per_sec": tokens_per_sec,
    }


def profile_inmemory_loader(batch_size: int = 8, seq_len: int = 512, num_batches: int = 100):
    """Profile the in-memory loader."""
    print("\n[3/3] InMemoryLoader (Full RAM)")
    print("-" * 50)
    
    from data.fast_loader import InMemoryLoader
    from data.tokenizer import get_tokenizer
    
    tokenizer = get_tokenizer()
    loader = InMemoryLoader(
        seq_len=seq_len, 
        batch_size=batch_size, 
        tokenizer=tokenizer,
    )
    
    # Pre-load
    load_start = time.perf_counter()
    loader.load_tinystories()
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time:.2f}s")
    
    batch_times = []
    start = time.perf_counter()
    total_start = start
    
    for i, batch in enumerate(loader.iterate_batches()):
        batch_time = time.perf_counter() - start
        batch_times.append(batch_time * 1000)
        
        if i >= num_batches:
            break
        
        start = time.perf_counter()
    
    total_time = time.perf_counter() - total_start
    avg_time = sum(batch_times) / len(batch_times)
    
    print(f"  Batches: {len(batch_times)}")
    print(f"  Avg batch time: {avg_time:.2f}ms")
    print(f"  Min/Max: {min(batch_times):.2f} / {max(batch_times):.2f}ms")
    print(f"  Total time: {total_time:.2f}s")
    
    tokens_per_sec = (num_batches * batch_size * seq_len) / total_time
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    return {
        "name": "InMemory (Full RAM)",
        "avg_batch_ms": avg_time,
        "load_time_s": load_time,
        "tokens_per_sec": tokens_per_sec,
    }


def profile_with_training(batch_size: int = 8, seq_len: int = 512, num_steps: int = 50):
    """Profile data loading integrated with actual training steps."""
    print("\n" + "=" * 60)
    print("DATA + TRAINING INTEGRATION TEST")
    print("=" * 60)
    
    from functools import partial
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    
    from data.fast_loader import InMemoryLoader
    from data.tokenizer import get_tokenizer
    
    mx.set_default_device(mx.gpu)
    
    # Create model (150M)
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
    
    print("\nCreating 150M model...")
    model = TransformerLM(50257, 12, 768, 12, checkpoint=True)
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
    
    # Load data
    print("Loading data...")
    tokenizer = get_tokenizer()
    loader = InMemoryLoader(seq_len=seq_len, batch_size=batch_size, tokenizer=tokenizer)
    loader.load_tinystories()
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(loader.iterate_batches()):
        batch_mx = mx.array(batch)
        x, y = batch_mx[:, :-1], batch_mx[:, 1:]
        loss = step(x, y)
        mx.eval(loss)
        if i >= 5:
            break
    
    # Profile
    print(f"\nProfiling {num_steps} training steps...")
    
    data_times = []
    compute_times = []
    total_times = []
    
    step_start = time.perf_counter()
    
    for i, batch in enumerate(loader.iterate_batches()):
        # Data loading time
        data_time = time.perf_counter() - step_start
        data_times.append(data_time * 1000)
        
        # Compute time
        compute_start = time.perf_counter()
        batch_mx = mx.array(batch)
        x, y = batch_mx[:, :-1], batch_mx[:, 1:]
        loss = step(x, y)
        mx.eval(loss)
        compute_time = time.perf_counter() - compute_start
        compute_times.append(compute_time * 1000)
        
        total_time = time.perf_counter() - step_start
        total_times.append(total_time * 1000)
        
        if i >= num_steps:
            break
        
        step_start = time.perf_counter()
    
    # Results
    avg_data = sum(data_times) / len(data_times)
    avg_compute = sum(compute_times) / len(compute_times)
    avg_total = sum(total_times) / len(total_times)
    
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / (avg_total / 1000)
    
    compute_efficiency = avg_compute / avg_total * 100
    
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    print(f"  Avg data load time:  {avg_data:.2f}ms")
    print(f"  Avg compute time:    {avg_compute:.2f}ms")
    print(f"  Avg total step time: {avg_total:.2f}ms")
    print(f"  Compute efficiency:  {compute_efficiency:.1f}%")
    print(f"  Tokens/sec:          {tokens_per_sec:,.0f}")
    
    # Time estimates
    total_tokens = 476_000_000
    hours = total_tokens / tokens_per_sec / 3600
    
    print(f"\n  Estimated time for TinyStories (476M):")
    print(f"    {hours:.2f} hours")
    
    return {
        "avg_data_ms": avg_data,
        "avg_compute_ms": avg_compute,
        "avg_total_ms": avg_total,
        "compute_efficiency": compute_efficiency,
        "tokens_per_sec": tokens_per_sec,
        "estimated_hours": hours,
    }


def main():
    print("=" * 60)
    print("DATA LOADING PROFILER")
    print("=" * 60)
    
    batch_size = 8
    seq_len = 512
    num_batches = 100
    
    print(f"\nConfig: batch_size={batch_size}, seq_len={seq_len}")
    
    results = []
    
    # Profile each loader
    gc.collect()
    r1 = profile_original_loader(batch_size, seq_len, num_batches)
    results.append(r1)
    
    gc.collect()
    r2 = profile_fast_loader(batch_size, seq_len, num_batches)
    results.append(r2)
    
    gc.collect()
    r3 = profile_inmemory_loader(batch_size, seq_len, num_batches)
    results.append(r3)
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Loader':<25} {'Avg Batch':<12} {'Tokens/s':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline = results[0]["tokens_per_sec"]
    for r in results:
        speedup = r["tokens_per_sec"] / baseline
        print(f"{r['name']:<25} {r['avg_batch_ms']:.2f}ms{'':<6} {r['tokens_per_sec']:>10,.0f} {speedup:.1f}x")
    
    # Profile with training
    gc.collect()
    training_result = profile_with_training(batch_size, seq_len, num_steps=50)
    
    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"\n  Use InMemoryLoader for TinyStories training")
    print(f"  Expected: {training_result['tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Time estimate: {training_result['estimated_hours']:.2f} hours")
    print(f"  Compute efficiency: {training_result['compute_efficiency']:.1f}%")


if __name__ == "__main__":
    main()
