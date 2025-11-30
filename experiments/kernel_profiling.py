#!/usr/bin/env python3
"""
🔬 KERNEL PROFILING

Deep analysis of GPU kernel performance:
1. Forward pass breakdown
2. Backward pass breakdown  
3. Attention kernel analysis
4. Memory bandwidth utilization
5. Compute vs memory bound analysis
6. torch.compile effects
"""

import gc
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_model(use_compile: bool = False):
    """Create test model."""
    from profile_cuda import MiniModel, MODEL_CONFIGS
    config = MODEL_CONFIGS["50M"]
    model = MiniModel(**config).cuda()
    
    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model


def profile_forward_backward(
    batch_size: int = 16,
    seq_len: int = 512,
    num_iterations: int = 50,
    use_compile: bool = False,
) -> Dict:
    """Profile forward and backward pass separately."""
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model = create_model(use_compile=use_compile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    model.train()
    
    # Warmup (especially important for torch.compile)
    warmup_iters = 10 if use_compile else 5
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    
    # Profile
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward
        torch.cuda.synchronize()
        fwd_start = time.perf_counter()
        
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        torch.cuda.synchronize()
        fwd_end = time.perf_counter()
        forward_times.append((fwd_end - fwd_start) * 1000)
        
        # Backward
        bwd_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_end = time.perf_counter()
        backward_times.append((bwd_end - bwd_start) * 1000)
        
        # Optimizer
        opt_start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_end = time.perf_counter()
        optimizer_times.append((opt_end - opt_start) * 1000)
    
    del model, optimizer
    gc.collect()
    
    avg_fwd = sum(forward_times) / len(forward_times)
    avg_bwd = sum(backward_times) / len(backward_times)
    avg_opt = sum(optimizer_times) / len(optimizer_times)
    total = avg_fwd + avg_bwd + avg_opt
    
    return {
        "forward_ms": avg_fwd,
        "backward_ms": avg_bwd,
        "optimizer_ms": avg_opt,
        "total_ms": total,
        "forward_pct": avg_fwd / total * 100,
        "backward_pct": avg_bwd / total * 100,
        "optimizer_pct": avg_opt / total * 100,
    }


def profile_attention_variants(
    batch_size: int = 16,
    seq_len: int = 512,
    num_iterations: int = 50,
) -> Dict:
    """Compare attention implementations."""
    
    print("\n  📊 Attention Implementation Comparison")
    print("  " + "─" * 50)
    
    results = {}
    
    # Test Flash Attention (via scaled_dot_product_attention)
    dim = 512
    num_heads = 8
    head_dim = dim // num_heads
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    # 1. Flash Attention (SDPA with is_causal)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["flash_attention"] = {"time_ms": flash_time}
    print(f"    Flash Attention (SDPA): {flash_time:.3f}ms")
    
    # 2. Manual attention
    scale = head_dim ** -0.5
    mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
    torch.cuda.synchronize()
    manual_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["manual_attention"] = {"time_ms": manual_time}
    print(f"    Manual Attention: {manual_time:.3f}ms")
    
    # Speedup
    speedup = manual_time / flash_time
    print(f"\n    Flash Attention speedup: {speedup:.2f}x")
    
    results["speedup"] = speedup
    
    return results


def profile_compile_effect(
    batch_size: int = 16,
    seq_len: int = 512,
    num_iterations: int = 50,
) -> Dict:
    """Compare with and without torch.compile."""
    
    print("\n  📊 torch.compile Effect")
    print("  " + "─" * 50)
    
    results = {}
    
    for use_compile in [False, True]:
        name = "compiled" if use_compile else "eager"
        
        result = profile_forward_backward(
            batch_size=batch_size,
            seq_len=seq_len,
            num_iterations=num_iterations,
            use_compile=use_compile,
        )
        
        results[name] = result
        tokens_per_sec = (batch_size * seq_len) / (result["total_ms"] / 1000)
        print(f"    {name}: {result['total_ms']:.2f}ms/step, {tokens_per_sec:,.0f} tok/s")
    
    # Speedup
    if "eager" in results and "compiled" in results:
        speedup = results["eager"]["total_ms"] / results["compiled"]["total_ms"]
        print(f"\n    torch.compile speedup: {speedup:.2f}x")
        results["speedup"] = speedup
    
    return results


def profile_memory_bandwidth(
    batch_size: int = 16,
    seq_len: int = 512,
    dim: int = 768,
    num_iterations: int = 100,
) -> Dict:
    """Estimate memory bandwidth utilization."""
    
    print("\n  📊 Memory Bandwidth Analysis")
    print("  " + "─" * 50)
    
    # Create test tensors
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float16)
    w = torch.randn(dim, dim * 4, device='cuda', dtype=torch.float16)
    
    # Measure matmul throughput
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(num_iterations):
        y = torch.matmul(x, w)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate bandwidth
    bytes_read = (x.numel() + w.numel()) * 2  # FP16 = 2 bytes
    bytes_written = y.numel() * 2
    total_bytes = (bytes_read + bytes_written) * num_iterations
    
    bandwidth_gb_s = total_bytes / elapsed / 1e9
    
    # A40 theoretical: ~696 GB/s
    theoretical_bw = 696  # GB/s for A40
    utilization = bandwidth_gb_s / theoretical_bw * 100
    
    print(f"    Measured bandwidth: {bandwidth_gb_s:.1f} GB/s")
    print(f"    Theoretical (A40): {theoretical_bw} GB/s")
    print(f"    Utilization: {utilization:.1f}%")
    
    # Compute intensity analysis
    flops = 2 * batch_size * seq_len * dim * (dim * 4) * num_iterations
    tflops = flops / elapsed / 1e12
    
    # A40 theoretical: ~37.4 TFLOPS (FP16)
    theoretical_tflops = 37.4
    compute_util = tflops / theoretical_tflops * 100
    
    print(f"\n    Measured compute: {tflops:.1f} TFLOPS")
    print(f"    Theoretical (A40 FP16): {theoretical_tflops} TFLOPS")
    print(f"    Compute utilization: {compute_util:.1f}%")
    
    # Compute vs memory bound
    arithmetic_intensity = flops / total_bytes
    roofline_crossover = theoretical_tflops * 1e12 / (theoretical_bw * 1e9)
    
    if arithmetic_intensity > roofline_crossover:
        bound = "compute-bound"
    else:
        bound = "memory-bound"
    
    print(f"\n    Arithmetic intensity: {arithmetic_intensity:.1f} FLOP/byte")
    print(f"    Roofline crossover: {roofline_crossover:.1f} FLOP/byte")
    print(f"    Bottleneck: {bound}")
    
    return {
        "bandwidth_gb_s": bandwidth_gb_s,
        "utilization_pct": utilization,
        "tflops": tflops,
        "compute_util_pct": compute_util,
        "bottleneck": bound,
    }


def profile_layer_breakdown(batch_size: int = 16, seq_len: int = 512) -> Dict:
    """Profile individual layer types."""
    
    print("\n  📊 Layer-wise Breakdown")
    print("  " + "─" * 50)
    
    dim = 768
    num_heads = 12
    num_iterations = 50
    
    results = {}
    
    # 1. Embedding
    embed = nn.Embedding(50257, dim).cuda()
    x_idx = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = embed(x_idx)
    torch.cuda.synchronize()
    embed_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["embedding"] = embed_time
    print(f"    Embedding: {embed_time:.3f}ms")
    
    # 2. Attention
    attn = nn.MultiheadAttention(dim, num_heads, batch_first=True).cuda().half()
    x = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float16)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out, _ = attn(x, x, x, is_causal=True)
    torch.cuda.synchronize()
    attn_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["attention"] = attn_time
    print(f"    Attention: {attn_time:.3f}ms")
    
    # 3. FFN
    ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim),
    ).cuda().half()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = ffn(x)
    torch.cuda.synchronize()
    ffn_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["ffn"] = ffn_time
    print(f"    FFN: {ffn_time:.3f}ms")
    
    # 4. LayerNorm
    ln = nn.LayerNorm(dim).cuda().half()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out = ln(x)
    torch.cuda.synchronize()
    ln_time = (time.perf_counter() - start) / num_iterations * 1000
    
    results["layernorm"] = ln_time
    print(f"    LayerNorm: {ln_time:.3f}ms")
    
    # Breakdown
    total = sum(results.values())
    print(f"\n    Total per layer: {total:.3f}ms")
    print(f"\n    Breakdown:")
    for name, t in sorted(results.items(), key=lambda x: -x[1]):
        pct = t / total * 100
        print(f"      {name}: {pct:.1f}%")
    
    return results


def run_kernel_profiling(quick: bool = False) -> Dict:
    """Run all kernel profiling experiments."""
    
    print("\n" + "=" * 60)
    print("🔬 KERNEL PROFILING")
    print("=" * 60)
    
    num_iterations = 30 if quick else 100
    
    results = {}
    
    # Forward/backward breakdown
    print("\n  📊 Forward/Backward Breakdown")
    print("  " + "─" * 50)
    
    result = profile_forward_backward(num_iterations=num_iterations)
    results["fwd_bwd"] = result
    
    print(f"    Forward: {result['forward_ms']:.2f}ms ({result['forward_pct']:.1f}%)")
    print(f"    Backward: {result['backward_ms']:.2f}ms ({result['backward_pct']:.1f}%)")
    print(f"    Optimizer: {result['optimizer_ms']:.2f}ms ({result['optimizer_pct']:.1f}%)")
    
    # Attention comparison
    results["attention"] = profile_attention_variants(num_iterations=num_iterations)
    
    # torch.compile effect
    if hasattr(torch, 'compile'):
        results["compile"] = profile_compile_effect(num_iterations=num_iterations)
    
    # Memory bandwidth
    results["bandwidth"] = profile_memory_bandwidth(num_iterations=num_iterations)
    
    # Layer breakdown
    results["layers"] = profile_layer_breakdown()
    
    # Summary
    print("\n" + "─" * 60)
    print("  📊 KERNEL PROFILING SUMMARY")
    print("─" * 60)
    print(f"  Flash Attention: {results.get('attention', {}).get('speedup', 0):.1f}x faster than manual")
    if "compile" in results:
        print(f"  torch.compile: {results['compile'].get('speedup', 1):.2f}x speedup")
    print(f"  Bottleneck: {results.get('bandwidth', {}).get('bottleneck', 'unknown')}")
    
    return results


if __name__ == "__main__":
    run_kernel_profiling()
