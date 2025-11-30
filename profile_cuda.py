#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE CUDA PROFILING SCRIPT

Run this FIRST before any long training to:
1. Verify GPU setup
2. Find optimal batch size
3. Estimate training time
4. Catch OOM issues early

Usage:
    python profile_cuda.py                    # Full profiling
    python profile_cuda.py --quick            # Fast check (2 min)
    python profile_cuda.py --size 350M        # Test larger model
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(text: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text: str):
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")


def format_bytes(b: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.2f}h"


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# MODEL (same as train_cuda.py)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: float = 3.5):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ffn_norm = RMSNorm(dim)
        hidden_dim = int(dim * ff_mult)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniModel(nn.Module):
    def __init__(self, vocab_size=50257, num_layers=12, dim=768, num_heads=12, ff_mult=3.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads, ff_mult) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.output(self.norm(h))


MODEL_CONFIGS = {
    "10M": {"num_layers": 4, "dim": 256, "num_heads": 4},
    "50M": {"num_layers": 8, "dim": 512, "num_heads": 8},
    "150M": {"num_layers": 12, "dim": 768, "num_heads": 12},
    "350M": {"num_layers": 24, "dim": 1024, "num_heads": 16},
    "760M": {"num_layers": 24, "dim": 1536, "num_heads": 16},
}


# =============================================================================
# PROFILING FUNCTIONS
# =============================================================================

@dataclass
class ProfileResult:
    batch_size: int
    seq_len: int
    tokens_per_sec: float
    step_time_ms: float
    memory_used_gb: float
    memory_peak_gb: float
    forward_ms: float
    backward_ms: float
    success: bool
    error: str = ""


def profile_gpu_info():
    """Get detailed GPU information."""
    print_header("🖥️  GPU INFORMATION")
    
    if not torch.cuda.is_available():
        print("  ❌ CUDA NOT AVAILABLE!")
        return None
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / 1e9,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    
    print(f"""
  GPU:              {info['name']}
  Compute:          {info['compute_capability']}
  VRAM:             {info['total_memory_gb']:.1f} GB
  SMs:              {info['multi_processor_count']}
  CUDA:             {info['cuda_version']}
  PyTorch:          {info['pytorch_version']}
  cuDNN:            {info['cudnn_version']}
  Flash Attention:  {'✓ Available' if hasattr(F, 'scaled_dot_product_attention') else '✗ Not available'}
  torch.compile:    {'✓ Available' if hasattr(torch, 'compile') else '✗ Not available'}
    """)
    
    return info


def profile_memory_baseline():
    """Get baseline memory usage."""
    print_section("📊 Memory Baseline")
    
    clear_gpu_memory()
    
    # Allocate and measure
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"  Allocated:  {allocated:.2f} GB")
    print(f"  Reserved:   {reserved:.2f} GB")
    print(f"  Total VRAM: {total:.1f} GB")
    print(f"  Free:       {total - allocated:.2f} GB")
    
    return total


def profile_model_memory(model_size: str):
    """Profile model memory usage."""
    print_section(f"🧠 Model Memory ({model_size})")
    
    clear_gpu_memory()
    
    config = MODEL_CONFIGS[model_size]
    model = MiniModel(**config).cuda()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    # Measure actual GPU memory
    torch.cuda.synchronize()
    model_memory = torch.cuda.memory_allocated() / 1e9
    
    print(f"  Parameters:     {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"  Param Memory:   {param_memory:.3f} GB")
    print(f"  GPU Allocated:  {model_memory:.3f} GB")
    
    # Optimizer memory (Adam: 2 states per param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.synchronize()
    with_optim = torch.cuda.memory_allocated() / 1e9
    optim_memory = with_optim - model_memory
    
    print(f"  Optimizer:      {optim_memory:.3f} GB")
    print(f"  Total (static): {with_optim:.3f} GB")
    
    del model, optimizer
    clear_gpu_memory()
    
    return num_params, with_optim


def profile_single_config(
    model_size: str,
    batch_size: int,
    seq_len: int,
    num_steps: int = 10,
    warmup_steps: int = 3,
    use_amp: bool = True,
) -> ProfileResult:
    """Profile a single batch size / seq len configuration."""
    
    clear_gpu_memory()
    
    try:
        config = MODEL_CONFIGS[model_size]
        model = MiniModel(**config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scaler = GradScaler('cuda') if use_amp else None
        
        # Dummy data
        x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        
        model.train()
        
        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            if use_amp:
                with autocast('cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Profile
        forward_times = []
        backward_times = []
        step_times = []
        
        for _ in range(num_steps):
            optimizer.zero_grad()
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            # Forward
            fwd_start = time.perf_counter()
            if use_amp:
                with autocast('cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - fwd_start) * 1000)
            
            # Backward
            bwd_start = time.perf_counter()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            backward_times.append((time.perf_counter() - bwd_start) * 1000)
            
            step_times.append((time.perf_counter() - step_start) * 1000)
        
        # Calculate metrics
        avg_step = sum(step_times) / len(step_times)
        avg_fwd = sum(forward_times) / len(forward_times)
        avg_bwd = sum(backward_times) / len(backward_times)
        tokens_per_step = batch_size * seq_len
        tokens_per_sec = tokens_per_step / (avg_step / 1000)
        
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_peak = torch.cuda.max_memory_allocated() / 1e9
        
        del model, optimizer, x, y
        clear_gpu_memory()
        
        return ProfileResult(
            batch_size=batch_size,
            seq_len=seq_len,
            tokens_per_sec=tokens_per_sec,
            step_time_ms=avg_step,
            memory_used_gb=memory_used,
            memory_peak_gb=memory_peak,
            forward_ms=avg_fwd,
            backward_ms=avg_bwd,
            success=True,
        )
        
    except Exception as e:
        clear_gpu_memory()
        return ProfileResult(
            batch_size=batch_size,
            seq_len=seq_len,
            tokens_per_sec=0,
            step_time_ms=0,
            memory_used_gb=0,
            memory_peak_gb=0,
            forward_ms=0,
            backward_ms=0,
            success=False,
            error=str(e)[:50],
        )


def profile_throughput_sweep(model_size: str, quick: bool = False):
    """Sweep batch sizes to find optimal throughput."""
    print_header(f"⚡ THROUGHPUT SWEEP ({model_size})")
    
    if quick:
        batch_sizes = [8, 16, 32, 48]
        seq_lens = [512]
    else:
        batch_sizes = [4, 8, 16, 24, 32, 48, 64, 96, 128]
        seq_lens = [256, 512, 1024]
    
    results = []
    
    print(f"\n  {'BS':<6} {'SeqLen':<8} {'Tok/s':<12} {'Step(ms)':<10} {'Mem(GB)':<10} {'Status'}")
    print("  " + "─" * 60)
    
    for seq_len in seq_lens:
        for bs in batch_sizes:
            result = profile_single_config(model_size, bs, seq_len)
            results.append(result)
            
            if result.success:
                status = "✓"
                print(f"  {bs:<6} {seq_len:<8} {result.tokens_per_sec:<12,.0f} {result.step_time_ms:<10.1f} {result.memory_peak_gb:<10.2f} {status}")
            else:
                status = f"✗ {result.error}"
                print(f"  {bs:<6} {seq_len:<8} {'—':<12} {'—':<10} {'—':<10} {status}")
                # Stop this seq_len if OOM
                if "out of memory" in result.error.lower():
                    break
    
    return results


def profile_data_loading():
    """Profile data loading speed."""
    print_section("📂 Data Loading")
    
    try:
        import tiktoken
        from datasets import load_dataset
        
        print("  Loading TinyStories sample...")
        start = time.perf_counter()
        ds = load_dataset("roneneldan/TinyStories", split="train[:1000]")
        load_time = time.perf_counter() - start
        print(f"  Dataset load: {load_time:.2f}s")
        
        enc = tiktoken.get_encoding("gpt2")
        
        print("  Tokenizing 1000 samples...")
        start = time.perf_counter()
        tokens = []
        for ex in ds:
            if ex["text"]:
                tokens.extend(enc.encode(ex["text"]))
        tok_time = time.perf_counter() - start
        
        print(f"  Tokenization: {tok_time:.2f}s ({len(tokens):,} tokens)")
        print(f"  Speed: {len(tokens)/tok_time:,.0f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"  ⚠️ Error: {e}")
        return False


def run_live_training_test(model_size: str, batch_size: int, seq_len: int, num_steps: int = 50):
    """Run a live training test with real-time stats."""
    print_header(f"🚀 LIVE TRAINING TEST ({model_size}, BS={batch_size})")
    
    clear_gpu_memory()
    
    config = MODEL_CONFIGS[model_size]
    model = MiniModel(**config).cuda()
    
    # Compile for speed
    if hasattr(torch, 'compile'):
        print("  Compiling model...")
        model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Dummy data
    x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    model.train()
    tokens_per_step = batch_size * seq_len
    
    print(f"\n  Running {num_steps} steps...")
    print(f"\n  {'Step':<8} {'Loss':<10} {'Tok/s':<12} {'Mem(GB)':<10} {'Time'}")
    print("  " + "─" * 55)
    
    losses = []
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        
        loss_val = loss.item()
        losses.append(loss_val)
        tokens_per_sec = tokens_per_step / step_time
        mem_gb = torch.cuda.memory_allocated() / 1e9
        
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  {step+1:<8} {loss_val:<10.4f} {tokens_per_sec:<12,.0f} {mem_gb:<10.2f} {elapsed:.1f}s")
    
    # Summary
    total_time = time.perf_counter() - start_time
    total_tokens = num_steps * tokens_per_step
    
    # Calculate steady-state throughput (exclude first step which includes compilation)
    if num_steps > 10:
        # Use steps 10+ for steady-state measurement
        steady_state_tokens = (num_steps - 10) * tokens_per_step
        steady_state_time = total_time - (time.perf_counter() - start_time) * (10 / num_steps)
        # Actually, let's just use a simpler approach based on the last steps
        last_step_time = step_time  # Last step time
        steady_state_tps = tokens_per_step / last_step_time
    else:
        steady_state_tps = total_tokens / total_time
    
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\n  {'─' * 55}")
    print(f"  Total time:      {total_time:.2f}s")
    print(f"  Total tokens:    {total_tokens:,}")
    print(f"  Avg tokens/sec:  {avg_tokens_per_sec:,.0f} (includes compile)")
    print(f"  Real tokens/sec: {steady_state_tps:,.0f} ⬅️ steady-state")
    print(f"  Final loss:      {losses[-1]:.4f}")
    print(f"  Peak memory:     {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    # Time estimates using steady-state throughput
    tinystories_tokens = 476_000_000
    tinystories_hours = tinystories_tokens / steady_state_tps / 3600
    
    full_3b_tokens = 3_000_000_000
    full_3b_hours = full_3b_tokens / steady_state_tps / 3600
    
    print(f"\n  📊 TIME ESTIMATES (using steady-state {steady_state_tps:,.0f} tok/s):")
    print(f"  TinyStories (476M): {format_time(tinystories_hours * 3600)} (${tinystories_hours * 0.40:.2f} on A40)")
    print(f"  Full 3B curriculum: {format_time(full_3b_hours * 3600)} (${full_3b_hours * 0.40:.2f} on A40)")
    
    del model, optimizer
    clear_gpu_memory()
    
    return avg_tokens_per_sec


def find_optimal_config(model_size: str, results: List[ProfileResult]):
    """Analyze results and find optimal configuration."""
    print_header("🎯 OPTIMAL CONFIGURATION")
    
    valid = [r for r in results if r.success]
    
    if not valid:
        print("  ❌ All configurations failed!")
        return None
    
    # Best throughput
    best = max(valid, key=lambda r: r.tokens_per_sec)
    
    # Best efficiency (high throughput, reasonable memory)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    efficient = [r for r in valid if r.memory_peak_gb < total_vram * 0.8]
    if efficient:
        best_efficient = max(efficient, key=lambda r: r.tokens_per_sec)
    else:
        best_efficient = best
    
    print(f"""
  🏆 FASTEST:
     Batch Size: {best.batch_size}
     Seq Length: {best.seq_len}
     Throughput: {best.tokens_per_sec:,.0f} tokens/sec
     Step Time:  {best.step_time_ms:.1f}ms
     Memory:     {best.memory_peak_gb:.2f} GB

  ⚖️  RECOMMENDED (80% VRAM headroom):
     Batch Size: {best_efficient.batch_size}
     Seq Length: {best_efficient.seq_len}
     Throughput: {best_efficient.tokens_per_sec:,.0f} tokens/sec
     Memory:     {best_efficient.memory_peak_gb:.2f} GB
    """)
    
    # Time estimates
    tokens_per_sec = best_efficient.tokens_per_sec
    tinystories_time = 476_000_000 / tokens_per_sec
    full_time = 3_000_000_000 / tokens_per_sec
    
    print(f"""
  ⏱️  TIME ESTIMATES:
     TinyStories (476M): {format_time(tinystories_time)}
     Full 3B:            {format_time(full_time)}
    """)
    
    print(f"""
  📋 RECOMMENDED COMMAND:
     python train_cuda.py --size {model_size} --batch-size {best_efficient.batch_size} --seq-len {best_efficient.seq_len}
    """)
    
    return best_efficient


def main():
    parser = argparse.ArgumentParser(description="Profile CUDA training")
    parser.add_argument("--size", default="150M", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--quick", action="store_true", help="Quick profiling (fewer configs)")
    parser.add_argument("--live-steps", type=int, default=50, help="Steps for live test")
    
    args = parser.parse_args()
    
    print_header("🔬 MINI-MODEL CUDA PROFILER", "═")
    print(f"  Model: {args.size}")
    print(f"  Mode:  {'Quick' if args.quick else 'Full'}")
    
    # 1. GPU Info
    gpu_info = profile_gpu_info()
    if gpu_info is None:
        sys.exit(1)
    
    # 2. Memory baseline
    total_vram = profile_memory_baseline()
    
    # 3. Model memory
    num_params, static_memory = profile_model_memory(args.size)
    
    # 4. Data loading
    profile_data_loading()
    
    # 5. Throughput sweep
    results = profile_throughput_sweep(args.size, quick=args.quick)
    
    # 6. Find optimal
    optimal = find_optimal_config(args.size, results)
    
    # 7. Live test with optimal config
    if optimal:
        run_live_training_test(
            args.size, 
            optimal.batch_size, 
            optimal.seq_len, 
            num_steps=args.live_steps
        )
    
    print_header("✅ PROFILING COMPLETE", "═")
    print("""
  Next steps:
  1. Review the recommended configuration above
  2. Run training:
     python train_cuda.py --size {size} --batch-size {bs} --seq-len {sl}
  3. Monitor with: watch -n 1 nvidia-smi
    """.format(
        size=args.size,
        bs=optimal.batch_size if optimal else 32,
        sl=optimal.seq_len if optimal else 512,
    ))


if __name__ == "__main__":
    main()
