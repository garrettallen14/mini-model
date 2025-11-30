#!/usr/bin/env python3
"""
⚡ PRECISION EXPERIMENTS

Tests:
1. FP32 (baseline, slow but stable)
2. FP16 (fast, may have overflow issues)
3. BF16 (fast, better dynamic range)
4. TF32 (NVIDIA tensor cores)
5. Mixed precision strategies
"""

import gc
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_model(dtype=torch.float32):
    """Create test model with specified dtype."""
    from profile_cuda import MiniModel, MODEL_CONFIGS
    config = MODEL_CONFIGS["50M"]
    model = MiniModel(**config)
    return model.to(dtype=dtype)


def test_precision(
    dtype_name: str,
    batch_size: int = 16,
    seq_len: int = 512,
    num_steps: int = 100,
) -> Dict:
    """Test a specific precision setting."""
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Configure precision
    if dtype_name == "fp32":
        dtype = torch.float32
        use_amp = False
        amp_dtype = None
    elif dtype_name == "fp16":
        dtype = torch.float32  # Model in FP32, use AMP for FP16
        use_amp = True
        amp_dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.float32
        use_amp = True
        amp_dtype = torch.bfloat16
    elif dtype_name == "tf32":
        dtype = torch.float32
        use_amp = False
        amp_dtype = None
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif dtype_name == "bf16_pure":
        # Pure BF16 (no mixed precision, model weights in BF16)
        dtype = torch.bfloat16
        use_amp = False
        amp_dtype = None
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")
    
    try:
        model = create_model(dtype).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        
        if use_amp:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler('cuda')
        
        x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        
        model.train()
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            if use_amp:
                with autocast('cuda', dtype=amp_dtype):
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
        start = time.perf_counter()
        
        losses = []
        for _ in range(num_steps):
            optimizer.zero_grad()
            
            if use_amp:
                with autocast('cuda', dtype=amp_dtype):
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
            
            losses.append(loss.item())
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        tokens_per_sec = (num_steps * batch_size * seq_len) / total_time
        final_loss = sum(losses[-10:]) / 10
        
        # Check for NaN/Inf
        has_nan = any(not torch.isfinite(torch.tensor(l)) for l in losses)
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "dtype": dtype_name,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_gb": peak_memory,
            "final_loss": final_loss,
            "has_nan": has_nan,
            "success": not has_nan,
        }
        
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "dtype": dtype_name,
            "error": str(e),
            "success": False,
        }


def run_precision_experiments(quick: bool = False) -> Dict:
    """Run all precision experiments."""
    
    print("\n" + "=" * 60)
    print("⚡ PRECISION EXPERIMENTS")
    print("=" * 60)
    
    num_steps = 50 if quick else 150
    
    dtypes = ["fp32", "fp16", "bf16", "tf32", "bf16_pure"]
    if quick:
        dtypes = ["fp32", "bf16"]
    
    results = {}
    
    print(f"\n  {'Precision':<12} {'Tok/s':<12} {'Memory':<10} {'Loss':<10} {'Status'}")
    print("  " + "─" * 55)
    
    for dtype_name in dtypes:
        result = test_precision(dtype_name, num_steps=num_steps)
        results[dtype_name] = result
        
        if result.get("success"):
            print(f"  {dtype_name:<12} {result['tokens_per_sec']:<12,.0f} {result['peak_memory_gb']:<10.2f} {result['final_loss']:<10.4f} ✓")
        else:
            error = result.get("error", "NaN detected")[:30]
            print(f"  {dtype_name:<12} {'—':<12} {'—':<10} {'—':<10} ✗ {error}")
    
    # Find best
    valid = {k: v for k, v in results.items() if v.get("success")}
    if valid:
        best = max(valid.items(), key=lambda x: x[1]["tokens_per_sec"])
        
        print(f"\n  🏆 Best: {best[0]} ({best[1]['tokens_per_sec']:,.0f} tok/s)")
        
        # Compare to FP32 baseline
        if "fp32" in valid:
            speedup = best[1]["tokens_per_sec"] / valid["fp32"]["tokens_per_sec"]
            mem_save = valid["fp32"]["peak_memory_gb"] - best[1]["peak_memory_gb"]
            print(f"  Speedup vs FP32: {speedup:.2f}x")
            print(f"  Memory saved: {mem_save:.2f} GB")
        
        return {"results": results, "best": {"dtype": best[0], **best[1]}}
    
    return {"results": results, "best": None}


if __name__ == "__main__":
    run_precision_experiments()
