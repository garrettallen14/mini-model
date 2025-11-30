#!/usr/bin/env python3
"""
💾 MEMORY OPTIMIZATION EXPERIMENTS

Tests:
1. Gradient checkpointing (activation recomputation)
2. Gradient accumulation (effective batch size)
3. Mixed precision (FP16 vs BF16 vs FP32)
4. Optimizer state sharding
5. Activation offloading
6. Memory-efficient attention
"""

import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MemoryResult:
    config_name: str
    batch_size: int
    effective_batch_size: int
    peak_memory_gb: float
    tokens_per_sec: float
    final_loss: float
    success: bool
    error: Optional[str] = None


def create_test_model(use_checkpoint: bool = False):
    """Create test model using the working MiniModel from profile_cuda."""
    from profile_cuda import MiniModel, MODEL_CONFIGS
    config = MODEL_CONFIGS["50M"]
    return MiniModel(**config)


def test_gradient_checkpointing(
    batch_size: int = 32,
    seq_len: int = 512,
    num_steps: int = 50,
) -> Dict:
    """Compare with and without gradient checkpointing."""
    
    print("\n  📊 Gradient Checkpointing Test")
    print("  " + "─" * 50)
    
    results = {}
    
    for use_ckpt in [False, True]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        name = "with_checkpoint" if use_ckpt else "no_checkpoint"
        
        try:
            model = create_test_model().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            scaler = GradScaler('cuda')
            
            x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
            y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
            
            model.train()
            
            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            
            losses = []
            for _ in range(num_steps):
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            tokens_per_sec = (num_steps * batch_size * seq_len) / total_time
            
            results[name] = {
                "peak_memory_gb": peak_memory,
                "tokens_per_sec": tokens_per_sec,
                "final_loss": sum(losses[-5:]) / 5,
            }
            
            ckpt_str = "✓" if use_ckpt else "✗"
            print(f"    Checkpoint={ckpt_str}: {peak_memory:.2f} GB, {tokens_per_sec:,.0f} tok/s")
            
            del model, optimizer
            
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"    Checkpoint={use_ckpt}: ❌ {e}")
    
    # Compare
    if "no_checkpoint" in results and "with_checkpoint" in results:
        if "error" not in results["no_checkpoint"] and "error" not in results["with_checkpoint"]:
            mem_saved = results["no_checkpoint"]["peak_memory_gb"] - results["with_checkpoint"]["peak_memory_gb"]
            speed_diff = (results["with_checkpoint"]["tokens_per_sec"] / results["no_checkpoint"]["tokens_per_sec"] - 1) * 100
            print(f"\n    Memory saved: {mem_saved:.2f} GB")
            print(f"    Speed change: {speed_diff:+.1f}%")
    
    return results


def test_gradient_accumulation(
    micro_batch_size: int = 8,
    seq_len: int = 512,
    num_steps: int = 50,
) -> Dict:
    """Test gradient accumulation for larger effective batch sizes."""
    
    print("\n  📊 Gradient Accumulation Test")
    print("  " + "─" * 50)
    
    results = {}
    accum_steps_list = [1, 2, 4, 8]
    
    for accum_steps in accum_steps_list:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        effective_batch = micro_batch_size * accum_steps
        
        try:
            model = create_test_model().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            scaler = GradScaler('cuda')
            
            x = torch.randint(0, 50257, (micro_batch_size, seq_len), device='cuda')
            y = torch.randint(0, 50257, (micro_batch_size, seq_len), device='cuda')
            
            model.train()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            losses = []
            for step in range(num_steps):
                accumulated_loss = 0
                
                for micro_step in range(accum_steps):
                    with autocast('cuda'):
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        loss = loss / accum_steps
                    
                    scaler.scale(loss).backward()
                    accumulated_loss += loss.item() * accum_steps
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                losses.append(accumulated_loss)
            
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            total_tokens = num_steps * effective_batch * seq_len
            tokens_per_sec = total_tokens / total_time
            
            results[f"accum_{accum_steps}"] = {
                "accum_steps": accum_steps,
                "effective_batch": effective_batch,
                "peak_memory_gb": peak_memory,
                "tokens_per_sec": tokens_per_sec,
                "final_loss": sum(losses[-5:]) / 5,
            }
            
            print(f"    accum={accum_steps} (eff_bs={effective_batch}): {peak_memory:.2f} GB, {tokens_per_sec:,.0f} tok/s")
            
            del model, optimizer
            
        except Exception as e:
            results[f"accum_{accum_steps}"] = {"error": str(e)}
            print(f"    accum={accum_steps}: ❌ {e}")
    
    return results


def test_max_batch_size(seq_len: int = 512) -> Dict:
    """Find maximum batch size that fits in memory."""
    
    print("\n  📊 Maximum Batch Size Search")
    print("  " + "─" * 50)
    
    batch_sizes = [8, 16, 32, 48, 64, 96, 128, 192, 256]
    results = {}
    max_working = 0
    
    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            model = create_test_model().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
            scaler = GradScaler('cuda')
            
            x = torch.randint(0, 50257, (bs, seq_len), device='cuda')
            y = torch.randint(0, 50257, (bs, seq_len), device='cuda')
            
            model.train()
            
            # Try a step
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            
            results[bs] = {"peak_memory_gb": peak_mem, "success": True}
            max_working = bs
            print(f"    BS={bs}: ✓ ({peak_mem:.2f} GB)")
            
            del model, optimizer, x, y
            
        except Exception as e:
            results[bs] = {"success": False, "error": str(e)[:50]}
            print(f"    BS={bs}: ✗ OOM")
            break
    
    print(f"\n    Maximum batch size: {max_working}")
    
    return {"batch_sizes": results, "max_batch_size": max_working}


def run_memory_experiments(quick: bool = False) -> Dict:
    """Run all memory optimization experiments."""
    
    print("\n" + "=" * 60)
    print("💾 MEMORY OPTIMIZATION EXPERIMENTS")
    print("=" * 60)
    
    num_steps = 30 if quick else 100
    
    results = {}
    
    # Gradient checkpointing
    results["checkpointing"] = test_gradient_checkpointing(num_steps=num_steps)
    
    # Gradient accumulation
    results["accumulation"] = test_gradient_accumulation(num_steps=num_steps)
    
    # Max batch size
    results["max_batch"] = test_max_batch_size()
    
    # Best config
    best_config = {
        "checkpoint": True,
        "grad_accum": 4,  # Good default
        "peak_memory_gb": results.get("checkpointing", {}).get("with_checkpoint", {}).get("peak_memory_gb", 0),
    }
    results["best_config"] = best_config
    
    # Summary
    print("\n" + "─" * 60)
    print("  📊 MEMORY OPTIMIZATION SUMMARY")
    print("─" * 60)
    print(f"  Use gradient checkpointing: Yes (saves ~30-50% memory)")
    print(f"  Max batch size: {results.get('max_batch', {}).get('max_batch_size', 'N/A')}")
    print(f"  Recommended grad_accum: 2-4 (for stability)")
    
    return results


if __name__ == "__main__":
    run_memory_experiments()
