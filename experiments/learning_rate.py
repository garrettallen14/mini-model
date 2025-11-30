#!/usr/bin/env python3
"""
🎯 LEARNING RATE EXPERIMENTS

Tests:
1. LR sweep (1e-5 to 1e-3)
2. Warmup steps comparison
3. LR schedules (cosine, linear, constant)
4. Minimum LR ratio
5. LR finder (automatic)
"""

import gc
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_lr_cosine(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def get_lr_linear(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear decay with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr - (max_lr - min_lr) * progress


def get_lr_constant(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Constant LR with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    return max_lr


def create_model():
    """Create test model."""
    from profile_cuda import MiniModel, MODEL_CONFIGS
    config = MODEL_CONFIGS["50M"]
    return MiniModel(**config)


def run_training(
    lr: float,
    schedule: str = "cosine",
    warmup_steps: int = 100,
    num_steps: int = 500,
    batch_size: int = 16,
    seq_len: int = 512,
    min_lr_ratio: float = 0.1,
) -> Tuple[List[float], float]:
    """Run training with given LR config and return losses."""
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model = create_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    model.train()
    
    # Select schedule
    min_lr = lr * min_lr_ratio
    if schedule == "cosine":
        get_lr = lambda s: get_lr_cosine(s, warmup_steps, num_steps, lr, min_lr)
    elif schedule == "linear":
        get_lr = lambda s: get_lr_linear(s, warmup_steps, num_steps, lr, min_lr)
    else:
        get_lr = lambda s: get_lr_constant(s, warmup_steps, num_steps, lr, min_lr)
    
    losses = []
    lrs = []
    
    for step in range(num_steps):
        # Update LR
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        lrs.append(current_lr)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
    
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    final_loss = sum(losses[-20:]) / 20
    return losses, final_loss


def test_lr_sweep(num_steps: int = 300) -> Dict:
    """Sweep learning rates to find optimal."""
    
    print("\n  📊 Learning Rate Sweep")
    print("  " + "─" * 50)
    
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3, 3e-3]
    results = {}
    
    print(f"\n  {'LR':<12} {'Final Loss':<12} {'Status'}")
    print("  " + "─" * 40)
    
    for lr in lrs:
        try:
            losses, final_loss = run_training(lr=lr, num_steps=num_steps)
            
            # Check for divergence
            if math.isnan(final_loss) or final_loss > losses[0] * 2:
                status = "diverged"
            elif final_loss < losses[0] * 0.5:
                status = "good"
            else:
                status = "ok"
            
            results[lr] = {
                "final_loss": final_loss,
                "initial_loss": losses[0],
                "status": status,
            }
            
            status_icon = "✓" if status == "good" else ("~" if status == "ok" else "✗")
            print(f"  {lr:<12.0e} {final_loss:<12.4f} {status_icon}")
            
        except Exception as e:
            results[lr] = {"error": str(e), "status": "error"}
            print(f"  {lr:<12.0e} {'—':<12} ❌")
    
    # Find best
    valid = {k: v for k, v in results.items() if v.get("status") in ["good", "ok"]}
    if valid:
        best_lr = min(valid.keys(), key=lambda k: valid[k]["final_loss"])
        print(f"\n  🏆 Best LR: {best_lr:.0e} (loss: {valid[best_lr]['final_loss']:.4f})")
        return {"results": results, "best_lr": best_lr, "best_loss": valid[best_lr]["final_loss"]}
    
    return {"results": results, "best_lr": 5e-4}  # Default fallback


def test_warmup_steps(num_steps: int = 500) -> Dict:
    """Test different warmup durations."""
    
    print("\n  📊 Warmup Steps Comparison")
    print("  " + "─" * 50)
    
    warmups = [0, 50, 100, 200, 500, 1000]
    # Scale warmups to num_steps
    warmups = [min(w, num_steps // 2) for w in warmups]
    warmups = list(set(warmups))
    warmups.sort()
    
    results = {}
    
    print(f"\n  {'Warmup':<12} {'Final Loss':<12}")
    print("  " + "─" * 30)
    
    for warmup in warmups:
        try:
            losses, final_loss = run_training(
                lr=5e-4, 
                warmup_steps=warmup, 
                num_steps=num_steps
            )
            
            results[warmup] = {"final_loss": final_loss}
            print(f"  {warmup:<12} {final_loss:<12.4f}")
            
        except Exception as e:
            results[warmup] = {"error": str(e)}
            print(f"  {warmup:<12} ❌")
    
    # Find best
    valid = {k: v for k, v in results.items() if "final_loss" in v}
    if valid:
        best = min(valid.keys(), key=lambda k: valid[k]["final_loss"])
        print(f"\n  🏆 Best warmup: {best} steps")
    
    return {"results": results}


def test_schedules(num_steps: int = 500) -> Dict:
    """Compare LR schedules."""
    
    print("\n  📊 LR Schedule Comparison")
    print("  " + "─" * 50)
    
    schedules = ["constant", "linear", "cosine"]
    results = {}
    
    print(f"\n  {'Schedule':<12} {'Final Loss':<12}")
    print("  " + "─" * 30)
    
    for schedule in schedules:
        try:
            losses, final_loss = run_training(
                lr=5e-4,
                schedule=schedule,
                num_steps=num_steps
            )
            
            results[schedule] = {"final_loss": final_loss}
            print(f"  {schedule:<12} {final_loss:<12.4f}")
            
        except Exception as e:
            results[schedule] = {"error": str(e)}
            print(f"  {schedule:<12} ❌")
    
    return {"results": results}


def lr_finder(
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_steps: int = 200,
    batch_size: int = 16,
    seq_len: int = 512,
) -> Dict:
    """
    LR Range Test (Smith 2015).
    Exponentially increase LR and track loss.
    """
    
    print("\n  📊 LR Range Test (Finder)")
    print("  " + "─" * 50)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model = create_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
    
    model.train()
    
    # Exponential LR increase
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    
    losses = []
    lrs = []
    smoothed_loss = None
    best_loss = float('inf')
    
    current_lr = min_lr
    
    for step in range(num_steps):
        # Update LR
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        
        # Smooth loss
        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss_val
        
        losses.append(smoothed_loss)
        lrs.append(current_lr)
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Stop if loss explodes
        if smoothed_loss > best_loss * 4:
            print(f"    Stopped at LR={current_lr:.2e} (loss exploded)")
            break
        
        current_lr *= lr_mult
    
    del model, optimizer
    
    # Find optimal LR (steepest descent point)
    # Simple heuristic: LR where loss is minimum / 10
    min_loss_idx = losses.index(min(losses))
    suggested_lr = lrs[min_loss_idx] / 10
    
    print(f"    Tested LR range: {min_lr:.0e} to {lrs[-1]:.0e}")
    print(f"    Min loss at LR: {lrs[min_loss_idx]:.2e}")
    print(f"    Suggested max LR: {suggested_lr:.2e}")
    
    return {
        "lrs": lrs,
        "losses": losses,
        "suggested_lr": suggested_lr,
        "min_loss_lr": lrs[min_loss_idx],
    }


def run_lr_experiments(quick: bool = False) -> Dict:
    """Run all learning rate experiments."""
    
    print("\n" + "=" * 60)
    print("🎯 LEARNING RATE EXPERIMENTS")
    print("=" * 60)
    
    num_steps = 200 if quick else 500
    
    results = {}
    
    # LR sweep
    results["sweep"] = test_lr_sweep(num_steps=num_steps)
    
    # Warmup
    results["warmup"] = test_warmup_steps(num_steps=num_steps)
    
    # Schedules
    results["schedules"] = test_schedules(num_steps=num_steps)
    
    # LR finder
    results["finder"] = lr_finder(num_steps=num_steps // 2)
    
    # Extract best
    best_lr = results.get("sweep", {}).get("best_lr", 5e-4)
    best_loss = results.get("sweep", {}).get("best_loss", float('inf'))
    
    results["best_lr"] = best_lr
    results["best_loss"] = best_loss
    
    # Summary
    print("\n" + "─" * 60)
    print("  📊 LEARNING RATE SUMMARY")
    print("─" * 60)
    print(f"  Best learning rate: {best_lr:.0e}")
    print(f"  Best final loss: {best_loss:.4f}")
    print(f"  Recommended warmup: 5-10% of total steps")
    print(f"  Recommended schedule: cosine")
    
    return results


if __name__ == "__main__":
    run_lr_experiments()
