#!/usr/bin/env python3
"""
🔧 OPTIMIZER COMPARISON EXPERIMENTS

Tests:
1. AdamW (baseline)
2. Adam (without weight decay fix)
3. SGD + momentum
4. Lion (memory efficient)
5. Sophia (2nd order approximation)
6. AdaFactor (memory efficient)
7. 8-bit Adam (bitsandbytes)

Each tested for:
- Convergence speed (loss at step N)
- Memory usage
- Throughput
- Final loss quality
"""

import gc
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


# Import model
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class OptimizerResult:
    name: str
    final_loss: float
    losses: List[float]
    tokens_per_sec: float
    memory_gb: float
    converged: bool
    error: Optional[str] = None


def create_model():
    """Create test model (50M for faster experiments)."""
    from profile_cuda import MiniModel, MODEL_CONFIGS
    config = MODEL_CONFIGS["50M"]
    return MiniModel(**config)


def create_optimizer(name: str, model: nn.Module, lr: float = 5e-4) -> torch.optim.Optimizer:
    """Create optimizer by name."""
    
    params = model.parameters()
    
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95))
    
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.01)
    
    elif name == "sgd_nesterov":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.01, nesterov=True)
    
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=0.01)
    
    elif name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=0.01)
    
    elif name == "lion":
        # Lion optimizer (requires installation)
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr * 0.1, weight_decay=0.01)  # Lion uses lower LR
        except ImportError:
            print("    ⚠️ lion_pytorch not installed, using AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    elif name == "sophia":
        # Sophia optimizer
        try:
            from sophia import SophiaG
            return SophiaG(params, lr=lr, weight_decay=0.01)
        except ImportError:
            print("    ⚠️ sophia not installed, using AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    elif name == "adam_8bit":
        # 8-bit Adam (bitsandbytes)
        try:
            import bitsandbytes as bnb
            return bnb.optim.Adam8bit(params, lr=lr, betas=(0.9, 0.95))
        except ImportError:
            print("    ⚠️ bitsandbytes not installed, using AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    elif name == "adafactor":
        # AdaFactor (memory efficient)
        try:
            from transformers import Adafactor
            return Adafactor(params, lr=lr, relative_step=False, warmup_init=False)
        except ImportError:
            print("    ⚠️ transformers Adafactor not available, using AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def test_optimizer(
    name: str,
    num_steps: int = 200,
    batch_size: int = 16,
    seq_len: int = 512,
    lr: float = 5e-4,
) -> OptimizerResult:
    """Test a single optimizer."""
    
    print(f"\n  Testing {name}...")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model = create_model().cuda()
        optimizer = create_optimizer(name, model, lr)
        scaler = GradScaler('cuda')
        
        # Dummy data
        x = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        y = torch.randint(0, 50257, (batch_size, seq_len), device='cuda')
        
        model.train()
        losses = []
        tokens_per_step = batch_size * seq_len
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        
        # Training
        for step in range(num_steps):
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        
        tokens_per_sec = (num_steps * tokens_per_step) / total_time
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        final_loss = sum(losses[-10:]) / 10
        
        # Check convergence (loss decreased significantly)
        initial_loss = sum(losses[:10]) / 10
        converged = final_loss < initial_loss * 0.9
        
        print(f"    Loss: {initial_loss:.2f} → {final_loss:.2f}")
        print(f"    Throughput: {tokens_per_sec:,.0f} tok/s")
        print(f"    Memory: {memory_gb:.2f} GB")
        print(f"    Converged: {'✓' if converged else '✗'}")
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return OptimizerResult(
            name=name,
            final_loss=final_loss,
            losses=losses,
            tokens_per_sec=tokens_per_sec,
            memory_gb=memory_gb,
            converged=converged,
        )
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return OptimizerResult(
            name=name,
            final_loss=float('inf'),
            losses=[],
            tokens_per_sec=0,
            memory_gb=0,
            converged=False,
            error=str(e),
        )


def run_optimizer_experiments(quick: bool = False) -> Dict:
    """Run optimizer comparison experiments."""
    
    print("\n" + "=" * 60)
    print("🔧 OPTIMIZER COMPARISON")
    print("=" * 60)
    
    # Optimizers to test
    if quick:
        optimizers = ["adamw", "adam", "sgd"]
        num_steps = 100
    else:
        optimizers = [
            "adamw",      # Baseline
            "adam",       # Without decoupled weight decay
            "sgd",        # Simple baseline
            "sgd_nesterov",
            "rmsprop",
            "adafactor",  # Memory efficient
            "lion",       # If installed
            "adam_8bit",  # If bitsandbytes installed
        ]
        num_steps = 300
    
    results = {}
    
    for opt_name in optimizers:
        result = test_optimizer(opt_name, num_steps=num_steps)
        results[opt_name] = {
            "final_loss": result.final_loss,
            "tokens_per_sec": result.tokens_per_sec,
            "memory_gb": result.memory_gb,
            "converged": result.converged,
            "error": result.error,
        }
    
    # Find best
    valid_results = {k: v for k, v in results.items() if v["converged"] and not v.get("error")}
    
    if valid_results:
        # Best by final loss
        best_loss = min(valid_results.items(), key=lambda x: x[1]["final_loss"])
        # Best by throughput  
        best_speed = max(valid_results.items(), key=lambda x: x[1]["tokens_per_sec"])
        # Best by memory
        best_memory = min(valid_results.items(), key=lambda x: x[1]["memory_gb"])
        
        print("\n" + "─" * 60)
        print("  📊 OPTIMIZER RESULTS")
        print("─" * 60)
        print(f"\n  {'Optimizer':<15} {'Loss':<10} {'Tok/s':<12} {'Memory':<10} {'Status'}")
        print("  " + "─" * 55)
        
        for name, r in sorted(results.items(), key=lambda x: x[1].get("final_loss", float('inf'))):
            status = "✓" if r["converged"] else "✗"
            if r.get("error"):
                status = "❌"
            print(f"  {name:<15} {r['final_loss']:<10.4f} {r['tokens_per_sec']:<12,.0f} {r['memory_gb']:<10.2f} {status}")
        
        print(f"\n  🏆 Best Loss: {best_loss[0]} ({best_loss[1]['final_loss']:.4f})")
        print(f"  ⚡ Best Speed: {best_speed[0]} ({best_speed[1]['tokens_per_sec']:,.0f} tok/s)")
        print(f"  💾 Best Memory: {best_memory[0]} ({best_memory[1]['memory_gb']:.2f} GB)")
        
        return {
            "results": results,
            "best": {
                "name": best_loss[0],
                "final_loss": best_loss[1]["final_loss"],
                "tokens_per_sec": best_loss[1]["tokens_per_sec"],
            },
        }
    
    return {"results": results, "best": None}


if __name__ == "__main__":
    run_optimizer_experiments()
