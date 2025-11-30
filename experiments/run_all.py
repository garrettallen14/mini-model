#!/usr/bin/env python3
# Fix hf_transfer issue on RunPod
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

"""
🧪 COMPREHENSIVE EXPERIMENTAL SUITE

Run all experiments to find optimal training configuration:
1. Data loading optimization
2. Memory optimization (checkpointing, accumulation)
3. Optimizer comparison (AdamW, Adam, Lion, Sophia, etc.)
4. Learning rate sweep
5. Batch size / gradient accumulation
6. Precision (FP32, FP16, BF16)
7. Kernel profiling

Usage:
    python experiments/run_all.py              # Run all experiments
    python experiments/run_all.py --quick      # Quick sweep
    python experiments/run_all.py --optimizer  # Just optimizer experiments
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.data_loading import run_data_experiments
from experiments.memory_optimization import run_memory_experiments  
from experiments.optimizer_comparison import run_optimizer_experiments
from experiments.learning_rate import run_lr_experiments
from experiments.precision import run_precision_experiments
from experiments.kernel_profiling import run_kernel_profiling


def run_all_experiments(quick: bool = False, output_dir: str = "experiment_results"):
    """Run complete experimental suite."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("=" * 70)
    print("🧪 MINI-MODEL EXPERIMENTAL SUITE")
    print("=" * 70)
    
    # 1. Data Loading
    print("\n[1/6] Data Loading Experiments...")
    results["data_loading"] = run_data_experiments(quick=quick)
    
    # 2. Memory Optimization
    print("\n[2/6] Memory Optimization Experiments...")
    results["memory"] = run_memory_experiments(quick=quick)
    
    # 3. Optimizer Comparison
    print("\n[3/6] Optimizer Comparison...")
    results["optimizer"] = run_optimizer_experiments(quick=quick)
    
    # 4. Learning Rate
    print("\n[4/6] Learning Rate Sweep...")
    results["learning_rate"] = run_lr_experiments(quick=quick)
    
    # 5. Precision
    print("\n[5/6] Precision Experiments...")
    results["precision"] = run_precision_experiments(quick=quick)
    
    # 6. Kernel Profiling
    print("\n[6/6] Kernel Profiling...")
    results["kernels"] = run_kernel_profiling(quick=quick)
    
    # Save results
    with open(output_path / "all_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print_summary(results)
    
    return results


def print_summary(results: dict):
    """Print experiment summary with recommendations."""
    
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    # Best optimizer
    if "optimizer" in results and results["optimizer"]:
        opt_results = results["optimizer"]
        if opt_results.get("best"):
            print(f"\n  🏆 Best Optimizer: {opt_results['best']['name']}")
            print(f"     Final Loss: {opt_results['best']['final_loss']:.4f}")
            print(f"     Throughput: {opt_results['best']['tokens_per_sec']:,.0f} tok/s")
    
    # Best learning rate
    if "learning_rate" in results and results["learning_rate"]:
        lr_results = results["learning_rate"]
        if lr_results.get("best_lr"):
            print(f"\n  🎯 Best Learning Rate: {lr_results['best_lr']:.2e}")
            print(f"     Final Loss: {lr_results['best_loss']:.4f}")
    
    # Memory optimization
    if "memory" in results and results["memory"]:
        mem_results = results["memory"]
        print(f"\n  💾 Memory Optimization:")
        if mem_results.get("best_config"):
            cfg = mem_results["best_config"]
            print(f"     Checkpointing: {cfg.get('checkpoint', 'N/A')}")
            print(f"     Grad Accumulation: {cfg.get('grad_accum', 1)}")
            print(f"     Peak Memory: {cfg.get('peak_memory_gb', 'N/A'):.2f} GB")
    
    # Best precision
    if "precision" in results and results["precision"]:
        prec_results = results["precision"]
        if prec_results.get("best"):
            print(f"\n  ⚡ Best Precision: {prec_results['best']['dtype']}")
            print(f"     Throughput: {prec_results['best']['tokens_per_sec']:,.0f} tok/s")
    
    # Final recommendation
    print("\n" + "-" * 70)
    print("  📋 RECOMMENDED TRAINING COMMAND:")
    print("-" * 70)
    
    # Build command from results
    cmd_parts = ["python train_cuda.py --size 150M"]
    
    if "optimizer" in results and results["optimizer"].get("best"):
        opt_name = results["optimizer"]["best"]["name"]
        cmd_parts.append(f"--optimizer {opt_name.lower()}")
    
    if "learning_rate" in results and results["learning_rate"].get("best_lr"):
        cmd_parts.append(f"--lr {results['learning_rate']['best_lr']:.0e}")
    
    if "memory" in results and results["memory"].get("best_config"):
        cfg = results["memory"]["best_config"]
        if cfg.get("grad_accum", 1) > 1:
            cmd_parts.append(f"--grad-accum {cfg['grad_accum']}")
    
    print(f"\n  {' '.join(cmd_parts)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--optimizer", action="store_true", help="Only run optimizer experiments")
    parser.add_argument("--memory", action="store_true", help="Only run memory experiments")
    parser.add_argument("--lr", action="store_true", help="Only run learning rate experiments")
    parser.add_argument("--data", action="store_true", help="Only run data loading experiments")
    parser.add_argument("--precision", action="store_true", help="Only run precision experiments")
    parser.add_argument("--kernels", action="store_true", help="Only run kernel profiling")
    parser.add_argument("--output-dir", default="experiment_results")
    
    args = parser.parse_args()
    
    # Run specific experiment or all
    if args.optimizer:
        run_optimizer_experiments(quick=args.quick)
    elif args.memory:
        run_memory_experiments(quick=args.quick)
    elif args.lr:
        run_lr_experiments(quick=args.quick)
    elif args.data:
        run_data_experiments(quick=args.quick)
    elif args.precision:
        run_precision_experiments(quick=args.quick)
    elif args.kernels:
        run_kernel_profiling(quick=args.quick)
    else:
        run_all_experiments(quick=args.quick, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
