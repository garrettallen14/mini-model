# 🧪 Experimental Suite

Comprehensive experiments to find optimal training configuration.

## Quick Start

```bash
# Run all experiments (30-60 min)
python experiments/run_all.py

# Quick sweep (5-10 min)
python experiments/run_all.py --quick

# Individual experiments
python experiments/run_all.py --optimizer
python experiments/run_all.py --lr
python experiments/run_all.py --memory
python experiments/run_all.py --precision
python experiments/run_all.py --kernels
python experiments/run_all.py --data
```

## Experiments

### 1. Data Loading (`data_loading.py`)
- `num_workers` optimization (0, 2, 4, 8)
- `pin_memory` effect
- `prefetch_factor` tuning
- Tokenizer comparison (tiktoken vs HF)

### 2. Optimizer Comparison (`optimizer_comparison.py`)
- AdamW (baseline)
- Adam (without weight decay fix)
- SGD + momentum
- Lion (memory efficient)
- 8-bit Adam (bitsandbytes)
- AdaFactor

### 3. Memory Optimization (`memory_optimization.py`)
- Gradient checkpointing
- Gradient accumulation
- Maximum batch size search

### 4. Learning Rate (`learning_rate.py`)
- LR sweep (1e-5 to 3e-3)
- Warmup steps comparison
- Schedule comparison (cosine, linear, constant)
- LR Range Test (finder)

### 5. Precision (`precision.py`)
- FP32 (baseline)
- FP16 (mixed precision)
- BF16 (better dynamic range)
- TF32 (tensor cores)
- Pure BF16

### 6. Kernel Profiling (`kernel_profiling.py`)
- Forward/backward breakdown
- Attention implementation comparison
- torch.compile effect
- Memory bandwidth utilization
- Layer-wise breakdown

## Output

Results saved to `experiment_results/`:
```
experiment_results/
├── all_results.json      # Complete results
└── plots/                # Visualization (if matplotlib available)
```

## Expected Results (A40)

| Experiment | Best Config |
|------------|------------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 to 5e-4 |
| Precision | BF16 |
| Batch Size | 48-64 |
| torch.compile | ~1.3-1.5x speedup |

## Using Results

After running experiments, use the recommended command:

```bash
python train_cuda.py --size 150M \
    --batch-size 48 \
    --lr 5e-4 \
    --warmup 500
```
