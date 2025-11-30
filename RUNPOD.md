# Running on RunPod (A40 48GB)

## Quick Start

### 1. Start your pod and open terminal

### 2. Clone and setup
```bash
cd /workspace
git clone <your-repo-url> mini-model
cd mini-model
chmod +x setup_runpod.sh
./setup_runpod.sh
```

### 3. Run training
```bash
# Quick test (~2 min)
python train_cuda.py --size 150M --max-tokens 10000000

# Full TinyStories (~15-20 min)
python train_cuda.py --size 150M --batch-size 32

# Full 3B curriculum (~2-3 hours)
python train_cuda.py --size 150M --batch-size 48
```

## Expected Performance (A40 48GB)

| Model | Batch Size | Tokens/sec | TinyStories (476M) |
|-------|------------|------------|-------------------|
| 150M  | 32         | ~25,000    | ~20 min           |
| 150M  | 48         | ~35,000    | ~15 min           |
| 350M  | 16         | ~12,000    | ~40 min           |

## Model Sizes

| Size | Params | VRAM (BS=32) | Notes |
|------|--------|--------------|-------|
| 10M  | 29M    | ~2 GB        | Testing |
| 50M  | 82M    | ~4 GB        | Quick experiments |
| 150M | 162M   | ~8 GB        | Target model |
| 350M | 350M   | ~16 GB       | Larger if desired |

## Commands

```bash
# With Weights & Biases logging
python train_cuda.py --size 150M --wandb

# Custom batch size (larger = faster, more VRAM)
python train_cuda.py --size 150M --batch-size 64

# Resume from checkpoint
python train_cuda.py --size 150M --resume checkpoints/run_name/step_5000.pt

# Longer sequences
python train_cuda.py --size 150M --seq-len 1024 --batch-size 16
```

## Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training progress
tail -f checkpoints/*/training.log
```

## Cost Estimate

- A40: $0.40/hr
- TinyStories (476M): ~20 min = **~$0.13**
- Full 3B curriculum: ~2-3 hrs = **~$1.00-1.50**

## Troubleshooting

**OOM Error**: Reduce batch size
```bash
python train_cuda.py --size 150M --batch-size 16
```

**Slow training**: Make sure you're using GPU
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

**Data not found**: The script auto-downloads TinyStories on first run
