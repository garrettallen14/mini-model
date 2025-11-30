# Mini-Model: 150M Parameter LLM

Train a small but capable language model. Supports:
- **CUDA** (A40, RTX, etc.) - Recommended, fastest
- **MLX** (Apple Silicon) - For local development

## Quick Start (CUDA/RunPod)

```bash
# On RunPod or any CUDA machine
pip install datasets transformers tiktoken tqdm

# Quick test (2 min)
python train_cuda.py --size 150M --max-tokens 10000000

# Full TinyStories (15-20 min on A40)
python train_cuda.py --size 150M --batch-size 32

# Generate text
python generate_cuda.py --checkpoint checkpoints/*/final.pt --prompt "Once upon a time"
```

See [RUNPOD.md](RUNPOD.md) for detailed RunPod instructions.

## Project Structure

```
mini-model/
├── model/                   # Model architecture
│   ├── __init__.py
│   └── transformer.py       # MiniModelLM with RoPE, SwiGLU
├── data/                    # Data pipeline
│   ├── __init__.py
│   ├── config.py            # Dataset configs & 3-phase curriculum
│   ├── tokenizer.py         # GPT-2 tokenizer (tiktoken)
│   └── loader.py            # Streaming loader with packing
├── eval/                    # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py           # Perplexity, induction heads, few-shot
├── configs/                 # Configuration files
│   ├── __init__.py
│   └── model_configs.py     # 10M, 50M, 150M, 200M configs
├── scripts/
│   ├── setup.sh             # Environment setup
│   ├── test_baseline.py     # Validate MLX pipeline
│   ├── verify_150m.py       # Verify 150M fits in memory
│   ├── download_data.py     # Download datasets
│   ├── test_data_pipeline.py # Validate data pipeline
│   ├── train.py             # Main training script (curriculum)
│   ├── train_tinystories.py # TinyStories-only training
│   ├── generate.py          # Text generation
│   └── monitor_resources.py # Resource monitoring
├── checkpoints/             # Saved model weights
├── mlx-examples/            # Official MLX examples
├── venv/                    # Python virtual environment
├── requirements.txt
├── outline.md              # Full project strategy
├── datas.md                # Data plan (3B tokens)
└── README.md
```

## Phase 1: Environment & Baseline

### Quick Start

```bash
# 1. Setup environment
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. Run baseline test (small model, 1000 steps)
python scripts/test_baseline.py

# 3. Verify 150M config
python scripts/verify_150m.py
```

### Expected Results

| Metric | Target |
|--------|--------|
| Baseline test passes | ✓ |
| Loss decreases | ✓ |
| No OOM on 150M | ✓ |
| Training speed | >10 steps/sec (small), >2 steps/sec (150M) |

### Monitoring

In a separate terminal:
```bash
# Basic resource monitoring
python scripts/monitor_resources.py

# Detailed GPU metrics (requires sudo)
pip install asitop
sudo asitop
```

## Target Configuration (150M)

From `outline.md`:

```python
config = {
    "vocab_size": 50257,      # GPT-2 BPE
    "num_layers": 12,
    "dims": 768,
    "num_heads": 12,
    "ff_mult": 3.5,           # SwiGLU efficiency
    "max_seq_len": 512,
    "checkpoint": True,       # Gradient checkpointing
}
```

### Memory Budget (24GB M4)

| Component | Estimate |
|-----------|----------|
| Weights (BF16) | ~300MB |
| Optimizer states | ~900MB |
| Activations (checkpointed) | ~4-6GB |
| Overhead | ~1GB |
| **Total** | **~8-10GB** |
| **Headroom** | **14-16GB** ✓ |

## Phase 2: Data Pipeline ✓

### Quick Test (10M tokens, ~6 min)

```bash
./venv/bin/python scripts/train_tinystories.py --small --max-tokens 10000000
```

### Full TinyStories Training (~8 hours)

```bash
./venv/bin/python scripts/train_tinystories.py
```

### Training Results (10M test)

| Metric | Value |
|--------|-------|
| Tokens/sec | ~29,000 |
| Initial loss | 8.0 (PPL 3014) |
| Final loss | 2.7 (PPL 14.5) |
| Time | 6 minutes |

### Data Mix (3B tokens from datas.md)

| Category | Tokens | Source |
|----------|--------|--------|
| Math | 1.05B | OpenWebMath + MetaMathQA |
| Code | 900M | StarCoder Python |
| General | 750M | TinyStories + Cosmopedia |
| Books | 300M | Dolma |

## Next Phases

- **Phase 3**: Full 3B token training with curriculum
- **Phase 4**: Evaluation suite
- **Phase 5**: MoE upcycling (optional)

## Key Insights

1. **Synthetic data is leverage** — TinyStories + Cosmopedia at 1% cost
2. **Curriculum matters** — simple→complex trains 1.5-2x faster
3. **MLX gotchas** — always `mx.eval()`, use BF16, `@mx.compile` for speed
4. **Induction heads** — if not forming by 20% training, diagnose

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [mlx-examples](https://github.com/ml-explore/mlx-examples)
- [TinyStories Paper](https://arxiv.org/abs/2305.07759)
