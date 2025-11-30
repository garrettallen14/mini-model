#!/bin/bash
# RunPod Setup Script
# Run this after cloning the repo on RunPod

set -e

echo "=================================================="
echo "Mini-Model RunPod Setup"
echo "=================================================="

# Check GPU
echo ""
echo "[1/4] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -q datasets transformers tiktoken tqdm wandb accelerate

# Quick test
echo ""
echo "[3/4] Quick model test..."
python -c "
import torch
import torch.nn as nn

# Quick transformer test
class QuickTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 4, batch_first=True),
            num_layers=2
        )
        self.out = nn.Linear(256, 1000)
    
    def forward(self, x):
        return self.out(self.transformer(self.embed(x)))

model = QuickTest().cuda()
x = torch.randint(0, 1000, (4, 64)).cuda()
y = model(x)
print(f'Quick test passed: {x.shape} -> {y.shape}')
print(f'Memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "=================================================="
echo "Ready to train! Commands:"
echo "=================================================="
echo ""
echo "# Quick test (10M tokens, ~2 min):"
echo "python train_cuda.py --size 150M --max-tokens 10000000"
echo ""
echo "# Full TinyStories (476M tokens, ~15-20 min on A40):"
echo "python train_cuda.py --size 150M --batch-size 32"
echo ""
echo "# With W&B logging:"
echo "python train_cuda.py --size 150M --wandb"
echo ""
echo "# Larger model (350M):"
echo "python train_cuda.py --size 350M --batch-size 16"
echo ""
