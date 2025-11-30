#!/bin/bash
# Phase 1 Setup Script
# Run this to set up the mini-model training environment

set -e

echo "=================================================="
echo "Mini-Model Phase 1 Setup"
echo "=================================================="

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo ""
    echo "[1/5] Virtual environment exists, skipping creation"
fi

# Activate venv
source venv/bin/activate
PYTHON=./venv/bin/python
PIP=./venv/bin/pip

echo ""
echo "[2/5] Installing Python dependencies..."
$PIP install -q --upgrade pip
$PIP install -q mlx numpy tqdm psutil

echo ""
echo "[3/5] Verifying MLX installation..."
$PYTHON -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}'); print(f'Device: {mx.default_device()}')"

echo ""
echo "[4/5] Testing MLX Metal backend..."
$PYTHON -c "
import mlx.core as mx
mx.set_default_device(mx.gpu)
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a @ b
mx.eval(c)
print(f'Metal compute test passed: dot product = {c.item()}')
"

echo ""
echo "[5/5] Quick sanity check..."
$PYTHON -c "
import mlx.nn as nn
import mlx.core as mx
model = nn.TransformerEncoder(2, 64, 4)
mx.eval(model.parameters())
print('TransformerEncoder initialized successfully')
"

echo ""
echo "=================================================="
echo "Setup complete! Next steps:"
echo "=================================================="
echo ""
echo "1. Run baseline test:"
echo "   ./venv/bin/python scripts/test_baseline.py"
echo ""
echo "2. Verify 150M model fits in memory:"
echo "   ./venv/bin/python scripts/verify_150m.py"
echo ""
echo "3. (Optional) Monitor resources in another terminal:"
echo "   ./venv/bin/python scripts/monitor_resources.py"
echo ""
echo "4. For detailed GPU monitoring (requires sudo):"
echo "   ./venv/bin/pip install asitop"
echo "   sudo asitop"
echo ""
