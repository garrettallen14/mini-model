#!/usr/bin/env python3
"""
Phase 1: Baseline Test Script

This script validates that the MLX training pipeline works correctly.
Runs a quick 1000-step training on PTB with a small model.

Expected outcomes:
1. MLX imports and Metal backend works
2. Model can be initialized and evaluated
3. Training loop runs without OOM
4. Loss decreases over time
"""

import math
import sys
import time
from functools import partial
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "mlx-examples" / "transformer_lm"))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Import from mlx-examples
import datasets as mlx_datasets


class TransformerLM(nn.Module):
    """Simple transformer LM matching mlx-examples structure."""
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def count_params(model) -> int:
    """Count trainable parameters."""
    return sum(x.size for _, x in tree_flatten(model.parameters()))


def to_samples(context_size, dataset):
    """Convert dataset to fixed-size samples."""
    window_size = context_size + 1
    samples = dataset.size // window_size
    dataset = dataset[: samples * window_size]
    return mx.array(dataset.reshape(samples, -1))


def iterate_batches(batch_size, context_size, dataset):
    """Generate batches from dataset."""
    inputs = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            perm = mx.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main():
    print("=" * 60)
    print("Phase 1: MLX Baseline Validation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Verify MLX and Metal
    # -------------------------------------------------------------------------
    print("\n[1/5] Checking MLX installation...")
    print(f"  MLX version: {mx.__version__}")
    print(f"  Default device: {mx.default_device()}")
    
    # Force GPU
    mx.set_default_device(mx.gpu)
    print(f"  Set to GPU: {mx.default_device()}")
    
    # Quick Metal test
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    mx.eval(c)
    print(f"  Metal compute test: {c.tolist()} ✓")
    
    # -------------------------------------------------------------------------
    # Step 2: Load dataset
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading PTB dataset...")
    vocab, train, valid, test = mlx_datasets.ptb()
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Train tokens: {len(train):,}")
    print(f"  Valid tokens: {len(valid):,}")
    print(f"  Test tokens: {len(test):,}")
    
    # -------------------------------------------------------------------------
    # Step 3: Initialize model
    # -------------------------------------------------------------------------
    print("\n[3/5] Initializing model...")
    
    # Small config for quick test
    config = {
        "vocab_size": len(vocab),
        "num_layers": 4,
        "dims": 256,
        "num_heads": 4,
        "checkpoint": False,
    }
    
    model = TransformerLM(**config)
    mx.eval(model.parameters())
    
    num_params = count_params(model)
    print(f"  Config: {config['num_layers']}L, {config['dims']}D, {config['num_heads']}H")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # -------------------------------------------------------------------------
    # Step 4: Training loop test
    # -------------------------------------------------------------------------
    print("\n[4/5] Running training loop (1000 steps)...")
    
    # Hyperparameters
    batch_size = 4
    context_size = 256
    num_iters = 1000
    learning_rate = 3e-4
    warmup_steps = 100
    log_interval = 100
    
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    
    def loss_fn(model, inputs):
        x, y = inputs[..., :-1], inputs[..., 1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")
    
    train_iter = iterate_batches(batch_size, context_size, train)
    
    # Training state for mx.compile
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs):
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, inputs)
        optimizer.update(model, grads)
        return loss
    
    losses = []
    start_time = time.perf_counter()
    first_loss = None
    
    for it in range(num_iters):
        inputs = next(train_iter)
        
        # Linear warmup
        lr = min(1.0, (it + 1) / warmup_steps) * learning_rate
        optimizer.learning_rate = lr
        
        loss = step(inputs)
        mx.eval(loss)  # Eval the loss, state handled by compile
        
        loss_val = loss.item()
        losses.append(loss_val)
        
        if first_loss is None:
            first_loss = loss_val
        
        if (it + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            elapsed = time.perf_counter() - start_time
            steps_per_sec = (it + 1) / elapsed
            print(f"  Step {it+1:4d} | Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f} | {steps_per_sec:.1f} it/s")
    
    final_loss = sum(losses[-100:]) / 100
    total_time = time.perf_counter() - start_time
    
    # -------------------------------------------------------------------------
    # Step 5: Validation
    # -------------------------------------------------------------------------
    print("\n[5/5] Running validation...")
    
    val_inputs = to_samples(context_size, valid)
    val_loss = 0
    val_batches = 0
    
    for s in range(0, min(val_inputs.shape[0], 100), batch_size):
        batch = val_inputs[s : s + batch_size]
        x, y = batch[..., :-1], batch[..., 1:]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        val_loss += loss.item()
        val_batches += 1
    
    val_loss /= val_batches
    val_ppl = math.exp(val_loss)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE TEST RESULTS")
    print("=" * 60)
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Steps/sec:      {num_iters / total_time:.1f}")
    print(f"  Initial loss:   {first_loss:.4f}")
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Loss reduction: {((first_loss - final_loss) / first_loss * 100):.1f}%")
    print(f"  Val loss:       {val_loss:.4f}")
    print(f"  Val PPL:        {val_ppl:.2f}")
    
    # Sanity checks
    print("\n" + "-" * 60)
    all_passed = True
    
    # Check 1: Loss decreased
    if final_loss < first_loss:
        print("✓ Loss decreased during training")
    else:
        print("✗ Loss did not decrease - possible issue!")
        all_passed = False
    
    # Check 2: Reasonable speed
    if num_iters / total_time > 5:
        print(f"✓ Training speed OK ({num_iters/total_time:.1f} steps/sec)")
    else:
        print(f"⚠ Training slower than expected ({num_iters/total_time:.1f} steps/sec)")
    
    # Check 3: No NaN/Inf
    if math.isfinite(final_loss) and math.isfinite(val_loss):
        print("✓ No NaN/Inf in losses")
    else:
        print("✗ NaN or Inf detected!")
        all_passed = False
    
    # Check 4: Val PPL reasonable (should be < 500 for small model on PTB)
    if val_ppl < 500:
        print(f"✓ Val PPL reasonable ({val_ppl:.1f})")
    else:
        print(f"⚠ Val PPL high ({val_ppl:.1f}) - expected for short training")
    
    print("-" * 60)
    if all_passed:
        print("🎉 PHASE 1 COMPLETE: Pipeline validated successfully!")
        print("\nNext steps:")
        print("  1. Run: pip install datasets huggingface-hub")
        print("  2. Download TinyStories dataset")
        print("  3. Scale to 150M parameter model")
    else:
        print("❌ Some checks failed. Please review the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
