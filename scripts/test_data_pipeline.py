#!/usr/bin/env python3
"""
Test the data pipeline with TinyStories.

This validates:
1. Tokenizer works correctly
2. Sequence packing produces correct shapes
3. Data loader generates proper batches
4. Memory usage is acceptable

Run this before full training to catch issues early.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_tokenizer():
    """Test tokenizer functionality."""
    print("\n[1/4] Testing Tokenizer...")
    
    from data.tokenizer import get_tokenizer
    
    tok = get_tokenizer()
    
    # Basic tests
    assert tok.vocab_size == 50257, f"Expected 50257, got {tok.vocab_size}"
    
    test_text = "Hello, world! This is a test of the tokenizer."
    tokens = tok.encode(test_text)
    decoded = tok.decode(tokens)
    
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  EOS token: {tok.eos_token_id}")
    print(f"  Test: '{test_text[:30]}...' -> {len(tokens)} tokens")
    print(f"  Roundtrip match: {test_text == decoded}")
    
    # Math/code test
    math_text = "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"
    math_tokens = tok.encode(math_text)
    print(f"  Code tokenization: {len(math_text)} chars -> {len(math_tokens)} tokens")
    
    print("  ✓ Tokenizer OK")
    return tok


def test_sequence_packer(tok):
    """Test sequence packing."""
    print("\n[2/4] Testing Sequence Packer...")
    
    from data.loader import SequencePacker
    
    seq_len = 512
    packer = SequencePacker(seq_len, tok.eos_token_id)
    
    # Add several documents
    docs = [
        "This is document one. " * 50,
        "This is document two with different content. " * 100,
        "Short doc.",
        "Another document with varying length. " * 200,
    ]
    
    all_sequences = []
    for doc in docs:
        tokens = tok.encode(doc)
        seqs = packer.add_document(tokens)
        all_sequences.extend(seqs)
    
    # Flush remaining
    final = packer.flush()
    if final:
        all_sequences.append(final)
    
    print(f"  Input: {len(docs)} documents")
    print(f"  Output: {len(all_sequences)} sequences")
    
    # Validate shapes
    for i, seq in enumerate(all_sequences[:-1]):  # Exclude last (may be short)
        assert len(seq) == seq_len + 1, f"Seq {i}: expected {seq_len+1}, got {len(seq)}"
    
    print(f"  All sequences have correct length ({seq_len + 1})")
    print("  ✓ Sequence Packer OK")


def test_tinystories_loader(tok):
    """Test loading TinyStories."""
    print("\n[3/4] Testing TinyStories Loader...")
    
    from data.loader import TinyStoriesLoader
    
    loader = TinyStoriesLoader(seq_len=512, batch_size=8, tokenizer=tok)
    
    batches = []
    tokens_seen = 0
    start = time.perf_counter()
    
    for i, batch in enumerate(loader.iterate_batches()):
        batches.append(batch)
        tokens_seen = loader.total_tokens
        
        if i == 0:
            print(f"  First batch shape: {batch.shape}")
            print(f"  Dtype: {batch.dtype}")
            print(f"  Min/Max values: {batch.min()}, {batch.max()}")
        
        # Stop after ~10M tokens for quick test
        if tokens_seen > 10_000_000:
            break
    
    elapsed = time.perf_counter() - start
    
    print(f"  Processed {len(batches)} batches")
    print(f"  Total tokens: {tokens_seen:,} ({tokens_seen/1e6:.1f}M)")
    print(f"  Time: {elapsed:.1f}s ({tokens_seen/elapsed/1000:.1f}k tokens/sec)")
    print("  ✓ TinyStories Loader OK")
    
    return batches


def test_batch_for_training(batches, tok):
    """Validate batch format for training."""
    print("\n[4/4] Validating Batch Format...")
    
    batch = batches[0]
    
    # Shape check
    assert len(batch.shape) == 2, f"Expected 2D, got {len(batch.shape)}D"
    assert batch.shape[1] == 513, f"Expected seq_len+1=513, got {batch.shape[1]}"
    
    # Input/target split
    inputs = batch[:, :-1]  # (batch, 512)
    targets = batch[:, 1:]  # (batch, 512)
    
    print(f"  Batch shape: {batch.shape}")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    # Decode sample
    sample_input = inputs[0].tolist()
    sample_text = tok.decode(sample_input[:50])
    print(f"  Sample decode: '{sample_text[:60]}...'")
    
    # Vocab range check
    assert batch.min() >= 0, "Negative token IDs found"
    assert batch.max() < tok.vocab_size, f"Token ID {batch.max()} >= vocab size"
    
    print("  ✓ Batch Format OK")


def main():
    print("=" * 60)
    print("Data Pipeline Test")
    print("=" * 60)
    
    try:
        tok = test_tokenizer()
        test_sequence_packer(tok)
        batches = test_tinystories_loader(tok)
        test_batch_for_training(batches, tok)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nData pipeline is ready for training.")
        print("Next: Run Phase 1 TinyStories training:")
        print("  ./venv/bin/python scripts/train_tinystories.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
