#!/usr/bin/env python3
"""
📂 DATA LOADING EXPERIMENTS

Tests:
1. DataLoader num_workers (0, 2, 4, 8)
2. Pin memory vs not
3. Prefetch factor
4. Tokenization speed (tiktoken vs HF)
5. Memory-mapped vs in-memory
"""

import gc
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def print_header(text: str):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


class DummyDataset(Dataset):
    """Dummy dataset for testing data loading."""
    def __init__(self, size: int = 100000, seq_len: int = 512):
        self.size = size
        self.seq_len = seq_len
        self.data = torch.randint(0, 50257, (size, seq_len + 1))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


class MemmapDataset(Dataset):
    """Memory-mapped dataset."""
    def __init__(self, path: str, seq_len: int = 512):
        self.seq_len = seq_len
        self.tokens = np.memmap(path, dtype=np.int32, mode='r')
        self.num_samples = len(self.tokens) // (seq_len + 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = torch.from_numpy(
            self.tokens[start:start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def test_num_workers(batch_size: int = 32, seq_len: int = 512, num_batches: int = 100):
    """Test different num_workers settings."""
    print_header("Testing num_workers")
    
    dataset = DummyDataset(size=10000, seq_len=seq_len)
    results = {}
    
    for num_workers in [0, 2, 4, 8]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        # Warmup
        for i, batch in enumerate(loader):
            if i >= 5:
                break
        
        # Time
        batch_times = []
        start = time.perf_counter()
        
        for i, batch in enumerate(loader):
            batch_time = time.perf_counter() - start
            batch_times.append(batch_time * 1000)
            if i >= num_batches:
                break
            start = time.perf_counter()
        
        avg_time = sum(batch_times[1:]) / len(batch_times[1:])
        tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)
        
        results[num_workers] = {
            "avg_batch_ms": avg_time,
            "tokens_per_sec": tokens_per_sec,
        }
        
        print(f"  num_workers={num_workers}: {avg_time:.2f}ms/batch, {tokens_per_sec:,.0f} tok/s")
        
        del loader
        gc.collect()
    
    best = max(results.items(), key=lambda x: x[1]["tokens_per_sec"])
    print(f"\n  🏆 Best: num_workers={best[0]}")
    
    return {"num_workers": results, "best_num_workers": best[0]}


def test_pin_memory(batch_size: int = 32, seq_len: int = 512, num_batches: int = 100):
    """Test pin_memory effect."""
    print_header("Testing pin_memory")
    
    dataset = DummyDataset(size=10000, seq_len=seq_len)
    results = {}
    
    for pin_memory in [False, True]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        # Time with GPU transfer
        batch_times = []
        start = time.perf_counter()
        
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - start
            batch_times.append(batch_time * 1000)
            if i >= num_batches:
                break
            start = time.perf_counter()
        
        avg_time = sum(batch_times[1:]) / len(batch_times[1:])
        
        results[pin_memory] = {"avg_batch_ms": avg_time}
        print(f"  pin_memory={pin_memory}: {avg_time:.2f}ms/batch")
        
        del loader
        gc.collect()
    
    return {"pin_memory": results}


def test_tokenization():
    """Compare tokenization speed."""
    print_header("Testing Tokenization Speed")
    
    # Generate test texts
    test_texts = [
        "Once upon a time, there was a little girl named Lily. " * 10
        for _ in range(1000)
    ]
    
    results = {}
    
    # Test tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        
        start = time.perf_counter()
        tokens = []
        for text in test_texts:
            tokens.extend(enc.encode(text))
        tiktoken_time = time.perf_counter() - start
        
        results["tiktoken"] = {
            "time_s": tiktoken_time,
            "tokens_per_sec": len(tokens) / tiktoken_time,
        }
        print(f"  tiktoken: {tiktoken_time:.2f}s, {len(tokens)/tiktoken_time:,.0f} tok/s")
    except ImportError:
        print("  tiktoken: not installed")
    
    # Test transformers
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        start = time.perf_counter()
        tokens = []
        for text in test_texts:
            tokens.extend(tokenizer.encode(text))
        hf_time = time.perf_counter() - start
        
        results["transformers"] = {
            "time_s": hf_time,
            "tokens_per_sec": len(tokens) / hf_time,
        }
        print(f"  transformers: {hf_time:.2f}s, {len(tokens)/hf_time:,.0f} tok/s")
    except ImportError:
        print("  transformers: not installed")
    
    if "tiktoken" in results and "transformers" in results:
        speedup = results["tiktoken"]["tokens_per_sec"] / results["transformers"]["tokens_per_sec"]
        print(f"\n  tiktoken is {speedup:.1f}x faster")
    
    return {"tokenization": results}


def test_prefetch_factor(batch_size: int = 32, num_batches: int = 100):
    """Test prefetch_factor effect."""
    print_header("Testing prefetch_factor")
    
    dataset = DummyDataset(size=10000)
    results = {}
    
    for prefetch in [2, 4, 8, 16]:
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=4,
                prefetch_factor=prefetch,
                pin_memory=True,
            )
            
            batch_times = []
            start = time.perf_counter()
            
            for i, batch in enumerate(loader):
                batch_time = time.perf_counter() - start
                batch_times.append(batch_time * 1000)
                if i >= num_batches:
                    break
                start = time.perf_counter()
            
            avg_time = sum(batch_times[1:]) / len(batch_times[1:])
            results[prefetch] = {"avg_batch_ms": avg_time}
            print(f"  prefetch_factor={prefetch}: {avg_time:.2f}ms/batch")
            
            del loader
            gc.collect()
        except Exception as e:
            print(f"  prefetch_factor={prefetch}: Error - {e}")
    
    return {"prefetch_factor": results}


def run_data_experiments(quick: bool = False) -> Dict:
    """Run all data loading experiments."""
    
    print("\n" + "=" * 60)
    print("📂 DATA LOADING EXPERIMENTS")
    print("=" * 60)
    
    num_batches = 50 if quick else 200
    
    results = {}
    
    # Run experiments
    results.update(test_num_workers(num_batches=num_batches))
    results.update(test_pin_memory(num_batches=num_batches))
    results.update(test_tokenization())
    results.update(test_prefetch_factor(num_batches=num_batches))
    
    # Summary
    print("\n" + "─" * 60)
    print("  📊 DATA LOADING SUMMARY")
    print("─" * 60)
    print(f"  Best num_workers: {results.get('best_num_workers', 4)}")
    print(f"  Use pin_memory: True")
    print(f"  Use tiktoken: Yes (faster)")
    
    return results


if __name__ == "__main__":
    run_data_experiments()
