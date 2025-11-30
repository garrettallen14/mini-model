"""
Optimized data loader with:
1. Pre-tokenization and caching to disk
2. Memory-mapped loading
3. Prefetching with threading
4. Efficient sequence packing
"""

import json
import mmap
import os
import struct
import threading
from collections import deque
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

from .tokenizer import Tokenizer, get_tokenizer


class TokenCache:
    """
    Cache tokenized data to disk for fast loading.
    Format: Binary file with packed int32 tokens.
    """
    
    def __init__(self, cache_dir: str = ".cache/tokens"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, dataset_name: str) -> Path:
        return self.cache_dir / f"{dataset_name}.bin"
    
    def get_meta_path(self, dataset_name: str) -> Path:
        return self.cache_dir / f"{dataset_name}.meta.json"
    
    def exists(self, dataset_name: str) -> bool:
        return self.get_cache_path(dataset_name).exists()
    
    def save(self, dataset_name: str, tokens: np.ndarray, metadata: dict = None):
        """Save tokenized data to binary cache."""
        cache_path = self.get_cache_path(dataset_name)
        
        # Save as raw binary (int32)
        tokens = tokens.astype(np.int32)
        tokens.tofile(cache_path)
        
        # Save metadata
        meta = {
            "num_tokens": len(tokens),
            "dtype": "int32",
            "dataset_name": dataset_name,
            **(metadata or {}),
        }
        with open(self.get_meta_path(dataset_name), "w") as f:
            json.dump(meta, f)
        
        print(f"  Cached {len(tokens):,} tokens to {cache_path}")
    
    def load(self, dataset_name: str) -> np.ndarray:
        """Load tokenized data from cache."""
        cache_path = self.get_cache_path(dataset_name)
        return np.fromfile(cache_path, dtype=np.int32)
    
    def load_mmap(self, dataset_name: str) -> np.ndarray:
        """Memory-map the cache file for zero-copy loading."""
        cache_path = self.get_cache_path(dataset_name)
        return np.memmap(cache_path, dtype=np.int32, mode='r')
    
    def get_metadata(self, dataset_name: str) -> dict:
        with open(self.get_meta_path(dataset_name)) as f:
            return json.load(f)


def tokenize_tinystories(
    tokenizer: Tokenizer,
    cache: TokenCache,
    force_rebuild: bool = False,
) -> np.ndarray:
    """Tokenize TinyStories and cache to disk."""
    
    dataset_name = "tinystories"
    
    if cache.exists(dataset_name) and not force_rebuild:
        print(f"Loading cached TinyStories...")
        meta = cache.get_metadata(dataset_name)
        print(f"  {meta['num_tokens']:,} tokens")
        return cache.load_mmap(dataset_name)
    
    print("Tokenizing TinyStories (one-time operation)...")
    
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    
    all_tokens = []
    eos = tokenizer.eos_token_id
    
    for i, example in enumerate(ds):
        text = example["text"]
        if text:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(eos)
        
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i+1:,} / {len(ds):,} stories...")
    
    tokens_array = np.array(all_tokens, dtype=np.int32)
    
    cache.save(dataset_name, tokens_array, {
        "num_stories": len(ds),
        "vocab_size": tokenizer.vocab_size,
    })
    
    return cache.load_mmap(dataset_name)


class PrefetchBuffer:
    """
    Threaded prefetch buffer for batches.
    Prepares batches in background while training.
    """
    
    def __init__(self, data_source, buffer_size: int = 8):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def _fill_buffer(self):
        """Background thread that fills the buffer."""
        for batch in self.data_source:
            with self.condition:
                while len(self.buffer) >= self.buffer_size:
                    if self.stop_event.is_set():
                        return
                    self.condition.wait(timeout=0.1)
                
                self.buffer.append(batch)
                self.condition.notify_all()
            
            if self.stop_event.is_set():
                return
    
    def start(self):
        """Start the prefetch thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def get_batch(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """Get the next batch from the buffer."""
        with self.condition:
            while len(self.buffer) == 0:
                if self.stop_event.is_set():
                    return None
                if not self.condition.wait(timeout=timeout):
                    return None
            
            batch = self.buffer.popleft()
            self.condition.notify_all()
            return batch
    
    def __iter__(self):
        self.start()
        try:
            while True:
                batch = self.get_batch()
                if batch is None:
                    break
                yield batch
        finally:
            self.stop()


class FastDataLoader:
    """
    Optimized data loader with:
    - Pre-tokenized disk cache
    - Memory-mapped loading
    - Efficient random access
    - Optional prefetching
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 8,
        cache_dir: str = ".cache/tokens",
        tokenizer: Optional[Tokenizer] = None,
        shuffle: bool = True,
        prefetch: bool = True,
        prefetch_buffer: int = 8,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.prefetch_buffer = prefetch_buffer
        
        self.tokenizer = tokenizer or get_tokenizer()
        self.cache = TokenCache(cache_dir)
        
        self.tokens = None
        self.total_tokens = 0
        self.current_position = 0
    
    def load_tinystories(self, force_rebuild: bool = False):
        """Load TinyStories (tokenize and cache if needed)."""
        self.tokens = tokenize_tinystories(
            self.tokenizer, self.cache, force_rebuild
        )
        self.total_tokens = len(self.tokens)
        print(f"Loaded {self.total_tokens:,} tokens ({self.total_tokens/1e6:.1f}M)")
    
    def _get_num_samples(self) -> int:
        """Number of complete samples in the dataset."""
        window = self.seq_len + 1
        return (self.total_tokens - 1) // window
    
    def _get_sample(self, idx: int) -> np.ndarray:
        """Get a single sample by index."""
        window = self.seq_len + 1
        start = idx * window
        return self.tokens[start:start + window]
    
    def _generate_batches(self) -> Generator[np.ndarray, None, None]:
        """Generate batches (core logic)."""
        num_samples = self._get_num_samples()
        
        if self.shuffle:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)
        
        batch = []
        for idx in indices:
            sample = self._get_sample(idx)
            if len(sample) == self.seq_len + 1:
                batch.append(sample)
            
            if len(batch) >= self.batch_size:
                yield np.array(batch, dtype=np.int32)
                batch = []
                self.current_position += self.batch_size * self.seq_len
        
        # Don't yield incomplete final batch
    
    def iterate_batches(self) -> Generator[np.ndarray, None, None]:
        """Main iteration method."""
        if self.tokens is None:
            self.load_tinystories()
        
        self.current_position = 0
        
        if self.prefetch:
            buffer = PrefetchBuffer(
                self._generate_batches(), 
                buffer_size=self.prefetch_buffer
            )
            yield from buffer
        else:
            yield from self._generate_batches()
    
    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "current_position": self.current_position,
            "progress_pct": self.current_position / self.total_tokens * 100,
        }


class InMemoryLoader:
    """
    Fastest possible loader: everything in RAM.
    Best for TinyStories (~500MB tokenized).
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 8,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer or get_tokenizer()
        
        self.tokens = None
        self.samples = None
        self.total_tokens = 0
        self.current_position = 0
    
    def load_tinystories(self):
        """Load and pre-chunk all data into memory."""
        cache = TokenCache()
        
        # Get tokens (from cache or tokenize)
        tokens = tokenize_tinystories(self.tokenizer, cache)
        
        # Pre-chunk into samples
        window = self.seq_len + 1
        num_samples = len(tokens) // window
        
        # Reshape into samples
        usable_tokens = num_samples * window
        self.samples = tokens[:usable_tokens].reshape(num_samples, window)
        self.total_tokens = usable_tokens
        
        print(f"Pre-chunked into {num_samples:,} samples")
    
    def iterate_batches(self) -> Generator[np.ndarray, None, None]:
        """Generate batches with minimal overhead."""
        if self.samples is None:
            self.load_tinystories()
        
        num_samples = len(self.samples)
        indices = np.random.permutation(num_samples)
        
        self.current_position = 0
        
        for i in range(0, num_samples - self.batch_size + 1, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = self.samples[batch_indices]
            
            self.current_position += self.batch_size * self.seq_len
            yield np.array(batch, dtype=np.int32)


if __name__ == "__main__":
    import time
    
    print("="*60)
    print("Data Loader Comparison")
    print("="*60)
    
    # Test InMemoryLoader
    print("\n[1] InMemoryLoader Test")
    print("-"*40)
    
    loader = InMemoryLoader(seq_len=512, batch_size=8)
    
    # First load (includes tokenization if needed)
    start = time.perf_counter()
    loader.load_tinystories()
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.2f}s")
    
    # Batch iteration
    batch_times = []
    start = time.perf_counter()
    for i, batch in enumerate(loader.iterate_batches()):
        batch_time = time.perf_counter() - start
        batch_times.append(batch_time * 1000)
        
        if i >= 100:
            break
        start = time.perf_counter()
    
    print(f"Batch shape: {batch.shape}")
    print(f"Avg batch time: {sum(batch_times)/len(batch_times):.2f}ms")
    print(f"Min/Max: {min(batch_times):.2f} / {max(batch_times):.2f}ms")
    
    # Test FastDataLoader with prefetch
    print("\n[2] FastDataLoader (with prefetch) Test")
    print("-"*40)
    
    loader2 = FastDataLoader(seq_len=512, batch_size=8, prefetch=True)
    loader2.load_tinystories()
    
    batch_times = []
    start = time.perf_counter()
    for i, batch in enumerate(loader2.iterate_batches()):
        batch_time = time.perf_counter() - start
        batch_times.append(batch_time * 1000)
        
        if i >= 100:
            break
        start = time.perf_counter()
    
    print(f"Avg batch time: {sum(batch_times)/len(batch_times):.2f}ms")
    print(f"Min/Max: {min(batch_times):.2f} / {max(batch_times):.2f}ms")
    
    print("\n" + "="*60)
