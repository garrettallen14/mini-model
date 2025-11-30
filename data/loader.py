"""
Data loader for mini-model training.

Implements:
- Streaming from HuggingFace datasets
- Curriculum-based mixing (3 phases from datas.md)
- Sequence packing for efficiency
- Memory-efficient processing
"""

import random
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np

from .config import CURRICULUM, DATASETS, get_mix_weights, get_phase_for_token_count
from .tokenizer import Tokenizer, get_tokenizer


class SequencePacker:
    """
    Pack multiple documents into fixed-length sequences.
    Avoids wasting tokens on padding (from datas.md).
    """
    
    def __init__(self, seq_len: int, eos_token_id: int):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.buffer: List[int] = []
    
    def add_document(self, tokens: List[int]) -> List[List[int]]:
        """
        Add tokens from a document, return any complete sequences.
        Adds EOS between documents.
        """
        # Add document tokens with EOS separator
        self.buffer.extend(tokens)
        self.buffer.append(self.eos_token_id)
        
        # Extract complete sequences
        sequences = []
        while len(self.buffer) >= self.seq_len + 1:  # +1 for target
            seq = self.buffer[:self.seq_len + 1]
            sequences.append(seq)
            self.buffer = self.buffer[self.seq_len:]
        
        return sequences
    
    def flush(self) -> Optional[List[int]]:
        """Return any remaining tokens as a final sequence (may be short)."""
        if len(self.buffer) > 1:
            seq = self.buffer[:self.seq_len + 1]
            self.buffer = []
            return seq
        return None


class DatasetStream:
    """Wrapper for streaming a single HuggingFace dataset."""
    
    def __init__(
        self, 
        name: str,
        tokenizer: Tokenizer,
        max_tokens: Optional[int] = None,
    ):
        self.name = name
        self.config = DATASETS[name]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.tokens_yielded = 0
        self._iterator = None
    
    def _load_dataset(self):
        """Lazy load the dataset."""
        from datasets import load_dataset
        
        kwargs = {
            "path": self.config.hf_path,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        if self.config.data_dir:
            kwargs["data_dir"] = self.config.data_dir
        
        return load_dataset(**kwargs)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over tokenized documents."""
        dataset = self._load_dataset()
        text_col = self.config.text_column
        
        for example in dataset:
            # Handle different column structures
            if text_col in example:
                text = example[text_col]
            elif self.name == "metamathqa":
                # MetaMathQA has query + response
                text = f"Question: {example['query']}\nAnswer: {example['response']}"
            else:
                # Fallback: try common column names
                for col in ["text", "content", "code"]:
                    if col in example:
                        text = example[col]
                        break
                else:
                    continue
            
            if not text or not isinstance(text, str):
                continue
            
            tokens = self.tokenizer.encode(text)
            self.tokens_yielded += len(tokens)
            
            # Check token limit
            if self.max_tokens and self.tokens_yielded >= self.max_tokens:
                return
            
            yield tokens


class CurriculumDataLoader:
    """
    Main data loader implementing the 3-phase curriculum from datas.md.
    
    Phase 1 (0-1B): TinyStories + Cosmopedia (foundation)
    Phase 2 (1B-2B): Math + Code ramp up
    Phase 3 (2B-3B): Full mix
    """
    
    def __init__(
        self,
        seq_len: int = 512,
        batch_size: int = 8,
        tokenizer: Optional[Tokenizer] = None,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer or get_tokenizer()
        self.seed = seed
        self.rng = random.Random(seed)
        
        self.packer = SequencePacker(seq_len, self.tokenizer.eos_token_id)
        self.total_tokens = 0
        self.current_phase = "phase_1"
        
        # Dataset streams (lazy loaded)
        self._streams: Dict[str, DatasetStream] = {}
    
    def _get_stream(self, name: str) -> DatasetStream:
        """Get or create a dataset stream."""
        if name not in self._streams:
            max_tokens = DATASETS[name].target_tokens
            self._streams[name] = DatasetStream(name, self.tokenizer, max_tokens)
        return self._streams[name]
    
    def _sample_dataset(self, weights: Dict[str, float]) -> str:
        """Sample a dataset name based on weights."""
        names = list(weights.keys())
        probs = [weights[n] for n in names]
        return self.rng.choices(names, weights=probs, k=1)[0]
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get mixing weights for current phase."""
        phase = get_phase_for_token_count(self.total_tokens)
        if phase != self.current_phase:
            print(f"\n>>> Transitioning to {CURRICULUM[phase]['name']} phase <<<\n")
            self.current_phase = phase
        return get_mix_weights(phase)
    
    def iterate_batches(self) -> Generator[np.ndarray, None, None]:
        """
        Generate batches of packed sequences.
        
        Yields: numpy array of shape (batch_size, seq_len + 1)
                where [:, :-1] is input and [:, 1:] is target
        """
        batch = []
        active_iters: Dict[str, Iterator] = {}
        
        while self.total_tokens < 3_000_000_000:  # 3B total
            weights = self._get_current_weights()
            dataset_name = self._sample_dataset(weights)
            
            # Get or create iterator for this dataset
            if dataset_name not in active_iters:
                try:
                    stream = self._get_stream(dataset_name)
                    active_iters[dataset_name] = iter(stream)
                except Exception as e:
                    print(f"Warning: Could not load {dataset_name}: {e}")
                    continue
            
            # Get next document
            try:
                tokens = next(active_iters[dataset_name])
            except StopIteration:
                # Dataset exhausted, remove from active
                del active_iters[dataset_name]
                if not active_iters:
                    print("All datasets exhausted")
                    break
                continue
            
            # Pack into sequences
            sequences = self.packer.add_document(tokens)
            for seq in sequences:
                batch.append(seq)
                self.total_tokens += len(seq) - 1  # -1 because last token is target only
                
                if len(batch) >= self.batch_size:
                    yield np.array(batch, dtype=np.int32)
                    batch = []
        
        # Yield remaining
        if batch:
            # Pad last batch if needed
            while len(batch) < self.batch_size:
                batch.append(batch[-1])  # Duplicate last
            yield np.array(batch, dtype=np.int32)
    
    def get_stats(self) -> Dict:
        """Get current training statistics."""
        phase_info = CURRICULUM[self.current_phase]
        return {
            "total_tokens": self.total_tokens,
            "total_tokens_b": self.total_tokens / 1e9,
            "current_phase": self.current_phase,
            "phase_name": phase_info["name"],
            "phase_description": phase_info["description"],
            "progress_pct": (self.total_tokens / 3e9) * 100,
        }


class TinyStoriesLoader:
    """
    Simplified loader for TinyStories only.
    Use this to validate the pipeline before full training.
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
        self.packer = SequencePacker(seq_len, self.tokenizer.eos_token_id)
        self.total_tokens = 0
    
    def iterate_batches(self) -> Generator[np.ndarray, None, None]:
        """Generate batches from TinyStories."""
        from datasets import load_dataset
        
        print("Loading TinyStories dataset...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        print(f"Loaded {len(ds)} stories")
        
        batch = []
        
        for i, example in enumerate(ds):
            text = example["text"]
            if not text:
                continue
            
            tokens = self.tokenizer.encode(text)
            sequences = self.packer.add_document(tokens)
            
            for seq in sequences:
                batch.append(seq)
                self.total_tokens += len(seq) - 1
                
                if len(batch) >= self.batch_size:
                    yield np.array(batch, dtype=np.int32)
                    batch = []
            
            if (i + 1) % 50000 == 0:
                print(f"  Processed {i+1} stories, {self.total_tokens/1e6:.1f}M tokens")
        
        if batch:
            while len(batch) < self.batch_size:
                batch.append(batch[-1])
            yield np.array(batch, dtype=np.int32)
        
        print(f"Total: {self.total_tokens/1e6:.1f}M tokens")


if __name__ == "__main__":
    # Test the loader
    print("Testing TinyStories loader...")
    
    loader = TinyStoriesLoader(seq_len=512, batch_size=4)
    
    for i, batch in enumerate(loader.iterate_batches()):
        print(f"Batch {i}: shape={batch.shape}, tokens so far={loader.total_tokens}")
        if i >= 5:
            print("(stopping after 5 batches for test)")
            break
