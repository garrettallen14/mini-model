#!/usr/bin/env python3
# Fix RunPod environment issues
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
🎓 3B CURRICULUM TRAINING (CUDA)

Full curriculum training based on datas.md:
- Phase 1 (0-1B): TinyStories + Cosmopedia (foundation)
- Phase 2 (1B-2B): Add code + math
- Phase 3 (2B-3B): Full mix + harder content

Usage:
    # Full 3B curriculum
    python train_curriculum_cuda.py --size 150M
    
    # Continue from TinyStories checkpoint
    python train_curriculum_cuda.py --size 150M --resume checkpoints/150M_*/final.pt
    
    # Quick test (100M tokens)
    python train_curriculum_cuda.py --size 150M --max-tokens 100000000
"""

import argparse
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import random

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Optional wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# Model (same as train_cuda.py)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, offset: int = 0):
        seq_len = x.shape[1]
        t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, None, :]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos = torch.cos(rope_cos)
        sin = torch.sin(rope_sin)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, head_dim)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int = 50257,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_dim: int = 3072,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(head_dim, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim, mlp_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.tok_emb.weight = self.output.weight

    def forward(self, x):
        h = self.tok_emb(x)
        rope_emb = self.rope(h)
        rope_cos, rope_sin = rope_emb, rope_emb
        
        for layer in self.layers:
            h = layer(h, rope_cos, rope_sin)
        
        return self.output(self.norm(h))


MODEL_CONFIGS = {
    "10M": {"dim": 256, "num_layers": 6, "num_heads": 4, "head_dim": 64, "mlp_dim": 1024},
    "50M": {"dim": 512, "num_layers": 8, "num_heads": 8, "head_dim": 64, "mlp_dim": 2048},
    "150M": {"dim": 768, "num_layers": 12, "num_heads": 12, "head_dim": 64, "mlp_dim": 3072},
    "350M": {"dim": 1024, "num_layers": 24, "num_heads": 16, "head_dim": 64, "mlp_dim": 4096},
}


def create_model(size: str = "150M") -> MiniModel:
    return MiniModel(**MODEL_CONFIGS[size])


# =============================================================================
# Curriculum Dataset
# =============================================================================

CURRICULUM_DATASETS = {
    "tinystories": {
        "hf_path": "roneneldan/TinyStories",
        "text_col": "text",
        "split": "train",
        "streaming": False,
        "tokens_estimate": 476_000_000,
    },
    "cosmopedia_stories": {
        "hf_path": "HuggingFaceTB/cosmopedia",
        "name": "stories",  # Config name required
        "text_col": "text", 
        "split": "train",
        "streaming": True,
        "tokens_estimate": 150_000_000,
    },
    "cosmopedia_wikihow": {
        "hf_path": "HuggingFaceTB/cosmopedia",
        "name": "wikihow",
        "text_col": "text", 
        "split": "train",
        "streaming": True,
        "tokens_estimate": 100_000_000,
    },
    "openwebmath": {
        "hf_path": "open-web-math/open-web-math",
        "text_col": "text",
        "split": "train", 
        "streaming": True,
        "tokens_estimate": 800_000_000,
    },
    "python_code": {
        "hf_path": "flytech/python-codes-25k",
        "text_col": "code",
        "split": "train",
        "streaming": False,
        "tokens_estimate": 50_000_000,
    },
    "metamathqa": {
        "hf_path": "meta-math/MetaMathQA",
        "text_col": "query",  # Will combine query + response
        "split": "train",
        "streaming": False,
        "tokens_estimate": 250_000_000,
    },
}

# Curriculum phases
CURRICULUM_PHASES = {
    # tokens_seen: {dataset: weight}
    0: {"tinystories": 0.6, "cosmopedia_stories": 0.4},                            # Phase 1: Foundation
    1_000_000_000: {"tinystories": 0.2, "cosmopedia_stories": 0.2, "openwebmath": 0.3, "python_code": 0.3},  # Phase 2: Add code/math
    2_000_000_000: {"cosmopedia_wikihow": 0.2, "openwebmath": 0.3, "python_code": 0.3, "metamathqa": 0.2},   # Phase 3: Full mix
}


def get_curriculum_weights(tokens_seen: int) -> Dict[str, float]:
    """Get dataset weights based on tokens seen."""
    phase_tokens = sorted(CURRICULUM_PHASES.keys(), reverse=True)
    for threshold in phase_tokens:
        if tokens_seen >= threshold:
            return CURRICULUM_PHASES[threshold]
    return CURRICULUM_PHASES[0]


def validate_datasets():
    """Validate all curriculum datasets are accessible."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("🔍 VALIDATING DATASETS")
    print("=" * 60)
    
    # Get all unique datasets from all phases
    all_datasets = set()
    for phase_weights in CURRICULUM_PHASES.values():
        all_datasets.update(phase_weights.keys())
    
    results = {}
    for name in sorted(all_datasets):
        config = CURRICULUM_DATASETS[name]
        print(f"\n  Checking {name}...")
        
        kwargs = {
            "path": config["hf_path"],
            "split": config["split"],
            "streaming": True,  # Always stream for validation
        }
        if "data_dir" in config:
            kwargs["data_dir"] = config["data_dir"]
        if "name" in config:
            kwargs["name"] = config["name"]
        
        try:
            ds = load_dataset(**kwargs)
            # Try to read one example
            example = next(iter(ds))
            text_col = config["text_col"]
            
            if text_col in example:
                text = example[text_col][:100] if example[text_col] else "(empty)"
                print(f"    ✓ OK - Sample: {text}...")
                results[name] = True
            else:
                print(f"    ✗ Missing column '{text_col}'")
                results[name] = False
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:80]}")
            results[name] = False
    
    print("\n" + "-" * 60)
    ok = all(results.values())
    if ok:
        print("✓ All datasets validated!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"✗ Failed datasets: {failed}")
    print("-" * 60 + "\n")
    
    return ok


class CurriculumDataset(IterableDataset):
    """Streaming curriculum dataset with dynamic mixing."""
    
    def __init__(
        self, 
        seq_len: int = 512,
        seed: int = 42,
        start_tokens: int = 0,
    ):
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.seed = seed
        self.tokens_seen = start_tokens
        self.datasets = {}
        self.iterators = {}
        
    def _load_dataset(self, name: str):
        """Lazy load a dataset."""
        if name in self.datasets:
            return
        
        from datasets import load_dataset
        
        config = CURRICULUM_DATASETS[name]
        kwargs = {
            "path": config["hf_path"],
            "split": config["split"],
            "streaming": config["streaming"],
        }
        if "data_dir" in config:
            kwargs["data_dir"] = config["data_dir"]
        if "name" in config:
            kwargs["name"] = config["name"]
        
        print(f"  Loading {name}...")
        self.datasets[name] = load_dataset(**kwargs)
        self.iterators[name] = iter(self.datasets[name])
    
    def _get_text(self, name: str) -> Optional[str]:
        """Get next text from a dataset."""
        self._load_dataset(name)
        config = CURRICULUM_DATASETS[name]
        
        try:
            example = next(self.iterators[name])
            
            # Handle MetaMathQA specially (combine query + response)
            if name == "metamathqa":
                text = example.get("query", "") + "\n\n" + example.get("response", "")
            else:
                text = example.get(config["text_col"], "")
            
            return text if text else None
            
        except StopIteration:
            # Reset iterator
            self.iterators[name] = iter(self.datasets[name])
            return self._get_text(name)
    
    def _sample_dataset(self) -> str:
        """Sample a dataset based on current curriculum weights."""
        weights = get_curriculum_weights(self.tokens_seen)
        datasets = list(weights.keys())
        probs = list(weights.values())
        
        # Weighted random choice
        return random.choices(datasets, weights=probs, k=1)[0]
    
    def __iter__(self):
        buffer = []
        random.seed(self.seed)
        
        while True:
            # Fill buffer to seq_len + 1
            while len(buffer) < self.seq_len + 1:
                dataset_name = self._sample_dataset()
                text = self._get_text(dataset_name)
                
                if text:
                    tokens = self.enc.encode(text)
                    buffer.extend(tokens)
                    buffer.append(self.enc.eot_token)
            
            # Yield chunk
            chunk = buffer[:self.seq_len + 1]
            buffer = buffer[self.seq_len:]
            
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            
            self.tokens_seen += self.seq_len
            yield x, y


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset from file (for TinyStories warmstart)."""
    
    def __init__(self, tokens_path: str, seq_len: int):
        self.seq_len = seq_len
        tokens_np = np.array(np.memmap(tokens_path, dtype='int32', mode='r'))
        self.tokens = torch.from_numpy(tokens_np)
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1].long()
        return chunk[:-1], chunk[1:]


# =============================================================================
# Training
# =============================================================================

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def generate_sample(model, prompt: str = "Once upon a time", max_tokens: int = 50, temperature: float = 0.8):
    """Generate a sample from the model."""
    enc = tiktoken.get_encoding("gpt2")
    
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], device=next(model.parameters()).device)
    
    for _ in range(max_tokens):
        with autocast('cuda'):
            logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        if next_token.item() == enc.eot_token:
            break
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        
        if x.shape[1] > 512:
            x = x[:, -512:]
    
    model.train()
    return enc.decode(x[0].tolist())


def train_curriculum(
    model_size: str = "150M",
    batch_size: int = 32,
    seq_len: int = 512,
    learning_rate: float = 5e-4,
    min_lr: float = 1e-6,
    warmup_steps: int = 1000,
    max_tokens: int = 3_000_000_000,
    log_interval: int = 100,
    eval_interval: int = 5000,
    save_interval: int = 50_000,
    output_dir: str = "checkpoints",
    resume: Optional[str] = None,
    use_wandb: bool = False,
    compile_model: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output dir
    run_name = f"curriculum_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Model
    print(f"\nCreating {model_size} model...")
    model = create_model(model_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load checkpoint if resuming
    start_step = 0
    start_tokens = 0
    
    if resume:
        print(f"\nLoading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_tokens = checkpoint.get('tokens_seen', 0)
        print(f"Resuming from {start_tokens:,} tokens")
    
    # Compile
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Dataset
    print("\nSetting up curriculum dataset...")
    print(f"Current phase: {get_curriculum_weights(start_tokens)}")
    
    dataset = CurriculumDataset(seq_len=seq_len, start_tokens=start_tokens)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = GradScaler('cuda')
    
    # Training stats
    tokens_per_step = batch_size * seq_len
    max_steps = max_tokens // tokens_per_step
    
    print(f"\n{'='*60}")
    print(f"🎓 CURRICULUM TRAINING")
    print(f"{'='*60}")
    print(f"  Model: {model_size} ({num_params/1e6:.1f}M params)")
    print(f"  Target: {max_tokens/1e9:.1f}B tokens ({max_steps:,} steps)")
    print(f"  Batch: {batch_size} x {seq_len} = {tokens_per_step:,} tokens/step")
    print(f"  Starting from: {start_tokens/1e6:.1f}M tokens")
    print(f"{'='*60}")
    
    # Save config
    config = {
        "model_size": model_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "max_tokens": max_tokens,
        "curriculum_phases": {str(k): v for k, v in CURRICULUM_PHASES.items()},
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # W&B
    if use_wandb and HAS_WANDB:
        wandb.init(project="mini-model-curriculum", name=run_name, config=config)
    
    # Logging
    log_file = open(output_path / "training.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    # Training loop
    model.train()
    tokens_seen = start_tokens
    step = 0
    start_time = time.time()
    loss_accum = 0.0
    loss_count = 0
    
    pbar = tqdm(total=max_steps, desc="Training", initial=start_step)
    
    for x, y in dataloader:
        if tokens_seen >= max_tokens:
            break
        
        x, y = x.to(device), y.to(device)
        
        # LR schedule
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Forward/backward
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Track
        loss_accum += loss.item()
        loss_count += 1
        tokens_seen += tokens_per_step
        dataset.tokens_seen = tokens_seen  # Update curriculum
        step += 1
        pbar.update(1)
        
        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (tokens_seen - start_tokens) / elapsed
            eta_sec = (max_tokens - tokens_seen) / tokens_per_sec
            eta_str = f"{eta_sec/3600:.1f}h"
            
            avg_loss = loss_accum / loss_count
            ppl = math.exp(min(avg_loss, 10))
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            
            # Current curriculum phase
            weights = get_curriculum_weights(tokens_seen)
            phase_str = "+".join([f"{k[:3]}" for k in weights.keys()])
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'ppl': f'{ppl:.1f}',
                'tok/s': f'{tokens_per_sec:.0f}',
                'eta': eta_str,
                'phase': phase_str,
            })
            
            log_msg = (
                f"step={step:>6} | tokens={tokens_seen/1e9:.2f}B | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                f"lr={lr:.2e} | grad={grad_norm:.2f} | tok/s={tokens_per_sec:.0f} | "
                f"gpu={gpu_mem:.1f}GB | eta={eta_str} | phase={phase_str}"
            )
            log_file.write(log_msg + "\n")
            log_file.flush()
            
            if use_wandb and HAS_WANDB:
                wandb.log({
                    "loss": avg_loss,
                    "ppl": ppl,
                    "lr": lr,
                    "grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_B": tokens_seen / 1e9,
                    "gpu_memory_gb": gpu_mem,
                }, step=step)
            
            loss_accum = 0.0
            loss_count = 0
        
        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = output_path / f"step_{step}_tokens_{tokens_seen//1e6:.0f}M.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokens_seen': tokens_seen,
                'config': config,
            }, ckpt_path)
            log(f"\n  💾 Checkpoint: {ckpt_path}")
        
        # Generate sample
        if step % eval_interval == 0:
            try:
                prompts = [
                    "Once upon a time",
                    "def fibonacci(n):",
                    "The derivative of x^2 is",
                ]
                log(f"\n  📝 Samples (step {step}, {tokens_seen/1e9:.2f}B tokens):")
                for prompt in prompts:
                    sample = generate_sample(model, prompt, max_tokens=40)
                    log(f"    {sample[:150]}...")
                log("")
            except Exception as e:
                log(f"  Sample generation failed: {e}")
    
    pbar.close()
    
    # Final save
    final_path = output_path / "final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'tokens_seen': tokens_seen,
        'config': config,
    }, final_path)
    
    # Summary
    total_time = time.time() - start_time
    log(f"\n{'='*60}")
    log("🎓 Curriculum Training Complete!")
    log(f"{'='*60}")
    log(f"  Time: {total_time/3600:.2f} hours")
    log(f"  Tokens: {tokens_seen:,} ({tokens_seen/1e9:.2f}B)")
    log(f"  Tokens/sec: {(tokens_seen-start_tokens)/total_time:,.0f}")
    log(f"  Final loss: {loss.item():.4f}")
    log(f"  Output: {output_path}")
    
    log_file.close()
    
    if use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="3B Curriculum Training on CUDA")
    
    # Model
    parser.add_argument("--size", default="150M", choices=["10M", "50M", "150M", "350M"])
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    
    # Curriculum
    parser.add_argument("--max-tokens", type=int, default=3_000_000_000, help="Total tokens to train on")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Performance
    parser.add_argument("--no-compile", action="store_true")
    
    # Validation
    parser.add_argument("--validate", action="store_true", help="Validate all datasets before training")
    parser.add_argument("--validate-only", action="store_true", help="Only validate datasets, don't train")
    
    args = parser.parse_args()
    
    # Validate datasets first
    if args.validate or args.validate_only:
        ok = validate_datasets()
        if args.validate_only:
            return
        if not ok:
            print("Dataset validation failed. Fix issues or run without --validate to skip.")
            return
    
    train_curriculum(
        model_size=args.size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        max_tokens=args.max_tokens,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        resume=args.resume,
        use_wandb=args.wandb,
        compile_model=not args.no_compile,
    )


if __name__ == "__main__":
    main()
