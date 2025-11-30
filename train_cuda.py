#!/usr/bin/env python3
# Fix hf_transfer issue on RunPod
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

"""
PyTorch/CUDA training script for mini-model.

Optimized for A40 (48GB VRAM):
- Can run 150M model with BS=32-64
- Much faster than M4 MLX
- Supports gradient checkpointing, mixed precision

Usage:
    python train_cuda.py                          # Full TinyStories
    python train_cuda.py --max-tokens 10000000    # Quick test
    python train_cuda.py --wandb                  # With W&B logging
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Optional wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================================================
# Model
# =============================================================================

class RMSNorm(nn.Module):
    """RMSNorm (faster than LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(x, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Flash attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    def __init__(self, dim: int, num_heads: int, ff_mult: float = 3.5, max_seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        hidden_dim = int(dim * ff_mult)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round to 64
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniModel(nn.Module):
    """Mini-Model LLM."""
    def __init__(
        self,
        vocab_size: int = 50257,
        num_layers: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        ff_mult: float = 3.5,
        max_seq_len: int = 2048,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_mult, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.output.weight = self.embed.weight
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.embed(x)
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        
        h = self.norm(h)
        return self.output(h)


def create_model(size: str = "150M") -> MiniModel:
    configs = {
        "10M": {"num_layers": 4, "dim": 256, "num_heads": 4, "use_checkpoint": False},
        "50M": {"num_layers": 8, "dim": 512, "num_heads": 8, "use_checkpoint": False},
        "150M": {"num_layers": 12, "dim": 768, "num_heads": 12, "use_checkpoint": False},
        "350M": {"num_layers": 24, "dim": 1024, "num_heads": 16, "use_checkpoint": True},
    }
    return MiniModel(**configs[size])


# =============================================================================
# Data
# =============================================================================

class TokenizedDataset(Dataset):
    """Pre-tokenized dataset stored as memory-mapped file."""
    
    def __init__(self, tokens_path: str, seq_len: int):
        self.seq_len = seq_len
        self.tokens = torch.from_numpy(
            __import__('numpy').memmap(tokens_path, dtype='int32', mode='r')
        )
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1].long()
        return chunk[:-1], chunk[1:]


def prepare_tinystories(cache_dir: str = ".cache/tokens") -> str:
    """Tokenize TinyStories and cache to disk."""
    import numpy as np
    import tiktoken
    from datasets import load_dataset
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    tokens_file = cache_path / "tinystories.bin"
    
    if tokens_file.exists():
        print(f"Using cached tokens: {tokens_file}")
        return str(tokens_file)
    
    print("Tokenizing TinyStories (one-time)...")
    
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    
    all_tokens = []
    for i, example in enumerate(tqdm(ds, desc="Tokenizing")):
        if example["text"]:
            tokens = enc.encode(example["text"])
            all_tokens.extend(tokens)
            all_tokens.append(enc.eot_token)
    
    tokens_array = np.array(all_tokens, dtype=np.int32)
    tokens_array.tofile(tokens_file)
    
    print(f"Saved {len(tokens_array):,} tokens to {tokens_file}")
    return str(tokens_file)


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
    """Generate a sample from the model for monitoring."""
    import tiktoken
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
        
        # Limit context
        if x.shape[1] > 512:
            x = x[:, -512:]
    
    model.train()
    return enc.decode(x[0].tolist())


def train(
    model_size: str = "150M",
    batch_size: int = 32,
    seq_len: int = 512,
    learning_rate: float = 5e-4,
    min_lr: float = 1e-6,
    warmup_steps: int = 500,
    max_tokens: Optional[int] = None,
    log_interval: int = 50,
    eval_interval: int = 1000,
    save_interval: int = 5000,
    output_dir: str = "checkpoints",
    use_wandb: bool = False,
    compile_model: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output dir
    run_name = f"{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Model
    print(f"\nCreating {model_size} model...")
    model = create_model(model_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Compile for speed (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Data
    print("\nPreparing data...")
    tokens_path = prepare_tinystories()
    dataset = TokenizedDataset(tokens_path, seq_len)
    print(f"Dataset: {len(dataset):,} samples ({len(dataset) * seq_len / 1e6:.1f}M tokens)")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    
    # Mixed precision
    scaler = GradScaler('cuda')
    
    # Calculate steps
    tokens_per_step = batch_size * seq_len
    total_tokens = len(dataset) * seq_len
    if max_tokens:
        total_tokens = min(total_tokens, max_tokens)
    max_steps = total_tokens // tokens_per_step
    
    print(f"\nTraining for {max_steps:,} steps ({total_tokens/1e6:.1f}M tokens)")
    
    # W&B
    if use_wandb and HAS_WANDB:
        wandb.init(project="mini-model", name=run_name, config={
            "model_size": model_size,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "learning_rate": learning_rate,
            "num_params": num_params,
        })
    
    # Save training config
    config = {
        "model_size": model_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "learning_rate": learning_rate,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "total_tokens": total_tokens,
        "num_params": num_params,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Setup file logging
    log_file = open(output_path / "training.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    
    log(f"Training {model_size} model")
    log(f"Parameters: {num_params:,}")
    log(f"Batch size: {batch_size}, Seq len: {seq_len}")
    log(f"Total tokens: {total_tokens/1e6:.1f}M, Steps: {max_steps:,}")
    log("-" * 60)
    
    # Training loop
    model.train()
    step = 0
    tokens_seen = 0
    start_time = time.time()
    loss_accum = 0.0
    loss_count = 0
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while step < max_steps:
        for x, y in dataloader:
            if step >= max_steps:
                break
            
            x, y = x.to(device), y.to(device)
            
            # LR schedule
            lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward/backward with mixed precision
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping and norm tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Track smoothed loss
            loss_accum += loss.item()
            loss_count += 1
            
            tokens_seen += tokens_per_step
            step += 1
            pbar.update(1)
            
            # Logging
            if step % log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                eta_sec = (max_steps - step) * (elapsed / step)
                eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.0f}m"
                
                avg_loss = loss_accum / loss_count
                ppl = math.exp(min(avg_loss, 10))
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.3f}',
                    'ppl': f'{ppl:.1f}',
                    'tok/s': f'{tokens_per_sec:.0f}',
                    'eta': eta_str,
                })
                
                # Log to file
                log_msg = (
                    f"step={step:>6} | loss={avg_loss:.4f} | ppl={ppl:.1f} | "
                    f"lr={lr:.2e} | grad={grad_norm:.2f} | tok/s={tokens_per_sec:.0f} | "
                    f"gpu={gpu_mem:.1f}GB | eta={eta_str}"
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
                        "tokens": tokens_seen,
                        "gpu_memory_gb": gpu_mem,
                        "gpu_memory_peak_gb": gpu_mem_peak,
                    }, step=step)
                
                # Reset smoothed loss
                loss_accum = 0.0
                loss_count = 0
            
            # Save checkpoint
            if step % save_interval == 0:
                ckpt_path = output_path / f"step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'tokens_seen': tokens_seen,
                    'config': config,
                }, ckpt_path)
                log(f"  Checkpoint saved: {ckpt_path}")
            
            # Generate sample for monitoring
            if step % eval_interval == 0:
                try:
                    sample = generate_sample(model, "Once upon a time", max_tokens=60)
                    log(f"\n  📝 Sample (step {step}):")
                    log(f"  {sample[:200]}...")
                    log("")
                    
                    if use_wandb and HAS_WANDB:
                        wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=step)
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
    final_ppl = math.exp(min(loss.item(), 10))
    
    log(f"\n{'='*60}")
    log("Training Complete!")
    log(f"{'='*60}")
    log(f"  Time: {total_time/3600:.2f} hours")
    log(f"  Tokens: {tokens_seen:,}")
    log(f"  Tokens/sec: {tokens_seen/total_time:,.0f}")
    log(f"  Final loss: {loss.item():.4f}")
    log(f"  Final PPL: {final_ppl:.2f}")
    log(f"  Output: {output_path}")
    log(f"  Config: {output_path}/config.json")
    log(f"  Log: {output_path}/training.log")
    
    log_file.close()
    
    if use_wandb and HAS_WANDB:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train mini-model on CUDA")
    
    # Model
    parser.add_argument("--size", default="150M", choices=["10M", "50M", "150M", "350M"])
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    
    # Optimizer
    parser.add_argument("--optimizer", default="adamw", 
                       choices=["adamw", "adam", "sgd", "lion", "adafactor"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    
    # Training limits
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1000, help="Generate sample every N steps")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    
    # Performance
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    
    args = parser.parse_args()
    
    train(
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
        use_wandb=args.wandb,
        compile_model=not args.no_compile,
    )


if __name__ == "__main__":
    main()
