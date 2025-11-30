#!/usr/bin/env python3
"""
Generate text from a trained model.

Usage:
    ./venv/bin/python scripts/generate.py --checkpoint checkpoints/10M_*/final.safetensors
    ./venv/bin/python scripts/generate.py --checkpoint checkpoints/best.safetensors --prompt "Once upon a time"
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlx.core as mx
import mlx.nn as nn

from data.tokenizer import get_tokenizer


class TransformerLM(nn.Module):
    """Same model as in train.py."""
    
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        
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


def create_model(size: str = "150M", vocab_size: int = 50257) -> TransformerLM:
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
        "200M": {"num_layers": 16, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    config = configs[size]
    config["vocab_size"] = vocab_size
    return TransformerLM(**config)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate text from prompt."""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])
    
    generated = []
    
    for _ in range(max_tokens):
        logits = model(x)
        next_logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        if top_k > 0:
            top_k_indices = mx.argpartition(-next_logits, top_k)[:top_k]
            top_k_logits = next_logits[top_k_indices]
            
            probs = mx.softmax(top_k_logits)
            idx = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = top_k_indices[idx].item()
        else:
            probs = mx.softmax(next_logits)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()
        
        mx.eval(next_token)
        
        # Stop at EOS
        if next_token == tokenizer.eos_token_id:
            break
        
        generated.append(next_token)
        x = mx.concatenate([x, mx.array([[next_token]])], axis=1)
    
    return tokenizer.decode(tokens + generated)


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--size", type=str, default="10M",
                       choices=["10M", "50M", "150M", "200M"])
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt to generate from")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mini-Model Text Generation")
    print("=" * 60)
    
    mx.set_default_device(mx.gpu)
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"Tokenizer: {tokenizer.vocab_size} vocab")
    
    # Load model
    print(f"Loading {args.size} model from {args.checkpoint}...")
    model = create_model(args.size, tokenizer.vocab_size)
    model.load_weights(args.checkpoint)
    mx.eval(model.parameters())
    print("Model loaded.")
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    
    for i in range(args.num_samples):
        output = generate(
            model, 
            tokenizer, 
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\n[Sample {i+1}]")
        print(output)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
