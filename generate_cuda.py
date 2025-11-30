#!/usr/bin/env python3
"""
Generate text from a trained PyTorch model.

Usage:
    python generate_cuda.py --checkpoint checkpoints/run/final.pt --prompt "Once upon a time"
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from train_cuda import create_model


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Generate text from prompt."""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                next_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if next_token.item() == tokenizer.eot_token:
                break
            
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--size", type=str, default="150M")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=3)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading {args.size} model from {args.checkpoint}...")
    model = create_model(args.size).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    
    for i in range(args.num_samples):
        output = generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print(f"\n[Sample {i+1}]")
        print(output)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
