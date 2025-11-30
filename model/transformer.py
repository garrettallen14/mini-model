"""
Transformer model for mini-model training.

Implements the 150M config from outline.md with:
- RoPE (Rotary Position Embeddings)
- SwiGLU activation
- Gradient checkpointing
- Pre-norm architecture
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embeddings."""
    
    def __init__(self, dims: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dims = dims
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2).astype(mx.float32) / dims))
        self._inv_freq = inv_freq
    
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary embeddings to x."""
        seq_len = x.shape[1]
        
        # Create position indices
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        
        # Compute angles
        freqs = mx.outer(positions, self._inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Apply rotary embedding to input tensor."""
        # Split into pairs and rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Reshape cos/sin for broadcasting
        cos = cos[:, :x1.shape[-1]]
        sin = sin[:, :x1.shape[-1]]
        
        # Apply rotation
        rotated = mx.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)
        
        return rotated.reshape(x.shape)


class SwiGLU(nn.Module):
    """SwiGLU activation (used in LLaMA, Phi, etc.)."""
    
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.w1 = nn.Linear(dims, hidden_dims, bias=False)
        self.w2 = nn.Linear(hidden_dims, dims, bias=False)
        self.w3 = nn.Linear(dims, hidden_dims, bias=False)
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, dims: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dims, dims, bias=False)
        self.wk = nn.Linear(dims, dims, bias=False)
        self.wv = nn.Linear(dims, dims, bias=False)
        self.wo = nn.Linear(dims, dims, bias=False)
        
        self.rope = RoPE(self.head_dim, max_seq_len)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape
        
        # Project to Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        # Reshape for multi-head attention
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset)
        k = self.rope(k, offset)
        
        # Handle KV cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        
        new_cache = (k, v)
        
        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v
        
        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.wo(output)
        
        return output, new_cache


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(
        self, 
        dims: int, 
        num_heads: int, 
        ff_mult: float = 3.5,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.attention = Attention(dims, num_heads, max_seq_len)
        self.attention_norm = nn.RMSNorm(dims)
        
        hidden_dims = int(dims * ff_mult)
        # Round to nearest multiple of 64 for efficiency
        hidden_dims = ((hidden_dims + 63) // 64) * 64
        
        self.ffn = SwiGLU(dims, hidden_dims)
        self.ffn_norm = nn.RMSNorm(dims)
    
    def __call__(
        self, 
        x: mx.array, 
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Attention with residual
        h, new_cache = self.attention(self.attention_norm(x), mask, cache)
        x = x + h
        
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, new_cache


class MiniModelLM(nn.Module):
    """
    Mini-Model Language Model.
    
    Configs from outline.md:
    - 150M: 12 layers, 768 dims, 12 heads
    - 200M: 16 layers, 768 dims, 12 heads
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        num_layers: int = 12,
        dims: int = 768,
        num_heads: int = 12,
        ff_mult: float = 3.5,
        max_seq_len: int = 2048,
        checkpoint: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dims = dims
        self.checkpoint = checkpoint
        
        self.embedding = nn.Embedding(vocab_size, dims)
        
        self.layers = [
            TransformerBlock(dims, num_heads, ff_mult, max_seq_len)
            for _ in range(num_layers)
        ]
        
        self.norm = nn.RMSNorm(dims)
        self.output = nn.Linear(dims, vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.embedding.weight
    
    def __call__(
        self, 
        x: mx.array,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[list]]:
        B, L = x.shape
        
        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        
        # Embedding
        h = self.embedding(x)
        
        # Transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            
            if self.checkpoint and self.training:
                # Gradient checkpointing during training
                h, c = mx.checkpoint(layer)(h, mask, layer_cache)
            else:
                h, c = layer(h, mask, layer_cache)
            
            new_cache.append(c)
        
        # Output
        h = self.norm(h)
        logits = self.output(h)
        
        return logits, new_cache if cache is not None else None
    
    def generate(
        self,
        prompt: mx.array,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> mx.array:
        """Generate tokens autoregressively."""
        cache = None
        tokens = prompt
        
        for _ in range(max_tokens):
            logits, cache = self(tokens if cache is None else tokens[:, -1:], cache)
            logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
            sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
            cumsum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
            
            # Find cutoff
            mask = cumsum - mx.softmax(sorted_logits, axis=-1) > top_p
            sorted_logits = mx.where(mask, -float('inf'), sorted_logits)
            
            # Sample
            probs = mx.softmax(sorted_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.take_along_axis(sorted_indices, next_token[:, None], axis=-1)
            
            tokens = mx.concatenate([tokens, next_token], axis=-1)
            
            # Stop at EOS
            if next_token.item() == 50256:  # GPT-2 EOS
                break
        
        return tokens


def create_model(size: str = "150M") -> MiniModelLM:
    """Create model by size string."""
    configs = {
        "10M": {"num_layers": 4, "dims": 256, "num_heads": 4, "checkpoint": False},
        "50M": {"num_layers": 8, "dims": 512, "num_heads": 8, "checkpoint": False},
        "150M": {"num_layers": 12, "dims": 768, "num_heads": 12, "checkpoint": True},
        "200M": {"num_layers": 16, "dims": 768, "num_heads": 12, "checkpoint": True},
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}")
    
    return MiniModelLM(**configs[size])


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    from mlx.utils import tree_flatten
    return sum(x.size for _, x in tree_flatten(model.parameters()))


if __name__ == "__main__":
    # Test model creation
    for size in ["10M", "50M", "150M"]:
        model = create_model(size)
        mx.eval(model.parameters())
        params = count_parameters(model)
        print(f"{size}: {params:,} parameters ({params/1e6:.1f}M)")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model = create_model("150M")
    mx.eval(model.parameters())
    
    x = mx.random.randint(0, 50257, (2, 128))
    logits, _ = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
