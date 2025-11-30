"""
Tokenizer setup for mini-model.

Uses GPT-2 tokenizer (50,257 vocab) as recommended in datas.md.
"""

from typing import List

# Try tiktoken first (faster), fall back to transformers
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from transformers import GPT2TokenizerFast
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class Tokenizer:
    """
    Unified tokenizer interface.
    Uses tiktoken if available, else transformers.
    """
    
    def __init__(self, name: str = "gpt2"):
        self.name = name
        self._tokenizer = None
        self._backend = None
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        if HAS_TIKTOKEN and self.name == "gpt2":
            self._tokenizer = tiktoken.get_encoding("gpt2")
            self._backend = "tiktoken"
        elif HAS_TRANSFORMERS:
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self._backend = "transformers"
        else:
            raise ImportError("Need either tiktoken or transformers installed")
        
        print(f"Tokenizer initialized: {self.name} via {self._backend}")
    
    @property
    def vocab_size(self) -> int:
        if self._backend == "tiktoken":
            return self._tokenizer.n_vocab
        return len(self._tokenizer)
    
    @property
    def eos_token_id(self) -> int:
        if self._backend == "tiktoken":
            return self._tokenizer.eot_token
        return self._tokenizer.eos_token_id
    
    @property
    def pad_token_id(self) -> int:
        return self.eos_token_id
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self._backend == "tiktoken":
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._backend == "tiktoken":
            return self._tokenizer.decode(token_ids)
        return self._tokenizer.decode(token_ids)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        if self._backend == "tiktoken":
            return self._tokenizer.encode_batch(texts)
        return [self.encode(t) for t in texts]


def get_tokenizer(name: str = "gpt2") -> Tokenizer:
    """Get a tokenizer instance."""
    return Tokenizer(name)


if __name__ == "__main__":
    tok = get_tokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    
    test = "Hello, world! This is a test."
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    
    print(f"Original: {test}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test == decoded}")
