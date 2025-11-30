"""
Data configuration based on datas.md plan.

Target: 3B tokens for 150M model (20 tokens/param Chinchilla-optimal)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    hf_path: str
    split: str = "train"
    data_dir: Optional[str] = None
    streaming: bool = True
    target_tokens: int = 0  # in millions
    text_column: str = "text"
    

@dataclass  
class DataMixConfig:
    """Full data mix configuration."""
    total_tokens: int = 3_000_000_000  # 3B
    seq_len: int = 512
    
    # Dataset weights (from datas.md)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    
    # Curriculum phases
    phase_1_end: float = 0.33  # Foundation: 0-33%
    phase_2_end: float = 0.66  # Math ramp: 33-66%
    # Phase 3: 66-100% (full mix)


# From datas.md "Your Exact Shopping List"
DATASETS = {
    # MATH (35% = 1.05B tokens)
    "openwebmath": DatasetConfig(
        name="openwebmath",
        hf_path="open-web-math/open-web-math",
        target_tokens=800_000_000,
        text_column="text",
    ),
    "metamathqa": DatasetConfig(
        name="metamathqa", 
        hf_path="meta-math/MetaMathQA",
        streaming=False,  # Small enough to load fully
        target_tokens=250_000_000,
        text_column="query",  # Has query + response columns
    ),
    
    # CODE (30% = 900M tokens)
    "starcoder": DatasetConfig(
        name="starcoder",
        hf_path="bigcode/starcoderdata",
        data_dir="python",
        target_tokens=900_000_000,
        text_column="content",
    ),
    
    # GENERAL (25% = 750M tokens)
    "tinystories": DatasetConfig(
        name="tinystories",
        hf_path="roneneldan/TinyStories",
        streaming=False,  # Only 476M tokens
        target_tokens=476_000_000,
        text_column="text",
    ),
    "cosmopedia": DatasetConfig(
        name="cosmopedia",
        hf_path="HuggingFaceTB/cosmopedia",
        target_tokens=274_000_000,
        text_column="text",
    ),
    
    # BOOKS (10% = 300M tokens)
    "dolma_books": DatasetConfig(
        name="dolma_books",
        hf_path="allenai/dolma",
        data_dir="books",
        target_tokens=300_000_000,
        text_column="text",
    ),
}


# Curriculum phases from datas.md
CURRICULUM = {
    "phase_1": {
        "name": "Foundation",
        "token_range": (0, 1_000_000_000),  # 0-1B
        "mix": {
            "tinystories": 0.70,
            "cosmopedia": 0.30,
        },
        "description": "Learn syntax, basic patterns",
    },
    "phase_2": {
        "name": "Math Ramp", 
        "token_range": (1_000_000_000, 2_000_000_000),  # 1B-2B
        "mix": {
            "openwebmath": 0.35,
            "metamathqa": 0.15,
            "starcoder": 0.30,
            "cosmopedia": 0.20,
        },
        "description": "Reasoning patterns emerge",
    },
    "phase_3": {
        "name": "Full Mix",
        "token_range": (2_000_000_000, 3_000_000_000),  # 2B-3B
        "mix": {
            "openwebmath": 0.27,
            "metamathqa": 0.08,
            "starcoder": 0.35,
            "tinystories": 0.10,
            "cosmopedia": 0.10,
            "dolma_books": 0.10,
        },
        "description": "Consolidate, generalize",
    },
}


def get_phase_for_token_count(token_count: int) -> str:
    """Determine which curriculum phase we're in."""
    for phase_name, phase_config in CURRICULUM.items():
        start, end = phase_config["token_range"]
        if start <= token_count < end:
            return phase_name
    return "phase_3"  # Default to final phase


def get_mix_weights(phase: str) -> Dict[str, float]:
    """Get dataset mixing weights for a phase."""
    return CURRICULUM[phase]["mix"]
