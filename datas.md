# The Concrete Data Plan

This is exactly what we needed. The existing datasets are *extremely* good — you don't need to generate much custom data for your first run.

---

## Your Specific Data Recipe: 3B Tokens for 150M Model

This gives you 20 tokens/param (Chinchilla-optimal floor) with headroom to scale to 4-5B if you want to overtrain.

### The Split

| Category | % | Tokens | Source | Why |
|----------|---|--------|--------|-----|
| **Math** | 35% | 1.05B | OpenWebMath (800M) + MetaMathQA (250M) | Reasoning foundation, already filtered |
| **Code** | 30% | 900M | StarCoder Python/Jupyter subset | Verifiable structure, math imports |
| **General/Textbook** | 25% | 750M | TinyStories (476M) + Cosmopedia (274M) | Coherence, curriculum base |
| **Books/Narrative** | 10% | 300M | Dolma books subset | Fluency, diversity |
| **Total** | 100% | **3B** | | |

---

## The Exact Downloads

```python
from datasets import load_dataset

# MATH (1.05B tokens)
openwebmath = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
metamathqa = load_dataset("meta-math/MetaMathQA", split="train")  # 395k samples

# CODE (900M tokens)  
starcoder = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)

# GENERAL (750M tokens)
tinystories = load_dataset("roneneldan/TinyStories", split="train")  # 476M tokens
cosmopedia = load_dataset("HuggingFaceTB/cosmopedia", split="train", streaming=True)

# BOOKS (300M tokens)
dolma = load_dataset("allenai/dolma", data_dir="books", split="train", streaming=True)
```

---

## Quality Already Applied (You Don't Need to Re-Filter)

| Dataset | Filtering Done | Decontamination | Quality |
|---------|---------------|-----------------|---------|
| **OpenWebMath** | Mixtral-8x7B classifier + heuristics | GSM8K/MATH removed | High |
| **MetaMathQA** | Verifiable solutions, GPT-3.5 rewrites | N/A (synthetic) | High |
| **StarCoder** | Deduped, license-filtered | N/A | Medium-High |
| **TinyStories** | GPT-4 generated, coherence-filtered | N/A (synthetic) | High |
| **Cosmopedia** | "Educational value" classifier | N/A (synthetic) | High |

**Bottom line:** These datasets did the hard filtering work. You can use them directly.

---

## Should You Generate Custom Code/Math Data?

**Short answer: Not for v1.** The research is clear:

> "Not essential if using these datasets—they cover well for general SLMs, with 80–90% of needs met" [Doc 16]

**When custom generation makes sense:**
- Domain-specific code (bioinformatics, finance, etc.)
- Scaling beyond 10B tokens
- Filling gaps after seeing eval results

**Better use of your generation capability:**
Instead of generating from scratch, **augment 20% of StarCoder** with GPT-4o rewrites:
- Take existing code samples
- Prompt: "Rewrite this with better comments and variable names" or "Add a docstring explaining the math"
- This yields 5-10% uplift with much less effort than pure generation [Doc 16]

---

## The Training Curriculum (Concrete Schedule)

```
Phase 1: Foundation (0-1B tokens, ~33% of training)
├── 100% TinyStories + simple Cosmopedia
├── Goal: Learn syntax, basic patterns
├── Check: PPL dropping, no instability

Phase 2: Math Ramp (1B-2B tokens, ~33% of training)  
├── 50% OpenWebMath + MetaMathQA
├── 30% Code (StarCoder easy examples)
├── 20% General (continue Cosmopedia)
├── Goal: Reasoning patterns emerge
├── Check: Induction heads forming

Phase 3: Full Mix (2B-3B tokens, ~33% of training)
├── 35% Math (harder OpenWebMath)
├── 35% Code (full StarCoder complexity)
├── 20% General
├── 10% Books
├── Goal: Consolidate, generalize
├── Check: Few-shot >10%, OOD gap <10%
```

---

## Practical Implementation

### Tokenization Strategy

Use GPT-2 tokenizer (already trained on code):
```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Vocab size: 50,257
```

### Streaming for Memory Efficiency

3B tokens won't fit in RAM. Use streaming + shuffle buffer:
```python
from datasets import interleave_datasets

# Create weighted mix
datasets = [openwebmath, metamathqa, starcoder, tinystories, cosmopedia, dolma]
weights = [0.27, 0.08, 0.30, 0.16, 0.09, 0.10]  # Matches our split

mixed = interleave_datasets(datasets, probabilities=weights, stopping_strategy="all_exhausted")
```

### Sequence Packing

Don't waste tokens on padding:
```python
def pack_sequences(examples, seq_len=512):
    # Concatenate all tokens, split into seq_len chunks
    all_tokens = [t for ex in examples for t in ex['input_ids']]
    return [all_tokens[i:i+seq_len] for i in range(0, len(all_tokens)-seq_len, seq_len)]
```

---

## Time & Compute Estimate

| Metric | Estimate |
|--------|----------|
| Total tokens | 3B |
| Tokens/sec (M4 24GB, BS=8, seq=512) | ~150-200 |
| Training time | **~4-5 days** |
| Storage (tokenized) | ~12-15GB |

---

## The Decision Tree

```
START HERE
    │
    ▼
Do you want to validate the pipeline first?
    │
    ├─► YES: Train on TinyStories only (476M tokens, ~8 hours)
    │        Then proceed to full mix
    │
    └─► NO: Go straight to full 3B mix (4-5 days)
    
AFTER TRAINING
    │
    ▼
Are math/code evals weak?
    │
    ├─► YES: Generate custom augmentations (20% rewrite)
    │        Add FineMath or OpenMathInstruct-1
    │
    └─► NO: Proceed to MoE upcycling
```

---

## Summary: Your Exact Shopping List

| Dataset | Tokens to Use | HF Path |
|---------|---------------|---------|
| OpenWebMath | 800M | `open-web-math/open-web-math` |
| MetaMathQA | 250M | `meta-math/MetaMathQA` |
| StarCoder (Python) | 900M | `bigcode/starcoderdata` |
| TinyStories | 476M | `roneneldan/TinyStories` |
| Cosmopedia | 274M | `HuggingFaceTB/cosmopedia` |
| Dolma Books | 300M | `allenai/dolma` |
| **TOTAL** | **3B** | |

You can start downloading today and be training by tomorrow.

**Your code/math generation capability?** Save it for v2 after you see where the model is weak. The existing datasets are genuinely good enough for a strong first run.

Want me to write the actual data loading and preprocessing script?