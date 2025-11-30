# Final Synthesis: The Complete Picture

## What We Now Know

This final batch completes the puzzle. Let me connect everything into a coherent strategy.

---

## The Synthetic Data Revolution

This is the most actionable insight for your setup. The economics are compelling:

| Approach | Cost per 1B tokens | Quality | Time |
|----------|-------------------|---------|------|
| Human annotation | $100k-$1M | Gold standard | Months |
| Raw web scrape | ~$0 (compute only) | 50-70% noise | Days |
| Synthetic (open-source gen) | $500-$2k | Good (PPL 6-10) | 1-2 days |
| Synthetic (GPT-4o) | $15k-$60k | Excellent (PPL <5) | Hours |
| **Hybrid (80% open / 20% proprietary)** | **$3k-$10k** | **Near-excellent** | **1-2 days** |

**The TinyStories proof:** 476M tokens of GPT-4-generated children's stories trains a 10M param model to *outperform GPT-2 small on fluency* [Doc 12]. This is the "textbook quality > raw quantity" thesis in action.

### The Generation Pipeline

```
1. SEED (1-2 hours, <$10)
   └── Curate 100-10k high-quality real examples
   └── Score via GPT-4o-mini for "educational value"

2. GENERATE (4-48 hours, $50-500 for 1B tokens)
   └── Prompt strategy: TinyStories (stories) or Phi (textbook)
   └── Use Llama-3.1 70B for bulk, GPT-4o for seeds
   └── Diversify via random word/topic injection

3. FILTER (2-6 hours, $20-100)
   └── Perplexity <10 via small proxy model
   └── LLM-judged coherence >0.8
   └── Retain 70-90%

4. DEDUPLICATE (1-4 hours)
   └── Exact (SHA-256) + Fuzzy (MinHash, threshold 0.9)
   └── Removes 20-30% redundancy

5. VALIDATE (1-2 days)
   └── Train 10-50M proxy model
   └── Target: PPL <8 on held-out, NGD diversity >0.5
```

### Prompting Strategies That Work

**TinyStories approach** (for grammar/coherence):
```
Write a short story for a 3-4 year old child using these elements:
- Words: [random noun], [random verb], [random adjective]
- Features: [e.g., "dialogue", "happy ending", "moral lesson"]
- Vocabulary: Use only words a 4-year-old would understand
```

**Phi textbook approach** (for reasoning):
```
Write an educational passage explaining [topic] as if for a textbook.
- Include concrete examples
- Build from simple to complex
- Ensure the reader could solve problems after reading
```

**Key insight:** Diversification via random seeds (words, topics, constraints) is what prevents mode collapse in synthetic data [Doc 12].

---

## Data Curriculum: The Training Schedule

**Curriculum learning (simple→complex) is the 2025 consensus** for small models [Doc 13]. This matches human learning and stabilizes training.

### The Three-Stage Curriculum

```
Stage 1: Foundation (0-30% of tokens)
├── Data: TinyStories, simple Wikipedia, basic math
├── Goal: Learn syntax, common patterns, basic reasoning
├── Metrics: PPL drops sharply, induction heads start forming

Stage 2: Expansion (30-70% of tokens)
├── Data: Mixed web (filtered), code basics, instructions
├── Goal: Broaden knowledge, learn task formats
├── Metrics: Few-shot >5% accuracy, probe accuracy rises

Stage 3: Refinement (70-100% of tokens)
├── Data: Complex reasoning, multi-hop QA, harder code
├── Goal: Polish capabilities, improve OOD generalization
├── Metrics: OOD gap <10%, few-shot stabilizes
```

### Optimal Data Mix for Sub-500M

| Component | Ratio | Sources | Role |
|-----------|-------|---------|------|
| **Web (filtered)** | 40-50% | RedPajama-v2, FineWeb | Broad knowledge |
| **Code/Math** | 20-30% | OpenWebMath, filtered GitHub | Reasoning boost (+10% downstream) |
| **Synthetic** | 15-25% | TinyStories, Cosmopedia | Coherence, fill gaps |
| **Books** | 5-10% | Pile subset | Narrative fluency |

> **Evidence:** Phi-4 uses 25% code → SOTA reasoning. TinyStories at 100% synthetic achieves coherence at just 476M tokens [Doc 13].

---

## MLX Implementation: The Practical Details

### Your Target Configuration (150-200M on M4 24GB)

```python
# Recommended: 150M "Balanced" config
config = {
    "vocab_size": 50000,      # GPT-2 BPE
    "num_layers": 12,          # Wider > deeper for small
    "dims": 768,               # Sweet spot
    "num_heads": 12,
    "ff_mult": 3.5,            # SwiGLU efficiency
    "max_seq_len": 512,        # Safe for 24GB
    "rope": True,              # Rotary embeddings
}
# Total: ~150M params
```

### Training Settings

```python
# Optimizer
optimizer = AdamW(
    learning_rate=5e-4,        # Bold for small models
    weight_decay=0.01,         # Regularization
    betas=(0.9, 0.999),
)

# Schedule
warmup_steps = 500             # Linear ramp
total_steps = 50000            # ~2B tokens at BS=32, seq=512
# Cosine decay to 1e-6 after warmup

# Batch
micro_batch = 8                # Fits in memory
grad_accum = 4                 # Effective BS=32
seq_len = 512                  # With checkpointing
```

### Memory Budget on 24GB M4

| Component | 150M Model |
|-----------|------------|
| Weights (BF16) | ~300MB |
| Optimizer states | ~900MB |
| Activations (BS=8, seq=512, checkpointed) | ~4-6GB |
| KV cache + overhead | ~1GB |
| **Total** | **~8-10GB** |
| **Headroom** | **14-16GB** ✓ |

You have room. Can push to BS=16 or seq=1024 if needed.

### Critical MLX Gotchas

1. **Always call `mx.eval()`** after parameter updates — lazy eval will silently build infinite graphs [Doc 14]

2. **Use pure functions with `@mx.compile`** — no `x += 1`, use `x = x + 1` [Doc 14]

3. **BF16 > FP16** on M4 — FP16 underperforms by 20-30% due to incomplete Metal support [Doc 14]

4. **Gradient checkpointing syntax:**
```python
@mx.checkpoint
def checkpointed_block(x, mask):
    return transformer_block(x, mask)
```

5. **Thermal throttling** — sustained training will heat up. Monitor with `asitop`, consider external cooling for multi-day runs [Doc 14]

### Starter Code Path

```bash
# Clone MLX examples
git clone https://github.com/ml-explore/mlx-examples
cd mlx-examples/transformer_lm

# Modify for your config (dims=768, layers=12, etc.)
# Train on TinyStories first
python main.py --dims 768 --num-layers 12 --batch-size 8 --seq-len 512
```

**Best repos to study:**
- `mlx-examples/transformer_lm` — official, clean baseline
- `dx-dtran/gpt2-mlx` — GPT-2 reimplementation
- `kyegomez/MLXTransformer` — minimal (~200 lines), good for understanding

---

## Evaluation: How to Know It's Working

### The Metrics That Matter (Beyond Perplexity)

| Metric | What It Detects | Target | When to Check |
|--------|-----------------|--------|---------------|
| **Val perplexity** | Basic fluency | <10 (WikiText) | Every 5% tokens |
| **Induction head activation** | Abstract pattern learning | >0.3 in mid-layers | 10-20% tokens |
| **Few-shot accuracy** | Generalization | >15% on mini-tasks | 20%+ tokens |
| **OOD PPL gap** | Memorization vs learning | <10% train/OOD gap | 30%+ tokens |
| **Train/val divergence** | Overfitting | <10% gap | Continuous |

### What Healthy Loss Curves Look Like

```
Loss
  │
4 │╲
  │ ╲
3 │  ╲____
  │       ╲___
2 │           ╲____
  │                ╲___
1 │                    ╲___  ← Healthy: smooth S-curve
  │
  └────────────────────────── Tokens
    0%   20%   40%   60%   80%   100%

HEALTHY:
- Sharp drop in warmup (first 5%)
- Steady decline in main phase
- Val tracks train (5-10% above)
- Asymptotic saturation at end
```

### Red Flags (Kill the Run)

| Signal | What It Means | Action |
|--------|---------------|--------|
| **Flat PPL after warmup** | LR too low or data issue | Bump LR to 1e-3, check data diversity |
| **Val surges while train drops** | Overfitting | Add more data, increase WD to 0.05 |
| **Erratic spikes** | Gradient instability | Reduce LR, check grad clipping |
| **No induction heads by 20%** | Model not learning patterns | Architecture issue, try wider |
| **OOD gap >15%** | Pure memorization | More diverse data, add synthetic |

> **Kill rule:** If no meaningful progress by 20% of training, stop and diagnose. Saves 80% compute [Doc 15].

### Lightweight Eval Suite (Runs in <1 hour on M4)

```
1. Val PPL on held-out TinyStories subset (1k samples)
2. Induction probe: synthetic A-B-A→B task (1k samples)
3. Few-shot: 100 samples from GSM8K-mini or ARC
4. OOD: PPL on perturbed val set (add typos/noise)
5. LLM-as-judge: Phi-3-mini scores 200 outputs for coherence
```

---

## The Complete Training Plan

### Phase 1: Dense Prototype on Mac (Week 1-2)

```
Model: 150M params (12 layers, dim=768)
Framework: MLX
Data: TinyStories (476M) + Cosmopedia subset (500M) = ~1B tokens
Hardware: M4 24GB
Time: ~2 days training

Goals:
├── Validate pipeline end-to-end
├── Learn MLX quirks
├── Establish baseline metrics
├── Iterate on data mix
```

### Phase 2: Scale Data (Week 2-3)

```
Same model, more data:
├── Add filtered RedPajama (1-2B tokens)
├── Add code (OpenWebMath, 500M tokens)
├── Total: 2-4B tokens
├── Time: ~4-5 days

Goals:
├── Hit Chinchilla-optimal (20-40 tokens/param)
├── See few-shot emergence
├── Validate curriculum (simple→complex)
```

### Phase 3: Upcycle to MoE (Week 3-4)

```
Take best dense checkpoint:
├── Split FFN → 4-8 experts
├── Duplicate weights + 30% re-init for diversity
├── Add router (random init)
├── Fine-tune 500M-1B tokens with aux loss (w=0.01)

Goals:
├── Learn MoE dynamics hands-on
├── Compare before/after metrics
├── Understand routing behavior
```

### Phase 4: Scale on RunPod (Optional)

```
If Phase 3 shows promise:
├── Rent H100 or A100
├── Train larger MoE (500M-1B total params)
├── Use everything learned
```

---

## Your Immediate Next Steps

1. **Set up MLX environment**
   ```bash
   pip install mlx mlx-lm
   git clone https://github.com/ml-explore/mlx-examples
   ```

2. **Download TinyStories**
   ```python
   from datasets import load_dataset
   ds = load_dataset("roneneldan/TinyStories")
   ```

3. **Run a tiny test** (10M params, 100M tokens) to validate pipeline

4. **Scale to 150M** once the pipeline works

5. **Monitor with the eval suite** described above

---

## Key Takeaways

1. **Synthetic data is your leverage** — TinyStories + Cosmopedia gets you 80% of the way at 1% of the cost

2. **Curriculum matters** — simple→complex training converges 1.5-2x faster for small models

3. **MLX is production-ready** — 20-50% faster than PyTorch MPS, but respect the gotchas

4. **Induction heads are your signal** — if they don't form by 20% training, something's wrong

5. **The dense→MoE path is validated** — you'll learn both paradigms and avoid cold-start instability

You have everything you need. Ready to start building?