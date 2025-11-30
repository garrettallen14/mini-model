"""
Evaluation metrics for mini-model training.

From outline.md:
- Val perplexity (target <10 on WikiText)
- Induction head activation (>0.3 in mid-layers by 20% training)
- Few-shot accuracy (>15% on mini-tasks)
- OOD PPL gap (<10% train/OOD gap)
- Train/val divergence (<10% gap)
"""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def compute_perplexity(logits: mx.array, targets: mx.array) -> float:
    """Compute perplexity from logits and targets."""
    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    mx.eval(loss)
    return math.exp(loss.item())


def compute_loss(model, batch: mx.array) -> float:
    """Compute cross-entropy loss on a batch."""
    x, y = batch[:, :-1], batch[:, 1:]
    logits, _ = model(x)
    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
    mx.eval(loss)
    return loss.item()


class InductionHeadProbe:
    """
    Probe for detecting induction head formation.
    
    Induction heads detect and complete patterns like:
    [A][B] ... [A] -> [B]
    
    From outline.md: Should activate >0.3 in mid-layers by 20% training.
    """
    
    def __init__(self, model, layer_indices: Optional[List[int]] = None):
        self.model = model
        # Default to middle layers
        if layer_indices is None:
            num_layers = len(model.layers)
            self.layer_indices = list(range(num_layers // 3, 2 * num_layers // 3))
        else:
            self.layer_indices = layer_indices
    
    def create_induction_test(
        self, 
        vocab_size: int = 50257,
        seq_len: int = 64,
        batch_size: int = 8,
    ) -> mx.array:
        """
        Create test sequences with repeated patterns.
        Format: [random prefix] [A] [B] [random middle] [A] [?]
        If induction heads work, model should predict [B]
        """
        sequences = []
        expected_tokens = []
        
        for _ in range(batch_size):
            # Random tokens (avoid special tokens)
            prefix_len = seq_len // 4
            middle_len = seq_len // 2
            
            prefix = mx.random.randint(100, vocab_size - 100, (prefix_len,))
            middle = mx.random.randint(100, vocab_size - 100, (middle_len,))
            
            # Pattern tokens
            token_a = mx.random.randint(100, vocab_size - 100, (1,))
            token_b = mx.random.randint(100, vocab_size - 100, (1,))
            
            # Build sequence: prefix + A + B + middle + A
            seq = mx.concatenate([prefix, token_a, token_b, middle, token_a])
            
            # Pad to seq_len + 1 (need space for prediction)
            if seq.shape[0] < seq_len + 1:
                pad = mx.zeros((seq_len + 1 - seq.shape[0],), dtype=mx.int32)
                seq = mx.concatenate([seq, pad])
            else:
                seq = seq[:seq_len + 1]
            
            sequences.append(seq)
            expected_tokens.append(token_b.item())
        
        return mx.stack(sequences), expected_tokens
    
    def measure_induction_score(self, batch_size: int = 16) -> Dict[str, float]:
        """
        Measure how well the model completes induction patterns.
        Returns accuracy and confidence scores.
        """
        test_seqs, expected = self.create_induction_test(batch_size=batch_size)
        
        # Get predictions
        x = test_seqs[:, :-1]
        logits, _ = self.model(x)
        mx.eval(logits)
        
        # Check last position predictions
        last_logits = logits[:, -1, :]
        predictions = mx.argmax(last_logits, axis=-1)
        mx.eval(predictions)
        
        # Compute accuracy
        expected_arr = mx.array(expected)
        correct = (predictions == expected_arr).astype(mx.float32)
        accuracy = mx.mean(correct).item()
        
        # Compute confidence (softmax probability of correct answer)
        probs = mx.softmax(last_logits, axis=-1)
        confidence = 0.0
        for i, exp in enumerate(expected):
            confidence += probs[i, exp].item()
        confidence /= len(expected)
        
        return {
            "induction_accuracy": accuracy,
            "induction_confidence": confidence,
        }


class FewShotEvaluator:
    """
    Simple few-shot evaluation on pattern completion tasks.
    
    From outline.md: Target >15% accuracy by 20% training.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def eval_simple_math(self, num_examples: int = 20) -> float:
        """
        Test simple arithmetic: "2 + 3 = " -> "5"
        """
        import random
        
        correct = 0
        for _ in range(num_examples):
            a = random.randint(1, 9)
            b = random.randint(1, 9)
            op = random.choice(["+", "-"])
            
            if op == "+":
                answer = a + b
            else:
                answer = a - b
            
            prompt = f"{a} {op} {b} ="
            tokens = self.tokenizer.encode(prompt)
            
            x = mx.array([tokens])
            logits, _ = self.model(x)
            mx.eval(logits)
            
            # Get predicted next token
            next_logits = logits[0, -1, :]
            pred_token = mx.argmax(next_logits).item()
            pred_text = self.tokenizer.decode([pred_token]).strip()
            
            # Check if correct
            try:
                if int(pred_text) == answer:
                    correct += 1
            except:
                pass
        
        return correct / num_examples
    
    def eval_word_completion(self, num_examples: int = 20) -> float:
        """
        Test word pattern completion: "cat, dog, bird, " -> animal word
        """
        patterns = [
            ("one, two, three, ", ["four", "4"]),
            ("red, blue, green, ", ["yellow", "orange", "purple"]),
            ("apple, banana, orange, ", ["grape", "mango", "pear"]),
            ("Monday, Tuesday, Wednesday, ", ["Thursday", "Thurs"]),
            ("January, February, March, ", ["April"]),
        ]
        
        correct = 0
        for prompt, valid_answers in patterns:
            tokens = self.tokenizer.encode(prompt)
            x = mx.array([tokens])
            
            logits, _ = self.model(x)
            mx.eval(logits)
            
            # Generate a few tokens
            generated = []
            for _ in range(3):
                next_logits = logits[0, -1, :]
                pred_token = mx.argmax(next_logits).item()
                generated.append(pred_token)
                
                x = mx.concatenate([x, mx.array([[pred_token]])], axis=1)
                logits, _ = self.model(x)
                mx.eval(logits)
            
            pred_text = self.tokenizer.decode(generated).strip().lower()
            
            # Check if any valid answer is in prediction
            for ans in valid_answers:
                if ans.lower() in pred_text:
                    correct += 1
                    break
        
        return correct / len(patterns)


class TrainingMetrics:
    """Track training metrics over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.tokens_seen: List[int] = []
        self.induction_scores: List[float] = []
    
    def add_train_loss(self, loss: float, tokens: int, lr: float):
        self.losses.append(loss)
        self.tokens_seen.append(tokens)
        self.learning_rates.append(lr)
    
    def add_val_loss(self, loss: float):
        self.val_losses.append(loss)
    
    def add_induction_score(self, score: float):
        self.induction_scores.append(score)
    
    def get_recent_avg(self) -> float:
        if not self.losses:
            return float('inf')
        return sum(self.losses[-self.window_size:]) / min(len(self.losses), self.window_size)
    
    def get_train_val_gap(self) -> Optional[float]:
        """Compute gap between train and val loss."""
        if not self.losses or not self.val_losses:
            return None
        
        recent_train = self.get_recent_avg()
        recent_val = self.val_losses[-1]
        
        # Gap as percentage
        return abs(recent_val - recent_train) / recent_train * 100
    
    def check_health(self) -> Dict[str, Tuple[bool, str]]:
        """
        Check training health based on outline.md criteria.
        Returns dict of (passed, message) tuples.
        """
        results = {}
        
        # Check 1: Loss decreasing
        if len(self.losses) >= 100:
            early = sum(self.losses[:50]) / 50
            recent = sum(self.losses[-50:]) / 50
            passed = recent < early
            results["loss_decreasing"] = (passed, f"Early: {early:.3f}, Recent: {recent:.3f}")
        
        # Check 2: Train/val gap
        gap = self.get_train_val_gap()
        if gap is not None:
            passed = gap < 10
            results["train_val_gap"] = (passed, f"Gap: {gap:.1f}%")
        
        # Check 3: Induction heads forming
        if self.induction_scores:
            recent_score = self.induction_scores[-1]
            passed = recent_score > 0.1  # Lower threshold early on
            results["induction_heads"] = (passed, f"Score: {recent_score:.3f}")
        
        # Check 4: No NaN/Inf
        if self.losses:
            has_nan = any(not math.isfinite(l) for l in self.losses[-100:])
            results["no_nan"] = (not has_nan, "Clean" if not has_nan else "NaN detected!")
        
        return results
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """
        Check if we should stop training early.
        From outline.md: "If no meaningful progress by 20% of training, stop"
        """
        if len(self.losses) < 200:
            return False, "Too early to judge"
        
        # Check if loss is stuck
        first_quarter = sum(self.losses[:50]) / 50
        recent = sum(self.losses[-50:]) / 50
        
        if recent >= first_quarter * 0.95:  # Less than 5% improvement
            return True, f"Loss stuck: {first_quarter:.3f} -> {recent:.3f}"
        
        return False, "Training progressing normally"


if __name__ == "__main__":
    print("Testing metrics module...")
    
    metrics = TrainingMetrics()
    
    # Simulate training
    for i in range(200):
        loss = 5.0 - (i / 200) * 2.0 + (0.1 * (i % 10 - 5))
        metrics.add_train_loss(loss, i * 1000, 5e-4)
    
    metrics.add_val_loss(3.2)
    metrics.add_induction_score(0.15)
    
    print(f"Recent avg loss: {metrics.get_recent_avg():.3f}")
    print(f"Train/val gap: {metrics.get_train_val_gap():.1f}%")
    
    print("\nHealth checks:")
    for name, (passed, msg) in metrics.check_health().items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {msg}")
    
    should_stop, reason = metrics.should_stop_early()
    print(f"\nShould stop: {should_stop} ({reason})")
