import subprocess
import re
import sys
import time
from typing import List, Dict

# Configuration
PYTHON_EXEC = sys.executable # Use the current python interpreter
SCRIPT = "train_curriculum_cuda.py"
MODEL_SIZE = "150M"
MAX_TOKENS = 500_000  # Fast shallow run (~10 steps at BS=32, ~30 secs each)

# Experiments to run - WIDER search space, SHALLOWER runs
EXPERIMENTS = [
    # Batch size sweep
    {"batch_size": 32, "lr": 5e-4, "seq_len": 512, "note": "BS32"},
    {"batch_size": 64, "lr": 5e-4, "seq_len": 512, "note": "BS64"},
    {"batch_size": 128, "lr": 5e-4, "seq_len": 512, "note": "BS128"},
    {"batch_size": 256, "lr": 5e-4, "seq_len": 512, "note": "BS256"},
    {"batch_size": 512, "lr": 5e-4, "seq_len": 512, "note": "BS512 (VRAM test)"},
    
    # Learning rate sweep (best BS from above)
    {"batch_size": 128, "lr": 3e-4, "seq_len": 512, "note": "LR 3e-4"},
    {"batch_size": 128, "lr": 1e-3, "seq_len": 512, "note": "LR 1e-3"},
    {"batch_size": 128, "lr": 2e-3, "seq_len": 512, "note": "LR 2e-3"},
    
    # Sequence length (for memory test)
    {"batch_size": 128, "lr": 5e-4, "seq_len": 1024, "note": "Seq1024"},
    {"batch_size": 128, "lr": 5e-4, "seq_len": 2048, "note": "Seq2048"},
]

def run_experiment(config: Dict) -> Dict:
    print(f"\n🧪 Running: BS={config['batch_size']}, LR={config['lr']}, SeqLen={config['seq_len']} ({config['note']})")
    
    cmd = [
        PYTHON_EXEC, SCRIPT,
        "--size", MODEL_SIZE,
        "--batch-size", str(config['batch_size']),
        "--seq-len", str(config['seq_len']),
        "--lr", str(config['lr']),
        "--max-tokens", str(MAX_TOKENS),
        "--log-interval", "10",
        "--save-interval", "100000", # Don't save
        "--eval-interval", "100000", # Don't eval
        "--output-dir", "experiments/temp",
        "--no-compile", # Faster startup for short tests
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e.stderr}")
        return {**config, "status": "Failed", "tok_sec": 0, "loss": 0}

    # Parse output for metrics
    # Look for last log line: "step=... | ... | loss=... | ... | tok/s=..."
    tok_sec_matches = re.findall(r"tok/s=(\d+)", output)
    loss_matches = re.findall(r"loss=([\d\.]+)", output)
    
    if tok_sec_matches:
        # Average the last few to be stable
        avg_tok_sec = sum(map(int, tok_sec_matches[-5:])) / len(tok_sec_matches[-5:])
        final_loss = float(loss_matches[-1])
        print(f"✅ Done! Speed: {avg_tok_sec:.0f} tok/s | Loss: {final_loss:.4f}")
        return {**config, "status": "Success", "tok_sec": avg_tok_sec, "loss": final_loss}
    else:
        print("⚠️  Finished but couldn't parse metrics.")
        return {**config, "status": "Unknown", "tok_sec": 0, "loss": 0}

def main():
    results = []
    print(f"🚀 Starting Experiments on {MODEL_SIZE} Model...")
    print(f"   Target: {MAX_TOKENS} tokens per run")
    
    for exp in EXPERIMENTS:
        res = run_experiment(exp)
        results.append(res)
    
    print("\n" + "="*80)
    print(f"{'Batch':<8} | {'SeqLen':<8} | {'LR':<8} | {'Speed (tok/s)':<15} | {'Loss':<8} | {'Note':<20}")
    print("-" * 80)
    
    best_speed = 0
    best_config = None
    
    for r in results:
        print(f"{r['batch_size']:<8} | {r['seq_len']:<8} | {r['lr']:<8} | {r['tok_sec']:<15.0f} | {r['loss']:<8.4f} | {r['note']:<20}")
        if r['tok_sec'] > best_speed:
            best_speed = r['tok_sec']
            best_config = r
            
    print("="*80)
    if best_config:
        print(f"\n🏆 Recommendation: Batch Size {best_config['batch_size']}, SeqLen {best_config['seq_len']} (Speed: {best_speed:.0f} tok/s)")

if __name__ == "__main__":
    main()
