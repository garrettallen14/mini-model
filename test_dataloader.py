import argparse
import time
import torch
from torch.utils.data import DataLoader
from train_curriculum_cuda import CurriculumDataset, validate_datasets
import sys

def test_dataloader(steps=1000, batch_size=32, seq_len=512):
    print(f"🔍 Testing DataLoader for {steps} steps...")
    print(f"   Batch Size: {batch_size}")
    print(f"   Seq Len: {seq_len}")
    
    # 1. Validate datasets availability first
    print("\n[1/2] Validating Dataset Availability...")
    if not validate_datasets():
        print("❌ Dataset validation failed. Check internet connection or HF token.")
        return

    # 2. Stress test the iterator
    print("\n[2/2] Stress Testing Iterator (Tiktoken Check)...")
    try:
        dataset = CurriculumDataset(seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
        
        start_time = time.time()
        tokens_seen = 0
        
        for i, (x, y) in enumerate(dataloader):
            if i >= steps:
                break
            
            tokens_seen += x.numel()
            if i % 10 == 0:
                elapsed = time.time() - start_time
                tok_sec = tokens_seen / elapsed if elapsed > 0 else 0
                print(f"\r   Step {i}/{steps} | Tokens: {tokens_seen:,} | Speed: {tok_sec:.0f} tok/s", end="")
                
        print(f"\n\n✅ DataLoader test passed! Processed {tokens_seen:,} tokens without error.")
        print("   The 'tiktoken' fix is working correctly on real data.")
        
    except Exception as e:
        print(f"\n\n❌ DataLoader test FAILED with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    
    test_dataloader(steps=args.steps)
