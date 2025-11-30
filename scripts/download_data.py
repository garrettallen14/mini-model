#!/usr/bin/env python3
"""
Download and validate datasets for mini-model training.

Based on datas.md "Your Exact Shopping List":
- TinyStories (476M tokens) - required for Phase 1
- Cosmopedia (274M tokens) - Phase 1-3
- OpenWebMath (800M tokens) - Phase 2-3
- MetaMathQA (250M tokens) - Phase 2-3
- StarCoder Python (900M tokens) - Phase 2-3
- Dolma Books (300M tokens) - Phase 3

Usage:
    python scripts/download_data.py              # Download TinyStories (quick start)
    python scripts/download_data.py --all        # Download all datasets
    python scripts/download_data.py --validate   # Validate existing downloads
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_tinystories():
    """Download TinyStories - essential for Phase 1."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Downloading TinyStories (476M tokens)")
    print("=" * 60)
    
    ds = load_dataset("roneneldan/TinyStories", split="train")
    print(f"✓ Downloaded {len(ds):,} stories")
    
    # Quick stats
    sample = ds[0]
    print(f"  Sample text: {sample['text'][:100]}...")
    
    return ds


def download_metamathqa():
    """Download MetaMathQA - small, loads fully into memory."""
    from datasets import load_dataset
    
    print("\n" + "=" * 60)
    print("Downloading MetaMathQA (250M tokens)")
    print("=" * 60)
    
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    print(f"✓ Downloaded {len(ds):,} examples")
    
    sample = ds[0]
    print(f"  Sample query: {sample['query'][:80]}...")
    
    return ds


def validate_streaming(name: str, path: str, data_dir: str = None, text_col: str = "text"):
    """Validate a streaming dataset by reading first few examples."""
    from datasets import load_dataset
    
    print(f"\nValidating {name}...")
    
    kwargs = {"path": path, "split": "train", "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    
    try:
        ds = load_dataset(**kwargs)
        
        # Read first 5 examples
        count = 0
        total_chars = 0
        for example in ds:
            if text_col in example and example[text_col]:
                total_chars += len(example[text_col])
                count += 1
            if count >= 5:
                break
        
        if count > 0:
            print(f"  ✓ {name}: {count} examples read, avg {total_chars/count:.0f} chars")
            return True
        else:
            print(f"  ✗ {name}: No valid examples found")
            return False
            
    except Exception as e:
        print(f"  ✗ {name}: Error - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets for mini-model")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--validate", action="store_true", help="Validate streaming datasets")
    parser.add_argument("--tinystories", action="store_true", help="Download TinyStories only")
    parser.add_argument("--metamath", action="store_true", help="Download MetaMathQA only")
    
    args = parser.parse_args()
    
    # Default: just TinyStories
    if not any([args.all, args.validate, args.tinystories, args.metamath]):
        args.tinystories = True
    
    print("=" * 60)
    print("Mini-Model Dataset Downloader")
    print("=" * 60)
    
    if args.tinystories or args.all:
        download_tinystories()
    
    if args.metamath or args.all:
        download_metamathqa()
    
    if args.validate or args.all:
        print("\n" + "=" * 60)
        print("Validating Streaming Datasets")
        print("=" * 60)
        
        validations = [
            ("OpenWebMath", "open-web-math/open-web-math", None, "text"),
            ("Cosmopedia", "HuggingFaceTB/cosmopedia", None, "text"),
            ("StarCoder Python", "bigcode/starcoderdata", "python", "content"),
        ]
        
        results = []
        for name, path, data_dir, text_col in validations:
            ok = validate_streaming(name, path, data_dir, text_col)
            results.append((name, ok))
        
        # Dolma requires agreement, might fail
        print("\nNote: Dolma Books requires accepting terms at huggingface.co/datasets/allenai/dolma")
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for name, ok in results:
            status = "✓" if ok else "✗"
            print(f"  {status} {name}")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Test data pipeline:")
    print("   ./venv/bin/python scripts/test_data_pipeline.py")
    print("")
    print("2. (Optional) Full validation:")
    print("   ./venv/bin/python scripts/download_data.py --validate")
    print("")


if __name__ == "__main__":
    main()
