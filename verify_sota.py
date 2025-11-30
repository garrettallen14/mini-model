import torch
from train_curriculum_cuda import create_model
import tiktoken

def verify_model():
    print("Verifying 150M SOTA Model...")
    try:
        model = create_model("150M")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameter count: {num_params:,}")
        
        # Expected ~156M
        if 150_000_000 <= num_params <= 160_000_000:
            print("✓ Parameter count looks correct.")
        else:
            print(f"✗ Parameter count unexpected: {num_params:,}")

        # Forward pass
        x = torch.randint(0, 50257, (1, 128))
        y = model(x)
        print(f"✓ Forward pass successful. Output shape: {y.shape}")
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        import traceback
        traceback.print_exc()

def verify_tiktoken():
    print("\nVerifying Tiktoken Fix...")
    enc = tiktoken.get_encoding("gpt2")
    text = "This is a test with ](cascade:incomplete-link) special token."
    try:
        # This mimics the fix in the code
        tokens = enc.encode(text, disallowed_special=())
        print(f"✓ Tiktoken encoding successful. Tokens: {tokens}")
    except Exception as e:
        print(f"✗ Tiktoken encoding failed: {e}")

if __name__ == "__main__":
    verify_model()
    verify_tiktoken()
