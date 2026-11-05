import os
import sys
import inspect
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice

def main():
    print("Loading model for inspection...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    
    # Inspect model architecture
    print("Model structure:", flush=True)
    for name, child in model.named_children():
        print(f"  Child: {name} -> {type(child)}", flush=True)
        
    # Find transformer layers and MLP
    mlp_module = None
    for name, mod in model.named_modules():
        if "mlp" in name.lower() and hasattr(mod, "forward"):
            print(f"\nFound MLP module at '{name}': {type(mod)}", flush=True)
            mlp_module = mod
            break
            
    if mlp_module:
        print("\nMLP Submodules/Parameters:", flush=True)
        for subname, param in mlp_module.named_parameters():
            print(f"  param: {subname} shape={list(param.shape)} dtype={param.dtype}", flush=True)
        for subname, child in mlp_module.named_children():
            print(f"  child: {subname} -> {type(child)}", flush=True)
            
        print("\nMLP Class Source Code:", flush=True)
        try:
            print(inspect.getsource(type(mlp_module)), flush=True)
        except Exception as e:
            print("Could not get source:", e, flush=True)

if __name__ == "__main__":
    main()
