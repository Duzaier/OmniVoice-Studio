"""
OmniVoice V2.5 - torch.compile Safe Diagnostic
Testing whether torch.compile / Inductor / Triton is supported and viable on Windows CPU (Python 3.13)
"""

import sys
import time
import torch
import torch.nn as nn

def test_torch_compile_diagnostic():
    print("=== torch.compile Environment Diagnostic ===", flush=True)
    print(f"PyTorch Version: {torch.__version__}", flush=True)
    print(f"Python Version: {sys.version}", flush=True)
    print(f"Platform: {sys.platform}", flush=True)
    
    # 1. Test basic torch.compile availability
    if not hasattr(torch, "compile"):
        print("[-] torch.compile is NOT available in this PyTorch build.", flush=True)
        return

    print("[+] torch.compile API exists.", flush=True)
    
    # 2. Test compiling a simple Linear + GELU/SiLU module
    class SimpleBlock(nn.Module):
        def __init__(self, d=1024, ffn=3072):
            super().__init__()
            self.linear1 = nn.Linear(d, ffn)
            self.linear2 = nn.Linear(ffn, d)
            self.act = nn.SiLU()
        def forward(self, x):
            return self.linear2(self.act(self.linear1(x)))

    block = SimpleBlock()
    x = torch.randn(1, 410, 1024)
    
    print("\n--- Testing compilation on SimpleBlock ---", flush=True)
    try:
        t0 = time.perf_counter()
        compiled_block = torch.compile(block, backend="inductor")
        # Run first time (triggers compilation)
        out1 = compiled_block(x)
        t_compile = (time.perf_counter() - t0) * 1000.0
        print(f"[+] Compiled with Inductor successfully in {t_compile:.2f} ms", flush=True)
        
        # Benchmark uncompiled vs compiled
        def bench(f, iters=20):
            for _ in range(5):
                f(x)
            t_start = time.perf_counter()
            for _ in range(iters):
                f(x)
            return ((time.perf_counter() - t_start) / iters) * 1000.0
            
        ms_eager = bench(block)
        ms_compiled = bench(compiled_block)
        print(f"  • Eager latency: {ms_eager:.3f} ms")
        print(f"  • Compiled latency: {ms_compiled:.3f} ms (Speedup: {ms_eager / ms_compiled:.2f}x)")
    except Exception as e:
        print(f"[-] Inductor compilation failed or unsupported on Windows CPU: {type(e).__name__}: {e}", flush=True)

if __name__ == "__main__":
    test_torch_compile_diagnostic()
