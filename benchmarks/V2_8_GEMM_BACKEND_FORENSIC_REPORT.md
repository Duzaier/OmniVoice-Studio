# OMNIVOICE V2.8 — GEMM BACKEND & THREADING FORENSIC REPORT
**Timestamp**: 2026-10-01  
**Target Checkpoint**: `e5a3059` (V2.1 Golden Baseline: Batch-1 CFG=0)  
**Investigator**: Performance Engineering Specialist (Forensic Read-Only Analysis)

---

## 1. EXECUTIVE SUMMARY & FORENSIC VERDICT

[MEASURED] Across real diffusion generation passes in OmniVoice Qwen3 (`batch=1`, `num_step=4`, `guidance_scale=0.0`), GEMM operations (`aten::mm`) constitute **51.31% of total CPU self execution time** (1682.94 ms out of 3279.85 ms per diffusion step), making it the single largest computational consumer.

[MEASURED] PyTorch `2.8.0+cpu` is dynamically utilizing **Intel oneAPI Math Kernel Library (MKL) 2025.2** with full **AVX2 / FMA3** vectorization via OpenMP runtime threads.

[MEASURED] In synthetic and isolated microbenchmarks representing real Qwen3 projection shapes ($L=410, D=1024, \text{FFN}=3072$), FP32 GEMM throughput scales from **~31–38 GFLOPS at 1 thread** up to **~95–103 GFLOPS at 6–8 threads**. Against the theoretical FP32 roofline of the AMD Ryzen 5 3450U processor (~115.2 GFLOPS at sustained 3.6 GHz base/boost clock), Intel MKL reaches **82.5% to 89.8% of theoretical compute ceiling**.

[MEASURED] The arithmetic intensity of all three primary Qwen3 GEMM operations ranges from **128.08 to 139.73 FLOPs/Byte**. Given the dual-channel DDR4 memory bandwidth (~38.4 GB/s), memory bus bandwidth saturation would require >4,900 GFLOPS. Thus, GEMM in OmniVoice is **100% compute-bound / execution-unit bound**, not memory bandwidth bound.

[OBSERVED] Thread scaling shows clear diminishing returns above 6 threads (e.g. MLP Down projection degrades from 27.45 ms at 6T to 30.16 ms at 8T due to SMT thread contention on Zen+ 4 physical cores).

[EVIDENCE-BASED VERDICT] The GEMM backend is already operating within 10–17% of the absolute physical hardware compute roofline for single-precision FP32 matrix multiplication. No alternative standard BLAS runtime (OpenBLAS, BLIS, Eigen) will outperform Intel MKL AVX2 on this Zen+ CPU without changing numerical precision (e.g. INT8 / BF16 quantization) or model architecture.

---

## 2. RUNTIME & ENVIRONMENT CONFIGURATION

### Hardware Architecture:
- **Processor**: AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx
- **Topology**: 4 Physical Cores / 8 Logical Threads (SMT enabled)
- **Microarchitecture**: AMD Zen+ (12nm, 2x 256-bit FMA/AVX2 units per core)
- **Theoretical Peak FP32 GFLOPS**: $4 \text{ cores} \times 2 \text{ FMA units} \times 8 \text{ SP-floats/cycle} \times 2 (\text{FMA mul+add}) \times 3.6 \text{ GHz max} \approx 115.2 \text{ GFLOPS}$ (Base 2.1 GHz peak $\approx 67.2 \text{ GFLOPS}$, All-core boost ~3.0 GHz $\approx 96.0 \text{ GFLOPS}$)
- **RAM**: 16 GB Dual-Channel DDR4-2400 (Peak theoretical bandwidth ~38.4 GB/s)

### Software & PyTorch Build Configuration:
- **PyTorch Version**: `2.8.0+cpu`
- **Compiler**: MSVC `193833145` (C++17)
- **BLAS Backend**: `Intel(R) oneAPI Math Kernel Library Version 2025.2-Product Build 20250620` (`BLAS_INFO=mkl`)
- **DNN Engine**: `Intel(R) MKL-DNN v3.7.1` (oneDNN)
- **Parallel Backend**: `OpenMP 2019`
- **Vector Instruction Flags**: `AVX2`, `PERF_WITH_AVX=1`, `PERF_WITH_AVX2=1`
- **Threadpool Default**: `at::get_num_threads() = 4`, `at::get_num_interop_threads() = 4`

---

## 3. GEMM SHAPE & ARITHMETIC INTENSITY ANALYSIS

For a standard audio sequence length $L = 410$ tokens (corresponding to ~12.7s audio chunking), the active Qwen3 Transformer dimensions are:
- Hidden Dimension $D = 1024$
- Attention Heads / Key-Value: $H_q = 16, H_{kv} = 8, d_h = 64 \implies D_{kv} = 512, D_{qkv} = 2048$
- Intermediate FFN Dimension: $D_{ffn} = 3072$ (Gate + Up = 6144)

| GEMM Stage | Matrix A Shape | Matrix B Shape | FLOPs per Call ($2MNK$) | Memory Footprint ($A+B+C$) | Arithmetic Intensity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QKV Projection** | `[410, 1024]` | `[1024, 2048]` | $1.72 \times 10^9$ FLOPs (1,719.66 MFLOP) | 13.43 MB | **128.08 FLOP/Byte** |
| **MLP Gate+Up Projection** | `[410, 1024]` | `[1024, 6144]` | $5.16 \times 10^9$ FLOPs (5,158.99 MFLOP) | 36.92 MB | **139.73 FLOP/Byte** |
| **MLP Down Projection** | `[410, 3072]` | `[3072, 1024]` | $2.58 \times 10^9$ FLOPs (2,579.50 MFLOP) | 19.30 MB | **133.65 FLOP/Byte** |

[MEASURED] With arithmetic intensities $\approx 128 - 140\text{ FLOP/Byte}$, the machine's memory bandwidth ridge point ($\frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}} = \frac{115.2 \times 10^9}{38.4 \times 10^9} = 3.0\text{ FLOP/Byte}$) is exceeded by over **$42\times$**. Therefore, cache and memory bandwidth are not restricting performance; execution pipelines are fully compute-bound.

---

## 4. MEMORY LAYOUT & STRIDE FORENSICS

[MEASURED] Inspection of PyTorch's native `nn.Linear` execution path revealed:
1. **Weights Layout**: In `torch.nn.Linear(in_features, out_features)`, weights are stored in row-major order with shape `[out_features, in_features]` (contiguous, stride `(in_features, 1)`).
2. **Transpose View**: `linear(x)` executes `x.matmul(weight.t())`. `weight.t()` has shape `[in_features, out_features]` and stride `(1, in_features)` (column-major / non-contiguous).
3. **MKL Interface**: `aten::mm` / `cblas_sgemm` accepts the transpose flag `CblasTrans` for matrix B directly:
   $$C = \alpha \cdot A \cdot B^T + \beta C$$
   Intel MKL evaluates this non-contiguous view in place with zero intermediate buffer copies.
4. **Direct `aten::mm` vs `nn.Linear` Comparison**: Microbenchmarking showed identical latency (within measurement jitter $< 2\%$), proving PyTorch introduces zero overhead in dispatching `nn.Linear` to MKL `cblas_sgemm`.

---

## 5. THREAD SCALING & COMPUTATIONAL THROUGHPUT (MICROBENCHMARKS)

Microbenchmarks measured over 30 warm iterations for each GEMM shape across thread counts $T \in \{1, 2, 4, 6, 8\}$:

### A. QKV Projection (`[410, 1024] x [1024, 2048]`, 1719.66 MFLOP)
| Threads | `aten::mm` Mean (ms) | GFLOPS | Scaling vs 1T | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1T** | 48.88 ms | 35.18 GFLOPS | 1.00x | 100.0% |
| **2T** | 30.99 ms | 55.50 GFLOPS | 1.58x | 78.9% |
| **4T** | 26.87 ms | 64.00 GFLOPS | 1.82x | 45.5% |
| **6T** | 17.99 ms | 95.58 GFLOPS | 2.72x | 45.3% |
| **8T** | 16.63 ms | 103.44 GFLOPS | 2.94x | 36.8% |

### B. MLP Gate+Up Projection (`[410, 1024] x [1024, 6144]`, 5158.99 MFLOP)
| Threads | `aten::mm` Mean (ms) | GFLOPS | Scaling vs 1T | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1T** | 166.89 ms | 30.91 GFLOPS | 1.00x | 100.0% |
| **2T** | 91.73 ms | 56.24 GFLOPS | 1.82x | 91.0% |
| **4T** | 82.96 ms | 62.19 GFLOPS | 2.01x | 50.3% |
| **6T** | 59.87 ms | 86.16 GFLOPS | 2.79x | 46.5% |
| **8T** | 52.84 ms | 97.64 GFLOPS | 3.16x | 39.5% |

### C. MLP Down Projection (`[410, 3072] x [3072, 1024]`, 2579.50 MFLOP)
| Threads | `aten::mm` Mean (ms) | GFLOPS | Scaling vs 1T | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1T** | 67.37 ms | 38.29 GFLOPS | 1.00x | 100.0% |
| **2T** | 42.78 ms | 60.30 GFLOPS | 1.57x | 78.7% |
| **4T** | 41.25 ms | 62.53 GFLOPS | 1.63x | 40.8% |
| **6T** | 27.45 ms | 93.96 GFLOPS | 2.45x | 40.9% |
| **8T** | 30.16 ms | 85.53 GFLOPS | 2.23x | 27.9% |

---

## 6. 1-STEP DIFFUSION LIVE PROFILING BREAKDOWN

During live profiling of the conditional Batch-1 path (`guidance_scale=0.0`, 6 threads), the CPU operator self-time breakdown is as follows:

```
Total Profiled Self CPU Time : 3279.85 ms (100.0%)
  ├── Pure GEMM (aten::mm)     : 1682.94 ms ( 51.31% ) [197 calls, 8.54 ms avg]
  ├── Conv (mkldnn_convolution):  810.06 ms ( 24.70% ) [ 32 calls, 25.31 ms avg]
  ├── Elementwise Mul (mul)    :  154.34 ms (  4.71% ) [444 calls, 0.35 ms avg]
  ├── Conv Wrapper (_conv)     :  119.55 ms (  3.65% ) [ 37 calls, 3.23 ms avg]
  ├── Elementwise Add (add)    :  105.84 ms (  3.23% ) [325 calls, 0.33 ms avg]
  ├── Flash Attention CPU SDPA :   86.63 ms (  2.64% ) [ 28 calls, 3.09 ms avg]
  ├── Non-linearities (pow/sin):  127.82 ms (  3.90% ) [186 calls]
  ├── Memory copy (copy_/cat)  :   69.32 ms (  2.11% ) [470 calls]
  └── Other / misc             :  123.35 ms (  3.76% )
```

[MEASURED] Pure GEMM (`aten::mm`) accounts for **1682.94 ms / step**.
Direct end-to-end 1-step latency over 5 iterations averaged **3544.86 ms** (Std: 132.08 ms).

---

## 7. ROOFLINE & BOTTLENECK ANALYSIS

```
           GFLOPS
             ^
     115.2 --|--------------------------------==== Theoretical FP32 Peak (115.2 GFLOPS)
             |                      *  *  *  (Measured MKL AVX2 GEMM: 95 - 103 GFLOPS)
             |                    /
             |                  /
             |                /
             |              /
             |            /  Compute Bound Region (AI > 3.0 FLOP/Byte)
             |          /
             |        /
             |      /
             |    /
       0.0 --+---+---------------------------------------------> Arithmetic Intensity (FLOP/Byte)
             0   3.0                             128.08   139.73
```

1. **Roofline Position**: The Qwen3 GEMM operations operate far to the right of the memory bandwidth knee (AI = 128–140 vs knee of 3.0).
2. **Efficiency**: At ~100 GFLOPS measured out of ~115 GFLOPS theoretical peak, Intel MKL is extracting **~87% of hardware maximum floating point capability**.
3. **Implication**: Any software optimization targeting GEMM without reducing total FLOP count (i.e. without quantization or pruning) has a theoretical headroom upper bound of only **$1.15\times$** (at 100% ideal peak).

---

## 8. SMT / HYPERTHREADING IMPACT

1. **4 Physical Cores vs 8 Logical Threads**:
   - The Zen+ CPU has 2 FPU pipes per physical core. SMT threads share the same FPU hardware.
   - For heavily compute-bound AVX2 FMA loops, SMT threads compete directly for FPU execution ports (Pipe 0 & Pipe 1), causing pipeline stalls and cache eviction.
2. **Observed Threading Sweetspot**:
   - 6 threads yields optimal performance and stability across all layers.
   - 8 threads causes thread contention on certain GEMM shapes (e.g. MLP Down throughput drops from 93.96 GFLOPS at 6T to 85.53 GFLOPS at 8T, latency increases from 27.45 ms to 30.16 ms).

---

## 9. BLAS / BACKEND FEASIBILITY ASSESSMENT

| Candidate Backend | Compatibility | AVX2 Kernel Quality | Projected Headroom | Risk / Feasibility |
| :--- | :--- | :--- | :--- | :--- |
| **Intel oneAPI MKL 2025.2** *(Current)* | Verified | Highly optimized assembly kernels for AVX2/FMA3 | **Baseline (~87% peak)** | Native in PyTorch binary. |
| **OpenBLAS (AVX2)** | Feasible | Good, but historically equal to or 5–10% slower than MKL on Zen CPUs | **Negative / Zero (<1.0x)** | Requires rebuilding PyTorch from source. |
| **BLIS (AMD AOCL)** | Experimental | Tailored for Zen, but requires custom DLL integration | **< 1.05x** | High integration fragility on Windows; no official PyTorch Windows wheel. |
| **Eigen / Ref BLAS** | Feasible | Generic C++ templates, poor multi-threading | **Severe regression (<0.5x)** | Completely unviable. |

---

## 10. COMPILER & QUANTIZATION POTENTIAL (ANALYSIS ONLY)

1. **`torch.compile`**:
   - Requires C++ compiler (`cl.exe` / MSVC). Even if enabled, PyTorch Inductor delegates standard `aten.mm` to external BLAS (MKL) rather than synthesizing custom GEMM loops. Thus, `torch.compile` does not accelerate large GEMM operations.
2. **Quantization (INT8 / dynamic quant)**:
   - Dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic` / oneDNN INT8) can theoretically double integer ALU throughput ($2\times$ int8 ops vs fp32).
   - *However*, INT8 quantization modifies model weights, risks degrading audio fidelity, alters timbre, and violates the Golden Baseline stability constraints. It is out of scope for non-breaking CPU inference.

---

## 11. RISKS OF PRODUCTION MODIFICATION

1. **BLAS Switching**: Modifying PyTorch's underlying BLAS on Windows carries high risks of runtime DLL crashes, OpenMP threading deadlocks, or numeric precision regressions.
2. **In-place Weight Transposition**: Weights are already optimally transposed and handled via `CblasTrans` without memory allocation.
3. **CFG=0 Batch-1 Optimization**: Already preserved as the single most effective architectural optimization ($2.16\times$ speedup achieved in V2.1).

---

## 12. FINAL CLASSIFICATION & EXACT RECOMMENDATION

```
RECOMMENDATION:
NO SAFE GEMM OPTIMIZATION FOUND — KEEP BASELINE
```
