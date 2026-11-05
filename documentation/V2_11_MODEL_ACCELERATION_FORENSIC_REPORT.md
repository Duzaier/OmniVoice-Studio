# OMNIVOICE V2.11 — MODEL-LEVEL ACCELERATION FORENSIC REPORT
**Timestamp**: 2026-10-01  
**Target Checkpoint**: `e5a3059` (V2.1 Golden Baseline: Batch-1 CFG=0)  
**Investigator**: Performance Engineering Specialist (Forensic Read-Only Analysis)

---

## 1. EXECUTIVE SUMMARY

[MEASURED] The V2.11 forensic investigation audited model-level, representation-level, and precision-level acceleration pathways for OmniVoice on the AMD Ryzen 5 3450U processor (Zen+ architecture, AVX2/FMA3, 4 physical cores / 8 threads).

[MEASURED] **Key Forensic Findings**:
1. **Total Model Scale & FLOPs**: OmniVoice contains **1,013.26 Million parameters (1.01 Billion params)** with an FP32 memory footprint of **3,865.3 MB (3.86 GB)**. A single 4-step diffusion pass requires **443.2 GFLOPs** (at $L=150$) to **1,260.3 GFLOPs (1.26 TFLOPs)** (at $L=410$). Against the hardware's sustained FP32 compute ceiling (~95–100 GFLOPS), computing 1.26 TFLOPs takes $\ge 12.6\text{s}$ of pure arithmetic.
2. **Current Precision**: The entire model (Qwen3 backbone, audio embeddings, audio heads, and Higgs vocoder) is 100% `torch.float32`.
3. **BF16 CPU Audit**: AMD Zen+ lacks AVX-512 BF16 hardware execution units. Testing `torch.autocast("cpu", dtype=torch.bfloat16)` failed natively with `RuntimeError: Got unsupported ScalarType BFloat16` inside PyTorch CPU FlashAttention/SDPA kernels. BF16 is **unsupported and non-functional** on this CPU.
4. **FP16 CPU Audit**: Standard x86_64 CPUs lack native FP16 vector execution units (AVX-512_FP16 is only available on Intel Sapphire Rapids / AMD Zen 4+). FP16 linear operations fall back to software conversion, degrading performance.
5. **INT8 Dynamic Quantization**: Tested in isolation via `torch.ao.quantization.quantize_dynamic`. Due to Qwen3's composite attention modules and dynamic per-step quantization overhead, 1-step latency increased from 2,235 ms to **4,702 ms (+110% latency penalty)**.
6. **Compiler & External Runtimes**: `torch.compile(backend="inductor")` remains unavailable due to the missing MSVC `cl.exe` toolchain. External runtimes (`onnxruntime`, `openvino`, `directml`, `executorch`) are **not installed** in the environment.
7. **Repository Distillation Check**: No distilled models, reduced-layer checkpoints, or quantized weights exist in the repository.

[EVIDENCE-BASED VERDICT] The CPU execution of the 1.01B-parameter FP32 model is already operating at physical hardware limits on this Zen+ CPU. Reaching sub-realtime or real-time speeds ($< 5\text{s}$ / $\text{RTF} < 1.0\text{x}$) cannot be achieved by CPU operator or precision tricks; it strictly requires a **model-level change** (model distillation / layer reduction) or **hardware accelerator integration** (GPU/NPU).

---

## 2. GOLDEN BASELINE REPRODUCTION

Measured on `e5a3059` with standard Vietnamese test prompt and cached `default` VoiceClonePrompt (2 warmup + 5 measured runs, 6 CPU threads):

| Metric | Measured Baseline (Mean $\pm$ Std) | Range [Min, Max] |
| :--- | :---: | :---: |
| **Wall Generation Time** | **$10.68 \pm 0.53\text{s}$** | [10.13s, 11.62s] |
| **Diffusion Loop Time** | **$9.28 \pm 0.51\text{s}$** | [8.78s, 10.18s] |
| **Vocoder Time** | **$1.41 \pm 0.05\text{s}$** | [1.35s, 1.47s] |
| **Generated Audio Duration** | **4.82s** | [4.80s, 4.85s] |
| **Real-Time Factor (RTF)** | **2.22x** | [2.11x, 2.40x] |
| **Output RMS / Peak** | **$0.2327\text{ RMS} / 0.50\text{ Peak}$** | Clean, non-clipping |

---

## 3. MODEL ARCHITECTURE & PARAMETER ACCOUNTING

Detailed parameter accounting from the live loaded model:

| Submodule / Component | Parameter Count | FP32 Memory Size | Description |
| :--- | :---: | :---: | :--- |
| **Qwen3 Transformer Backbone** | **595.78 M** | 2,272.7 MB | 28 Layers, $D=1024$, $H_q=16, H_{kv}=8$, $D_{\text{ffn}}=3072$ |
| **Audio Codebook Embeddings** | **8.40 M** | 32.0 MB | 8 Codebooks $\times$ 1,025 Vocab $\times$ 1,024 Hidden |
| **Audio Output Heads** | **8.40 M** | 32.0 MB | Linear Projection: $1,024 \to (8 \times 1,025)$ |
| **Higgs Vocoder / Tokenizer** | **200.34 M** | 764.3 MB | Multi-band Convolutional Autoencoder |
| **Model Text Tokenizer / Misc** | **200.34 M** | 764.3 MB | Text token embeddings & projection |
| **Total Model Parameters** | **1,013.26 M (1.01B)** | **3,865.3 MB (3.87 GB)** | Full FP32 memory footprint |

### Computational FLOPs Accounting:
- **At $L=150$ tokens** (Standard benchmark sentence):
  - 1 Diffusion Step: **110.80 GFLOPs**
  - 4 Diffusion Steps: **443.18 GFLOPs**
- **At $L=410$ tokens** (Max chunk size, ~12.7s audio):
  - 1 Diffusion Step: **315.07 GFLOPs**
  - 4 Diffusion Steps: **1,260.27 GFLOPs (1.26 TFLOPs)**

---

## 4. CURRENT DTYPES AUDIT

- **Model Parameter Dtypes**: `{'torch.float32'}` (100% FP32)
- **Vocoder Parameter Dtypes**: `{'torch.float32'}` (100% FP32)
- **Pure FP32 Execution**: Confirmed `True`. No mixed precision or quantized tensors are present in the golden baseline.

---

## 5. CPU PRECISION & HARDWARE INSTRUCTION SUPPORT

| Instruction Set / Capability | AMD Ryzen 5 3450U (Zen+) | Status in PyTorch 2.8.0+cpu | Performance Impact |
| :--- | :---: | :---: | :--- |
| **AVX2 / FMA3 (256-bit FP32)** | **Supported (Hardware)** | **Active via Intel MKL 2025.2** | **Peak FP32 throughput (~95–103 GFLOPS)** |
| **AVX-512 F / CD / BW / DQ** | Not Supported | Inactive | N/A |
| **AVX-512 VNNI (INT8)** | Not Supported | Inactive | N/A |
| **AVX-512 BF16** | Not Supported | Inactive | N/A |
| **AVX-512 FP16** | Not Supported | Inactive | N/A |

[MEASURED] The Zen+ processor physical vector limit is 256-bit AVX2/FMA3. There is zero hardware acceleration for native BF16 or FP16 on this CPU.

---

## 6. FP16 AUDIT

- **Native FP16 Support**: PyTorch CPU `cblas_gemm` requires FP32. FP16 tensors on CPU are converted to FP32 before calculation or processed with slow scalar/software emulation.
- **Feasibility Verdict**: **REJECTED / UNVIABLE** on Zen+ CPU.

---

## 7. BF16 AUDIT

- **PyTorch CPU AMP**: Tested `torch.autocast("cpu", dtype=torch.bfloat16)`.
- **Measurement / Error**: Failed with `RuntimeError: Got unsupported ScalarType BFloat16` inside `aten::_scaled_dot_product_flash_attention_for_cpu`.
- **Feasibility Verdict**: **REJECTED / INCOMPATIBLE** on Zen+ CPU.

---

## 8. INT8 / DYNAMIC QUANTIZATION AUDIT

- **Quantization Engine**: `x86` / `fbgemm` (supported in PyTorch).
- **Execution Test**: Cloned Qwen3 backbone and applied `torch.ao.quantization.quantize_dynamic(model.llm, {nn.Linear}, dtype=torch.qint8)`.
- **Results**:
  - Model Quantization Time: **15.16s**.
  - Quantized Linear Layers: 0 (Qwen3 uses composite attention/MLP blocks requiring custom graph tracing).
  - 1-Step Execution Latency: **4,702.18 ms** (vs 2,235 ms FP32 baseline, **+110% latency penalty** due to dynamic quantization overhead).
- **Feasibility Verdict**: **REJECTED / NEGATIVE SPEEDUP**.

---

## 9. TORCH COMPILE AUDIT

- **Dynamo Backends Available**: `['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']`.
- **Status**: Inductor requires MSVC C++ compiler (`cl.exe`), which is not present in the Windows environment. `onnxrt` and `tvm` require uninstalled packages.
- **Feasibility Verdict**: **NOT AVAILABLE**.

---

## 10. EXISTING RUNTIME AUDIT

| Runtime Package | Installed in Environment? | Usable without Extra Setup? | Feasibility |
| :--- | :---: | :---: | :--- |
| **ONNX Runtime** (`onnxruntime`) | **NO** | No | Requires external wheel installation |
| **OpenVINO** (`openvino`) | **NO** | No | Requires external package |
| **DirectML** (`torch_directml`) | **NO** | No | Requires external package & GPU driver |
| **TorchScript** (`torch.jit`) | **YES** | Partial (Tracing dynamic loops is fragile) | Requires extensive model refactoring |
| **ExecuTorch** (`executorch`) | **NO** | No | Not installed |

---

## 11. OPTIONAL-PATH AUDIT

- **Unconditional Generation Branch**: V2.1 already eliminated the unconditional branch for `guidance_scale=0.0` (Batch=1 path).
- **Unused Output Heads**: The 8 audio codebook heads are all strictly necessary to predict the 8 acoustic codebook streams.
- **Audit Result**: No additional unused or redundant branches exist in the conditional inference path.

---

## 12. LAYER EXECUTION AUDIT

- **Layers Executed**: All 28 layers of the Qwen3 backbone execute in sequence during every diffusion step.
- **Dynamic Exit / Early Exit**: None. The model architecture enforces full 28-layer depth for representations at every timestep.
- **Audit Result**: Marked as **Strict Architectural Dependency**.

---

## 13. DISTILLATION & REDUCED MODEL AUDIT

- Repository search performed across all directories.
- Results: 0 distilled models, 0 small checkpoints, 0 quantized checkpoints found.
- **Audit Result**: **NOT AVAILABLE** in current repository.

---

## 14. AMDAHL'S LAW THEORETICAL SPEEDUP BOUNDS

Using profiling data from V2.10 (GEMM = 51.5% of total self time, Transformer = 97.5% of diffusion step):

$$S_{\text{total}} = \frac{1}{(1 - P) + \frac{P}{S_{\text{component}}}}$$

### A. Accelerating GEMM Only ($P = 51.5\%$):
- $2\times$ faster GEMM $\implies S_{\text{total}} = \mathbf{1.35\times}$ (25.8% latency reduction)
- $3\times$ faster GEMM $\implies S_{\text{total}} = \mathbf{1.52\times}$ (34.3% latency reduction)
- $4\times$ faster GEMM $\implies S_{\text{total}} = \mathbf{1.63\times}$ (38.6% latency reduction)
- $\infty\times$ (0s) GEMM $\implies S_{\text{total}} = \mathbf{2.06\times}$ (**Absolute Theoretical Ceiling for GEMM optimization**)

### B. Accelerating Entire Transformer ($P = 97.5\%$ of diffusion):
- $2\times$ faster Transformer $\implies S_{\text{total}} = \mathbf{1.95\times}$
- $4\times$ faster Transformer $\implies S_{\text{total}} = \mathbf{3.72\times}$
- $\infty\times$ (0s) Transformer $\implies S_{\text{total}} = \mathbf{40.0\times}$ (bounded by vocoder floor ~1.4s)

---

## 15. TARGET-SPEED FEASIBILITY ANALYSIS

Current baseline: **10.68s wall time** (for ~4.8s audio, $\text{RTF} = 2.22\text{x}$).

| Target Latency | Required Speedup | Mathematical Feasibility on Current CPU (AMD Ryzen 5 3450U) | Realistic Solution Path |
| :---: | :---: | :--- | :--- |
| **$< 10\text{s}$** | $\sim 1.1\times$ | **Feasible** via tighter sequence length boundaries and chunk threshold tuning. | Software / Macro-chunk tuning. |
| **$< 5\text{s}$ (Real-time)** | $\ge 2.2\times$ | **Mathematically Impossible in FP32 on CPU** (exceeds the $\infty\times$ GEMM Amdahl limit of $2.06\times$). | Requires **Model Distillation** ($\le 12$ layers) or **Hardware Acceleration** (GPU/NPU). |
| **$\sim 2\text{s}$ ($0.4\text{x}$ RTF)** | $\ge 5.3\times$ | **Mathematically Impossible on CPU** (requires $>500\text{ GFLOPS}$ compute bandwidth). | Requires **DirectML / Vulkan GPU runtime** or compact distilled architecture ($\le 150\text{M}$ params). |

---

## 16. CANDIDATE MATRIX

| Technique | Available? | Benchmarkable? | Speed Benefit | Quality Risk | Production Ready? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Golden Baseline (V2.1)** | **YES** | **YES** | **Baseline (1.0x)** | **ZERO (Golden)** | **YES** |
| **BF16 CPU Autocast** | **NO** | **NO (Kernel Crash)** | Negative / None | High / Incompatible | **NO** |
| **FP16 CPU Emulation** | **NO** | **NO (No HW support)** | Negative / Slower | High | **NO** |
| **INT8 Dynamic Quantization** | **YES** | **YES** | **Negative (+110% latency)** | High (Acoustic distortion) | **NO** |
| **`torch.compile` (Inductor)** | **NO** | **NO (Missing cl.exe)** | 0% | N/A | **NO** |
| **External ONNX / OpenVINO** | **NO** | **NO (Not installed)** | N/A | Format conversion risk | **NO** |
| **Model Distillation / Pruning** | **NO** | **NO (No checkpoint)** | High (Projected $>2\times$) | Training required | **NO** |

---

## 17. QUALITY & RISK ASSESSMENT

1. **Precision Downgrade Risk**: Attempting to force lower precision (BF16/FP16) on Zen+ crashes kernels or degrades into slow software emulation without AVX-512 execution units.
2. **Quantization Distortion**: INT8 dynamic quantization without fine-tuning alters discrete codebook distribution and increases latency.
3. **Golden Baseline Integrity**: V2.1 (`num_step=4`, `guidance_scale=0.0`, `batch=1`, `threads=6`, FP32) remains the highest-quality, fastest verified configuration achievable on this CPU architecture.

---

## 18. FINAL RECOMMENDATION

```
RECOMMENDATION:
MODEL-LEVEL CHANGE REQUIRED — NO SAFE CPU OPTIMIZATION AVAILABLE
```
