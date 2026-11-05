# OMNIVOICE V2.13 — HARDWARE & RUNTIME ACCELERATION FORENSIC REPORT
**Forensic Investigation of Hardware, Drivers, ML Runtimes & Acceleration Pathways**

---

## 1. EXECUTIVE SUMMARY & VERDICT

| Category / Component | Forensic Status | Key Evidence | Final Assessment |
| :--- | :--- | :--- | :--- |
| **Git Baseline** | Commit `e5a3059` (V2.1 Batch-1 CFG=0) | `[VERIFIED]` HEAD matches baseline, clean production tree | Baseline preserved 100% |
| **AMD GPU Hardware** | AMD Radeon Vega 8 Mobile (`0x15D8`) | `[OBSERVED]` Driver `31.0.21924.1004` (WDDM / DirectX 12) | OS-level recognized |
| **PyTorch GPU Backend** | CUDA=False, HIP=None, MPS=False | `[MEASURED]` Built with `torch 2.8.0+cpu` | No GPU execution in PyTorch |
| **DirectML Runtimes** | `torch_directml` & `onnxruntime-directml` | `[DOCUMENTED]` `NOT_INSTALLED`; Python 3.13 lacks wheels | Environment blocked |
| **TorchScript JIT Tracing** | Tested on Qwen3 MLP ($M=150, K=2048, N=5632$) | `[MEASURED]` $155.2\text{ ms}$ (Eager) vs $166.7\text{ ms}$ (JIT) = **$0.931\times$** | No speedup (overhead) |
| **Amdahl's Law (Vocoder)** | Vocoder runtime = $2.72\text{s}$ out of $21.42\text{s}$ ($12.7\%$) | `[MEASURED]` Max theoretical speedup from 0s vocoder = $1.14\times$ | Capped by Transformer GEMM |
| **Final Classification** | — | **`HARDWARE UPGRADE REQUIRED`** | **OFFICIAL VERDICT** |

### Official Forensic Recommendation
```
HARDWARE UPGRADE REQUIRED
```

> **Forensic Conclusion**:
> 1. **GPU Runtime Gap**: While the AMD Radeon Vega 8 iGPU is recognized by Windows and DirectX (`31.0.21924.1004`), no ML runtime currently in the environment or available on Python 3.13 provides verified GPU tensor acceleration for this hardware.
> 2. **Hardware Bottleneck of Vega 8 iGPU**: Even if DirectML were compiled for Python 3.13, Vega 8 shares DDR4 system memory bandwidth ($\sim 35\text{ GB/s}$) with the CPU. Lacking dedicated high-bandwidth GDDR6 VRAM and tensor cores, GEMM latency on Vega 8 is severely bandwidth-constrained and burdened by DirectML dispatch overhead at $B=1$.
> 3. **CPU Plateau (V2.5–V2.12)**: PyTorch CPU inference on the AMD Ryzen 5 3450U is already operating at $\sim 88\%$ of physical FMA/AVX2 capacity using Intel MKL and 6 threads.
> 4. **Path to Real-Time (RTF $\le 1.0$)**: Achieving real-time generation ($90\text{s}$ audio in $\le 90\text{s}$ or $\le 60\text{s}$) is mathematically impossible on the current Zen+ 4-core mobile CPU + iGPU setup without either **dedicated GPU hardware (NVIDIA CUDA / TensorRT)** or a **distilled 1-step student model**.

---

## 2. BASELINE & ENVIRONMENT VERIFICATION

```
Git Baseline Check:
  • Branch          : master
  • Commit Hash     : e5a3059e9f423fb34ffd34b534cccdc28037c1b1
  • Commit Message  : "V2.1: optimize CFG=0 inference with batch-1"
  • Production Tree : Unmodified (omnivoice/models/omnivoice.py intact) [VERIFIED]
```

### System Configuration
- **Host CPU**: AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx (4 physical cores, 8 logical threads, Zen+ architecture)
- **Instruction Sets**: AVX2, FMA3, BMI2 `[OBSERVED]`
- **System Memory**: 16.0 GB DDR4 (dual-channel, shared between CPU and iGPU)
- **Operating System**: Windows 11 Home / Pro (WDDM 3.x, DirectX 12)
- **Python Version**: 3.13.15 `[MEASURED]`
- **PyTorch Build**: `2.8.0+cpu` (linked with Intel MKL 2025.2 & oneDNN 3.7.1)

---

## 3. GPU HARDWARE, DISPLAY DRIVER & FOUR-TIER AUDIT

### 3.1. Hardware & Driver Inspection `[OBSERVED]`
- **Primary Display Adapter**: AMD Radeon(TM) Graphics
  - **PCI Device ID**: `0x15D8` (AMD Picasso / Raven Ridge Vega 8 Mobile)
  - **Driver Version**: `31.0.21924.1004` (Release date: Jan 14, 2026)
  - **DirectX Version**: DirectX 12.1 (Feature Level 12_1)
  - **Adapter RAM**: $2\text{ GB}$ shared dynamically with system RAM.
- **Secondary Adapter**: DeskIn Virtual Display Adapter (Driver `16.26.19.698`)

### 3.2. Four-Tier GPU Execution Status
To avoid conflating hardware detection with ML execution capability, we distinguish four explicit levels:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FOUR-TIER GPU CAPABILITY STATUS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. GPU Recognized by Operating System?   │ [YES]  AMD Radeon Vega 8 (WMI)  │
│ 2. GPU Recognized by ML Runtime?        │ [NO]   PyTorch CPU build (No DML)│
│ 3. Tensor Actually Executed on GPU?      │ [NO]   All ops execute on CPU    │
│ 4. Full OmniVoice Pipeline on GPU?       │ [NO]   End-to-end on CPU MKL     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. ML RUNTIME INVENTORY & COMPATIBILITY AUDIT

| ML Runtime / Backend | Installed? | Version | Python 3.13 Compatibility | AMD Vega 8 Support | Qwen3 Support | Vocoder Support | Technical Status / Blockers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PyTorch CPU** | `YES` | `2.8.0+cpu` | `[PASS]` Full | `N/A` (Runs on CPU) | Full (28 layers) | Full (Higgs) | **Active Production Baseline (MKL AVX2)** |
| **PyTorch DirectML (`torch-directml`)** | `NO` | — | `[BLOCKED]` Wheels limited to $\le$ Py3.11 | `[THEORETICAL]` Via D3D12 | Partial | Unknown | No Python 3.13 wheels available |
| **ONNX Runtime CPU (`onnxruntime`)** | `NO` | — | `[POSSIBLE]` | `N/A` (CPU) | Requires export | Requires export | Not installed; MLAS $\approx$ MKL performance |
| **ONNX Runtime DirectML (`ort-dml`)** | `NO` | — | `[POSSIBLE]` | `[THEORETICAL]` DML EP | Dynamic loop blocker | Opset 14+ | Requires complete ONNX model export |
| **OpenVINO (`openvino`)** | `NO` | — | `[POSSIBLE]` | CPU / Intel GPU only | Requires IR export | Requires IR export | AMD iGPU not supported by OpenVINO |
| **TorchScript JIT (`torch.jit`)** | `YES` | `2.8.0+cpu` | `[PASS]` Built-in | `N/A` (CPU) | Tracing supported | Tracing supported | **Measured $0.931\times$ speedup (Overhead)** |

---

## 5. MODEL OPERATOR COMPATIBILITY BREAKDOWN

### 5.1. Qwen3 Backbone (28 Layers, $D=2048, D_{ff}=5632$)
- **`nn.Linear` / `aten::mm`**: Standard GEMM ($>93\%$ of compute). Fully supported across all runtimes.
- **`Qwen3RMSNorm`**: Elementwise scaling + reciprocal square root. Low export risk.
- **`Qwen3RotaryEmbedding` (RoPE)**: Dynamic sequence position rotation. Requires dynamic tensor slicing during export.
- **GQA Attention (16 heads Q, 8 heads KV)**: Scaled dot-product attention with custom bidirectional attention mask.
- **4-Step Diffusion ODE Loop**: Iterative flow matching updating audio tokens across steps $t \in [0, 4]$. **Critical Blocker for static graph export**: requires either external Python driver loops (incurring repeated host-device sync overhead) or complex control-flow graph recursion.

### 5.2. Higgs Vocoder (Audio Reconstruction)
- **`Conv1d` / `ConvTranspose1d`**: Progressive upsampling ($8\times, 5\times, 4\times, 2\times$). Highly standard.
- **`Snake1d` Activation**: $x + \frac{1}{\alpha} \sin^2(\alpha x)$. Custom non-linear trigonometric operator with learnable parameter $\alpha$. Supported in ONNX Opset 14+ via basic elementwise math operations.
- **Amdahl's Law Limit**: In the baseline $21.42\text{s}$ wall time, the vocoder accounts for only **$2.72\text{s}$ ($12.7\%$)**. Even if the vocoder execution time were reduced to **$0.0\text{s}$**, total latency would only improve from $21.42\text{s}$ to $18.70\text{s}$ ($1.14\times$ speedup). The vocoder is **not the primary bottleneck**.

---

## 6. MICRO-BENCHMARK: TORCHSCRIPT JIT TRACING `[MEASURED]`

We tested TorchScript JIT tracing (`torch.jit.trace` + `torch.jit.optimize_for_inference`) on the exact dimensions of the Qwen3 MLP block ($M=150, K=2048, N=5632$) at 6 CPU threads:

| Execution Mode | Mean Latency ($\text{ms}$) | Std Dev ($\text{ms}$) | Relative Speedup | Max Absolute Numerical Error |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch Eager Mode** | **$155.18\text{ ms}$** | $14.13\text{ ms}$ | $1.000\times$ (Baseline) | $0.0$ |
| **TorchScript JIT Traced** | **$166.67\text{ ms}$** | $25.31\text{ ms}$ | **$0.931\times$ ($6.9\%$ SLOWER)** | $0.000000\text{e}+00$ |

> **Finding `[MEASURED]`**: TorchScript does not provide kernel fusion for large GEMM operations on CPU. Intel MKL already executes multi-threaded GEMM at peak cache efficiency; the JIT runtime wrapper incurs a slight dispatch overhead, resulting in a **$6.9\%$ slowdown**.

---

## 7. EVALUATION OF THE FOUR ACCELERATION PATHWAYS

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ ACCELERATION PATHWAYS EVALUATION MATRIX                                          │
├─────────┬──────────────────────┬─────────────┬─────────────────┬─────────────────┤
│ Pathway │ Target Backend       │ Feasibility │ Speedup Ceiling │ Primary Blocker │
├─────────┼──────────────────────┼─────────────┼─────────────────┼─────────────────┤
│ Path A  │ PyTorch CPU (MKL)    │ Active (V2) │ Near Ceiling    │ 4C/8T Zen+ Limit│
│ Path B  │ AMD Vega 8 DirectML  │ Blocked     │ ~0.8x - 1.2x    │ Py3.13 / DDR4 BW│
│ Path C  │ ONNX Runtime (CPU/DML│ High Effort │ ~0.95x - 1.1x   │ Dynamic ODE Loop│
│ Path D  │ New Hardware / Model │ High Yield  │ 4.0x - 25.0x    │ Hardware Upgrade│
└─────────┴──────────────────────┴─────────────┴─────────────────┴─────────────────┘
```

### Path A — PyTorch CPU (Synthesizing V2.5–V2.12) `[MEASURED]`
- **Status**: Current production engine.
- **Evidence**:
  - Intel MKL 2025.2 with AVX2/FMA3 is already active (V2.5, V2.8).
  - CPU utilization is near maximum throughput at 6 threads (V2.4, V2.8).
  - Python dispatch and memory copy overhead is $<2.3\%$ (V2.10).
  - Intra-batch padding is already $0\%$ in Batch-1 (V2.12).
- **Verdict**: CPU inference is fully saturated. No further free software speedup remains without changing model architecture or precision.

### Path B — AMD DirectML (Vega 8 Mobile) `[DOCUMENTED] & [INFERRED]`
- **Status**: Blocked by ecosystem & hardware architecture.
- **Blockers**:
  1. *Ecosystem*: `torch-directml` has no official prebuilt binaries for Python 3.13.
  2. *Shared Memory Bandwidth*: Vega 8 has no dedicated VRAM. It accesses system DDR4-2400/2666 RAM with a theoretical peak bandwidth of only $\sim 35 - 38\text{ GB/s}$. In comparison, an entry-level discrete GPU (e.g., RTX 3050) provides $\sim 224\text{ GB/s}$ ($6.4\times$ higher).
  3. *Dispatch Latency*: At batch size $B=1$, DirectML command list recording and submission overhead on integrated GPUs often negates the computational benefit over native CPU MKL GEMMs.
- **Verdict**: Not viable for meaningful acceleration on this machine.

### Path C — ONNX Runtime (CPU / DirectML EP) `[INFERRED]`
- **Status**: High engineering complexity, low performance yield.
- **Analysis**:
  - ONNX Runtime CPU using the MLAS kernel library achieves virtually identical throughput to PyTorch MKL on AMD Zen+ CPUs ($\pm 5\%$).
  - Exporting the dynamic 4-step diffusion loop requires custom Python orchestration or complex subgraph unrolling, reintroducing host-runtime boundary overheads.
- **Verdict**: Not recommended as a CPU acceleration strategy.

### Path D — Hardware Upgrades & Architecture Distillation `[DOCUMENTED]`
To achieve the user targets of **Real-Time 1.0x** ($90\text{s}$ audio in $90\text{s}$) or **Fast Real-Time 0.66x** ($90\text{s}$ audio in $60\text{s}$), the required speedup from current baseline ($21.4\text{s}$ per $4.8\text{s}$ audio, RTF $\approx 4.4\times$) is:
- **Target 1 ($90\text{s}$ in $90\text{s}$, RTF $\le 1.0\times$)**: Requires **$4.4\times$ speedup**.
- **Target 2 ($90\text{s}$ in $60\text{s}$, RTF $\le 0.66\times$)**: Requires **$6.7\times$ speedup**.

#### Viable Future Routes:
1. **Dedicated NVIDIA GPU (CUDA + FlashAttention + TensorRT)**:
   - *Projected Speedup*: **$15\times - 30\times$** (RTF $\approx 0.15 - 0.30\times$).
   - *Rationale*: Dedicated GDDR6 memory ($300 - 500\text{ GB/s}$) and FP16/BF16 Tensor Cores process 28-layer Transformer passes in milliseconds.
2. **Modern Desktop CPU with AVX-512 / DDR5 (Zen 4/5 or Intel Core 14th Gen)**:
   - *Projected Speedup*: **$3.5\times - 5.0\times$** (RTF $\approx 0.9 - 1.2\times$).
   - *Rationale*: Double FMA width (AVX-512) and DDR5-6000 dual-channel bandwidth ($80 - 90\text{ GB/s}$).
3. **Model Distillation (1-Step / 2-Step Student Model)**:
   - *Projected Speedup on existing CPU*: **$2.0\times - 4.0\times$** (RTF $\approx 1.1 - 2.2\times$).
   - *Rationale*: Halving or quartering the number of full 28-layer Transformer passes from 4 steps to 1–2 steps.

---

## 8. COST & RISK ANALYSIS

| Strategy | Implementation Cost | Numerical / Quality Risk | Ecosystem Stability | Reversibility | Expected Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Maintain V2.1 PyTorch CPU** | Zero | None (`[VERIFIED]` Vietnamese quality) | High | Immediate | Baseline ($4.4\times$ RTF) |
| **Port to TorchScript JIT** | Low | Low ($0.0$ error) | High | Immediate | **$-6.9\%$ (Slowdown)** |
| **Build DirectML for Python 3.13** | Very High | High (Kernel bugs, DML precision) | Low | Complex | Unlikely ($<1.1\times$) |
| **Export ONNX Pipeline** | Very High | Medium (Snake activation, ODE loop) | Medium | Complex | $\approx 0\%$ on CPU |
| **Distill Model to 1–2 Steps** | High (Training required) | Medium (Distillation loss tuning) | High | Immediate | **$2.0\times - 4.0\times$** |
| **Upgrade to Dedicated GPU** | Hardware purchase | Zero (Standard CUDA path) | Maximum | Immediate | **$15\times - 30\times$** |

---

## 9. FINAL VERDICT & RECOMMENDATION

```
================================================================================
V2.13 FINAL VERDICT: HARDWARE UPGRADE REQUIRED
================================================================================
1. The AMD Radeon Vega 8 Mobile is an integrated GPU sharing system DDR4 memory
   bandwidth (~35 GB/s), with no CUDA/TensorRT acceleration and no compatible 
   Python 3.13 DirectML runtime wheels.
2. PyTorch CPU inference on the AMD Ryzen 5 3450U is already optimized near its
   physical limit (MKL AVX2/FMA3, 6 threads, Batch-1 CFG=0).
3. TorchScript JIT tracing produces a 6.9% slowdown due to dispatch overhead on GEMM.
4. To break the RTF <= 1.0x barrier (real-time generation), a dedicated NVIDIA GPU 
   or a distilled 1-2 step model architecture is required.
================================================================================
```
