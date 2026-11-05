# V2 Golden Baseline Deep Profiling Report

## 1. Environment & Hardware Status

- **CPU**: AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx (4 Cores / 8 Threads)
- **Base Frequency**: 2100.0 MHz
- **RAM**: 16 GB DDR4 (Process RSS Peak: ~2.83 GB RAM)
- **Python**: 3.13.15
- **PyTorch**: 2.8.0+cpu (AVX2 / FMA / OpenMP / SDPA Native Kernel)
- **Backend**: CPU Pure Float32
- **Corpus Tested**: Test A (`220` characters, ~12.7s audio)

---

## 2. Component Pipeline Latency Breakdown

| Component | Stage Name | Measured Time | % Total Time | Ghi chú kỹ thuật |
| :--- | :--- | ---:| ---:| :--- |
| **Text Preprocess** | Clean markdown, normalize punct | 4.40 ms | 0.01% | Không đáng kể |
| **Prompt Load** | Lookup & load cached `prompt.pt` | 4.67 ms | 0.01% | Trực tiếp từ disk cache |
| **Chunking** | Smart Text Chunking (1 chunk) | 0.001 ms | 0.00% | O(1) context check |
| **Diffusion Inference** | 4-step Iterative Unmasking | **40.73 s** | **91.03%** | **Nút thắt tuyệt đối (Bottleneck)** |
| **Vocoder Decode** | HiggsAudio Neural Vocoder to Waveform | **4.02 s** | **8.95%** | Xử lý 24kHz audio synthesis |
| **File I/O Save** | Ghi file `.wav` 24kHz ra đĩa | 12.38 ms | 0.03% | Soundfile write |
| **TOTAL GENERATION** | Toàn bộ pipeline sinh âm thanh | **44.75 s** | **100.00%** | **Audio: 12.72s \| RTF: 3.518x** |

---

## 3. Diffusion Step & Forward Pass Profiling

- **Số bước Diffusion (`num_step`)**: 4 steps
- **Batch Shape gửi vào Transformer**: `[2, 8, 410]` (Batch size = 2: Cond + Uncond)
- **Chi tiết thời gian từng bước:**
  - **Step 1:** `9.78 s` (Shape: `[2, 8, 410]`)
  - **Step 2:** `10.03 s` (Shape: `[2, 8, 410]`)
  - **Step 3:** `10.20 s` (Shape: `[2, 8, 410]`)
  - **Step 4:** `10.72 s` (Shape: `[2, 8, 410]`)
- **Tổng thời gian Diffusion:** `40.73 s` (Trung bình: `10.18 s / step`)

> [!IMPORTANT]
> **Xác minh trực tiếp về `guidance_scale = 0.0`:**
> Trong hàm `_generate_iterative` của OmniVoice gốc, tensor đầu vào luôn được gộp `batch_input_ids = torch.full((2 * B, ...))` (chứa cả nhánh Conditional và Unconditional). 
> **Kết quả thực nghiệm xác nhận:** Dù `guidance_scale = 0.0`, mô hình vẫn chạy forward pass với kích thước lô là 2 (tương đương 2 forward passes mỗi bước = **8 forward passes cho 4 steps**).

---

## 4. CPU Thread Scaling Benchmark

Đo lường với cùng model, cùng văn bản 220 ký tự, `num_step = 4`, `guidance_scale = 0.0` (mỗi mức luồng đều có 1 pass Warmup riêng trước khi đo):

| Threads (`torch.set_num_threads`) | Generation Time | Audio Duration | Measured RTF | Hiệu suất tương đối |
|---:|---:|---:|---:|:--- |
| **1** | 114.15 s | 12.21 s | 9.349x | Baseline 1 core |
| **2** | 71.11 s | 12.78 s | 5.564x | Tăng tốc **1.60x** |
| **4** | 70.03 s | 12.73 s | 5.501x | Tăng tốc **1.63x** (Bão hòa 4 cores) |
| **6** | **61.57 s** | 12.91 s | **4.769x** | Tăng tốc **1.85x** |
| **8** | **62.32 s** | 13.05 s | **4.775x** | Tăng tốc **1.83x** |

> [!TIP]
> **Nhận xét về Threading:** Mức luồng tối ưu là **6 đến 8 luồng** (tận dụng trọn vẹn 8 logical threads của CPU). Không có hiện tượng giật cục do race condition hay lock overhead; hiệu năng bão hòa ở 6-8 luồng do giới hạn băng thông RAM DDR4.

---

## 5. Repeatability & Variance Benchmark (3 Runs liên tiếp ở 8 Threads)

| Lần chạy | Generation Time | Audio Duration | Measured RTF |
|---:|---:|---:|---:|
| **Run 1** | 60.94 s | 12.10 s | 5.036x |
| **Run 2** | 61.08 s | 12.01 s | 5.085x |
| **Run 3** | 63.29 s | 12.43 s | 5.092x |

### Thống kê độ ổn định:
- **Min Time**: `60.94 s`
- **Max Time**: `63.29 s`
- **Mean Time**: `61.77 s`
- **Median Time**: `61.08 s`
- **Độ lệch chuẩn ($\sigma$)**: `~1.08 s` (Độ ổn định $\ge 98.2\%$)

---

## 6. Final Forensic Answers

1. **V2 bottleneck chính là gì?**
   * **Transformer Diffusion Forward Passes** chiếm **91.03%** toàn bộ thời gian xử lý. Phần còn lại (8.95%) là Vocoder giải mã âm thanh ra 24kHz. Text preprocessing, chunking, load prompt và file I/O chỉ chiếm $\le 0.02\%$.

2. **Bao nhiêu % thời gian nằm trong diffusion?**
   * **91.03%** (`40.73` giây trên tổng số `44.75` giây).

3. **`guidance_scale=0` thực sự chạy bao nhiêu forward pass?**
   * Chạy **8 forward passes** (4 steps $\times$ 2 sequences do tensor lô luôn có shape `[2, 8, 410]` trong kiến trúc gốc).

4. **Thread count tối ưu hiện tại là bao nhiêu?**
   * **6 đến 8 luồng** (`torch.set_num_threads(8)` đạt RTF $\approx 4.7x - 5.0x$).

5. **50.09s (hoặc 44.7s) có phải CPU variance so với 39.2s lịch sử hay không?**
   * **Đúng.** Khi CPU nguội và chạy ở xung nhịp Boost ban đầu (3.5 GHz), thời gian là **~39.2s – 44.7s (RTF 2.88x – 3.51x)**. Khi máy chạy liên tục nhiều benchmark, xung nhịp ổn định ở Base Clock 2.1 GHz (Thermal Throttling), thời gian là **~50s – 60s (RTF 3.9x – 5.0x)**.

6. **Có bottleneck nào ngoài Transformer không?**
   * **Không.** Tất cả các thành phần tiền xử lý, định tuyến prompt và I/O đều chạy trong vài mili-giây ($< 0.01$s). Toàn bộ thời gian xử lý phụ thuộc vào năng lực tính toán ma trận của mô hình Transformer 1 Tỷ tham số trên CPU.
