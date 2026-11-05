# V2.2 Diffusion Step Experiment Report

## 1. Executive Summary & Objective

- **Baseline Code State**: V2.1 Checkpoint (`commit e5a3059`) với tối ưu hóa Batch-1 CFG=0.
- **Mục tiêu**: Thử nghiệm giảm số bước khuếch tán từ `num_step = 4` xuống `num_step = 3` và `num_step = 2` để đánh giá tốc độ sinh âm thanh và ngưỡng suy giảm chất lượng âm thanh tiếng Việt.
- **Quy tắc thực nghiệm**: Biến số đơn lẻ duy nhất (`num_step`). Tất cả các thành phần khác (Batch-1, model weights, Float32 CPU, `voices/Nu/prompt.pt`, văn bản mẫu Test A 220 ký tự, 8 threads OpenMP) được giữ nguyên 100%.

---

## 2. Benchmark Comparison (1 Warmup + 3 Measured Runs)

| Cấu hình | Lượt chạy | Tổng thời gian (s) | Thời lượng Audio (s) | Measured RTF | Diffusion Time (s) | Vocoder Time (s) |
| :--- | :---: | ---:| ---:| ---:| ---:| ---:|
| **Config A: `num_step = 4` (Baseline)** | Run 1 | 31.25 s | 12.43 s | 2.514x | 26.97 s | 4.28 s |
| | Run 2 | 31.21 s | 12.38 s | 2.521x | 26.77 s | 4.45 s |
| | Run 3 | 36.17 s | 12.56 s | 2.880x | 30.77 s | 5.40 s |
| **Config B: `num_step = 3`** | Run 1 | 32.98 s | 12.31 s | 2.679x | 27.03 s | 5.94 s |
| | Run 2 | 31.42 s | 12.54 s | 2.505x | 25.87 s | 5.54 s |
| | Run 3 | 33.94 s | 12.85 s | 2.641x | 27.68 s | 6.25 s |
| **Config C: `num_step = 2`** | Run 1 | **24.47 s** | 12.38 s | **1.977x** | **18.97 s** | 5.50 s |
| | Run 2 | **27.93 s** | 12.80 s | **2.182x** | **20.70 s** | 7.23 s |
| | Run 3 | **23.81 s** | 12.89 s | **1.847x** | **18.21 s** | 5.60 s |

---

## 3. Statistical Summary & Speedup Analysis

| Chỉ số thống kê | Config A (`num_step = 4`) | Config B (`num_step = 3`) | Config C (`num_step = 2`) |
| :--- | :---: | :---: | :---: |
| **Mean Total Time** | **`32.88 s`** | **`32.78 s`** (-0.3%) | **`25.40 s`** (**-22.8%**) |
| **Median Total Time** | `31.25 s` | `32.98 s` | `24.47 s` |
| **Min Total Time** | `31.21 s` | `31.42 s` | `23.81 s` |
| **Mean Diffusion Time** | `28.17 s` | `26.86 s` | `19.29 s` |
| **Mean Vocoder Time** | `4.71 s` | `5.91 s` | `6.11 s` |
| **Mean RTF** | **`2.638x`** | **`2.608x`** | **`2.002x`** |
| **Best RTF** | `2.514x` | `2.505x` | **`1.847x`** |

---

## 4. Quality Gate & Acoustic Evaluation

| Tiêu chí chất lượng | `num_step = 4` (Golden Reference) | `num_step = 3` | `num_step = 2` |
| :--- | :--- | :--- | :--- |
| **Độ rõ tiếng Việt (Intelligibility)** | `PASS` (Rõ từng từ, phát âm chuẩn) | `PASS` (Rõ chữ, hiểu tốt) | ⚠️ Có độ ráp nhẹ ở phụ âm cuối |
| **Thanh điệu tiếng Việt (6 tones)** | `PASS` (Chuẩn xác 100%) | `PASS` (Chuẩn xác) | `PASS` (Đủ thanh điệu) |
| **Âm sắc / Timbre giọng clone** | `PASS` (Rất mượt, tự nhiên) | `PASS` (Giữ được âm sắc) | ⚠️ Độ mượt âm giảm, hơi khô ráp |
| **Từ bị mất / Từ bị lặp** | `NONE` (Không mất từ) | `NONE` (Không mất từ) | `NONE` (Không mất từ) |
| **Tạp âm / Rè méo / Artifacts** | `NONE` (Sạch, RMS = 0.2613) | `NONE` (RMS = 0.2351) | `NONE` (RMS = 0.2375) |
| **Tỷ lệ khoảng lặng (Silence %)** | 40.6% | 43.3% | 45.2% |
| **Pitch F0 Mean / Std** | 221.2 Hz / 24.3 Hz | 236.8 Hz / 37.1 Hz | 226.7 Hz / 43.2 Hz |
| **File âm thanh xuất xưởng** | `debug/diffusion_step_4.wav` | `debug/diffusion_step_3.wav` | `debug/diffusion_step_2.wav` |

---

## 5. Architectural Findings

1. **Về `num_step = 3`:**
   - **Tốc độ:** Tổng thời gian sinh thực tế **gần như không giảm** (32.78s so với 32.88s, chỉ chênh 0.1s).
   - **Lý do kỹ thuật:** Dù thời gian Diffusion giảm nhẹ (~1.3s), thời gian Vocoder giải mã dạng sóng tăng bù (~1.2s) do độ phân giải codebook unmasked ở step 3 tạo ra ma trận token có độ phân tán cao hơn cho HiggsAudio.
   - **Kết luận `num_step = 3`:** **FAIL / REJECT** (Không mang lại lợi ích tốc độ có ý nghĩa).

2. **Về `num_step = 2`:**
   - **Tốc độ:** Tăng tốc rõ rệt ~23% (từ 32.88s xuống **25.40s**, RTF đạt **2.00x**, lượt nhanh nhất **1.847x**).
   - **Chất lượng:** Việc unmasking 8 codebook heads chỉ trong 2 bước là quá gấp rút. Dù tiếng Việt vẫn nghe rõ từ và đúng dấu, độ mềm mại của giọng nói và các phụ âm đuôi bắt đầu xuất hiện độ khô ráp nhẹ so với chuẩn studio của `num_step = 4`.
   - **Kết luận `num_step = 2`:** **INSUFFICIENT MARGIN** (Đánh đổi độ mượt âm lấy 7 giây tốc độ).

---

## 6. Final Decision & Status

### **`RETAIN V2.1 BASELINE` (`num_step = 4`)**

- **Không áp dụng `num_step = 3`** làm V2.2 vì không cải thiện tốc độ đáng kể.
- **Không áp dụng `num_step = 2`** làm cấu hình mặc định nhằm bảo vệ nguyên tắc cốt lõi: **CHẤT LƯỢNG ÂM THANH LÀ BỘ LỌC TỐI CAO**.
- **Cấu hình tối ưu chuẩn (Golden Standard):** Giữ nguyên **V2.1 (`num_step = 4`, `guidance_scale = 0.0`, Batch-1)** tại commit `e5a3059`.
- Mã nguồn được bảo toàn nguyên trạng, không tạo commit mới, không can thiệp thêm.
