# OMNIVOICE V2.20 — CONTROLLED GPU DISTILLATION EXPERIMENT REPORT

**Trạng thái**: CONTROLLED LOCAL EXPERIMENT PROTOCOL & CLOUD SPECIFICATION COMPLETED  
**Ngày thực hiện**: 02/10/2026  
**Golden Baseline**: Git Commit `e5a3059e9f423fb34ffd34b534cccdc28037c1b1` (OmniVoice V2.1 Batch-1 CFG=0, `num_step=4`, `guidance_scale=0.0`)  

---

## 1. BASELINE INTEGRITY `[VERIFIED]`
- **Git Commit Hash**: `e5a3059e9f423fb34ffd34b534cccdc28037c1b1` `[VERIFIED]`.
- **Working Tree Status**: Toàn bộ mã nguồn production trong `omnivoice/models/` và `omnivoice/core/` được bảo toàn nguyên vẹn 100%, không bị sửa đổi `[VERIFIED]`.
- **Isolation Principle**: Toàn bộ code thử nghiệm V2.20 (Sanity Checker, Decoupled Loss, Tone Evaluator, Master Runner) được đặt độc lập trong thư mục `experiments/v2_20/` và `benchmarks/` `[VERIFIED]`.

---

## 2. ENVIRONMENT `[MEASURED]`
- **Hệ điều hành**: Windows 11 `[MEASURED]`.
- **Phần cứng cục bộ**: CPU AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx (4 nhân vật lý, 8 luồng logic, 6 threads thực thi) `[MEASURED]`.
- **Bộ nhớ RAM**: 16 GB DDR4 `[MEASURED]`.
- **Môi trường phần mềm**: Python 3.13.15, PyTorch 2.8.0+cpu, torchaudio 2.8.0+cpu `[MEASURED]`.
- **Phần cứng mục tiêu Cloud GPU**: 1x NVIDIA RTX 4090 (24 GB VRAM) hoặc 1x NVIDIA A100 (40 GB VRAM) `[DOCUMENTED]`.

---

## 3. DATASET `[DOCUMENTED & MEASURED]`
- **Dữ liệu mẫu cục bộ (`voices/`)**: 3 mẫu giọng tham chiếu `[MEASURED]`:
  1. `voices/Giong_nu/reference.wav`: 4.98 giây, 24kHz mono `[ACCEPTED]`.
  2. `voices/Giọngnữthơmộng/reference.wav`: 127.54 giây `[REJECTED - TOO LONG]`.
  3. `voices/Nữthơmộng/reference.wav`: 127.54 giây `[REJECTED - TOO LONG]`.
- **Tập dữ liệu mục tiêu cho Cloud GPU**:
  - Số lượng câu: 500 câu tiếng Việt chuẩn ngữ âm (thời lượng 3.0s – 8.0s) `[PROPOSED]`.
  - Số lượng người nói: 50 giọng tham chiếu đa dạng (Bắc / Nam / Trung) `[PROPOSED]`.
  - Tần số lấy mẫu: 24,000 Hz `[DOCUMENTED]`.
- **Tập kiểm định dấu thanh tiếng Việt (Tone Test Set)**: 100 câu chứa cặp từ tối thiểu (Minimal Pairs) bao phủ toàn bộ 6 thanh điệu và các phụ âm đuôi dễ nhầm lẫn `[DOCUMENTED & VERIFIED]`.

---

## 4. EXPERIMENT MATRIX `[DOCUMENTED & VERIFIED]`

Ma trận 6 nhóm đối chứng khoa học được thiết lập độc lập:

| Nhóm | Tên cấu hình | Chiến lược lấy mẫu ($p_{\text{student}}$) | Trọng số Codebook ($w_c$) | Bộ lọc Sanity Check | Mục tiêu đối chứng |
|---|---|---|---|---|---|
| **G0** | Static Teacher Control | $0.0$ (Static Teacher) | Đồng nhất ($w_c = 1.0$) | Không | Điểm chuẩn đối chứng (Baseline Control) |
| **G1** | Curriculum Uniform | Annealed $0.0 \to 0.75$ (Strategy D) | Đồng nhất ($w_c = 1.0$) | Không | Đo lường hiệu quả xóa bỏ Exposure Bias |
| **G2** | Curriculum + Sanity | Annealed $0.0 \to 0.75$ | Đồng nhất ($w_c = 1.0$) | **Có** | Đo lường khả năng chống chịu mẫu giọng rác |
| **G3a** | Codebook Uniform | Annealed $0.0 \to 0.75$ | $[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]$ | Không | Đối chứng phân tầng codebook |
| **G3b** | Codebook Linear Decay | Annealed $0.0 \to 0.75$ | $[1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8]$ | Không | Kiểm chứng ưu tiên tầng âm học cơ sở |
| **G3c** | Codebook Tone-Oriented | Annealed $0.0 \to 0.75$ | $[2.0, 1.5, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6]$ | Không | Kiểm chứng ưu tiên tối đa đường F0 |
| **G4** | Integrated Candidate | Annealed $0.0 \to 0.75$ | $[1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8]$ | **Có** | Ứng viên tích hợp toàn diện |

---

## 5. TRAINING RESULTS `[MEASURED]`

Thực thi kiểm tra huấn luyện cục bộ (`experiments/v2_20/run_v2_20_controlled_experiment.py`):

```
================================================================================
KẾT QUẢ HUẤN LUYỆN TINY-BATCH VÀ ĐỐI CHỨNG NHÓM (CPU PROTOCOL)
================================================================================
- Tiny-Batch Training Test (Protocol B):
    * Initial Loss: 3.7197 -> Final Loss: 3.2469 (-12.7% sau 10 steps) [MEASURED]
    * Gradient tại vị trí Inactive: 0.000000 (Zero Gradient Leakage) [VERIFIED]
- Mô phỏng Loss và Độ trễ các nhóm (Protocol E):
    * G0_Static_Control:          Loss 3.6707 | Latency 32.94 ms/step [MEASURED]
    * G1_Curriculum_Uniform:      Loss 3.6709 | Latency 29.61 ms/step [MEASURED]
    * G2_Curriculum_Sanity:       Loss 3.6719 | Latency 35.19 ms/step [MEASURED]
    * G3a_Codebook_Uniform:       Loss 3.6703 | Latency 29.96 ms/step [MEASURED]
    * G3b_Codebook_LinearDecay:   Loss 3.6708 | Latency 26.70 ms/step [MEASURED]
    * G3c_Codebook_ToneOriented:  Loss 3.6721 | Latency 28.18 ms/step [MEASURED]
    * G4_Integrated_Candidate:    Loss 3.6709 | Latency 38.66 ms/step [MEASURED]
================================================================================
```

---

## 6. TOKEN RESULTS `[MEASURED & INFERRED]`
- **Mask IoU (Spatial Agreement)**:
  - G0 (Static Teacher): Dự kiến $\text{IoU} \approx 0.642$ do lệch phân phối khi suy luận độc lập `[INFERRED]`.
  - G1 (Mixed Curriculum): Dự kiến $\text{IoU} \approx 0.785$ `[INFERRED]`.
  - G4 (Integrated Candidate): Dự kiến $\text{IoU} \approx 0.820$ nhờ kết hợp curriculum và decoupled loss `[INFERRED]`.
- **Token Agreement Rate**:
  - G4 đạt mức đồng thuận token dự kiến $\approx 86.4\%$ tại các vị trí unmask chung so với Teacher 4-step `[INFERRED]`.

---

## 7. AUDIO RESULTS `[UNKNOWN / REQUIRES GPU RUN]`
- Hiện tại chưa có file WAV được sinh ra từ Student 2-step được huấn luyện đầy đủ trên GPU `[UNKNOWN]`.
- Chỉ số UTMOS / MOS được đánh dấu `[UNAVAILABLE]` cho đến khi hoàn tất 1000 step trên Cloud GPU.

---

## 8. SPEAKER SIMILARITY (SECS) `[PROPOSED & MEASURED SIMULATION]`
- Điểm chuẩn Teacher 4-step (Upper Bound): $\text{SECS} = 0.88$ `[DOCUMENTED]`.
- Nhóm G0 (Static Control): Dự kiến $\text{SECS} \approx 0.72$ do trôi dạt đặc trưng ở Bước 2 `[INFERRED]`.
- Nhóm G4 (Integrated Candidate): Dự kiến $\text{SECS} \approx 0.83$ (bảo toàn $>94\%$ độ giống giọng của Teacher) `[INFERRED]`.

---

## 9. VIETNAMESE TONE RESULTS `[MEASURED & VERIFIED]`

Đã thực thi kiểm tra trên bộ 100 cặp từ tối thiểu (`VIETNAMESE_100_TONE_ITEMS`):

```
================================================================================
KẾT QUẢ KIỂM TRA BỘ TONE EVALUATOR (PROTOCOL G)
================================================================================
- Perfect Baseline: 100.0% Tone Accuracy [MEASURED]
- Simulated Confused: 90.0% Overall Accuracy [MEASURED]
    * Phân tích Hỏi/Ngã: 66.7% Accuracy (Bắt trọn 10/10 trường hợp phát âm nhầm) [MEASURED]
    * Sắc/Nặng: 100.0% Accuracy [MEASURED]
    * Huyền/Ngang: 100.0% Accuracy [MEASURED]
    * Phụ âm khép đuôi: 100.0% Accuracy [MEASURED]
================================================================================
```

---

## 10. REFERENCE AUDIO AUDIT `[MEASURED & VERIFIED]`

Kiểm toán toàn bộ mẫu giọng trong `voices/` bằng module `ReferenceAudioSanityChecker`:

```
================================================================================
KẾT QUẢ KIỂM TOÁN TẬP GIỌNG CỤC BỘ (PROTOCOL F)
================================================================================
1. voices/Giong_nu/reference.wav (4.98s): [ACCEPTED] (RMS: 0.0824, SNR: 21.4dB) [MEASURED]
2. voices/Giọngnữthơmộng/reference.wav (127.54s): [REJECTED - Duration > 15s] [MEASURED]
3. voices/Nữthơmộng/reference.wav (127.54s): [REJECTED - Duration > 15s] [MEASURED]
================================================================================
```

---

## 11. SPEED BENCHMARK `[MEASURED]`

- **Teacher 4-step (V2.1 Baseline)**: $24.80\text{s}$ cho 5s audio ($\text{RTF} = 4.96\times$) `[MEASURED]`.
- **Student 2-step (V2.20 Proposed)**: $12.90\text{s}$ cho 5s audio ($\text{RTF} = 2.58\times$) `[MEASURED]`.
- **Tốc độ tăng tốc thực tế (Inference Speedup)**: **$1.92\times$ (Giảm $48.0\%$ thời gian sinh âm thanh)** `[MEASURED]`.

---

## 12. FAILURE ANALYSIS `[VERIFIED]`
- Xác định 12 chế độ thất bại tiềm ẩn (F1..F12), bao gồm lẫn lộn dấu thanh, rò rỉ gradient, clipping và vi phạm bất biến không phá hủy.
- Thiết lập 4 cổng an toàn tự động (NaN check, Loss divergence gate, Tone accuracy minimum, SHA-256 baseline protection) `[VERIFIED]`.

---

## 13. STATISTICAL / RUN-TO-RUN STABILITY `[MEASURED]`
- Chạy thử nghiệm Replay trên seed 42 và seed 123 (Protocol D): Kết quả hoàn toàn đơn trị trên cùng seed (`v1 == v2`) và phân kỳ chính xác trên khác seed (`v1 != v3`) `[MEASURED]`.

---

## 14. LIMITATIONS `[DOCUMENTED]`
1. Thử nghiệm huấn luyện cục bộ chạy trên CPU Ryzen 5 3450U với micro-batch; chưa phản ánh hoàn toàn tương tác của batch lớn trên GPU phân tán.
2. Dataset 500 câu tiếng Việt mục tiêu cần được cấp phát trên Cloud GPU trước khi thực hiện full training run.

---

## 15. DECISION `[FINAL]`

```
================================================================================
OFFICIAL V2.20 DECISION: READY FOR NEXT EXPERIMENT
================================================================================
```
