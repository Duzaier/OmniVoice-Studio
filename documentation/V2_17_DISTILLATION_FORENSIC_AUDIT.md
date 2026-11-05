# OMNIVOICE V2.17 — DISTILLATION PIPELINE FORENSIC AUDIT REPORT
**Trạng thái**: STRICT READ-ONLY AUDIT & VERIFICATION PASSED  
**Ngày thực hiện**: 02/10/2026  
**Golden Baseline**: Git Commit `e5a3059e9f423fb34ffd34b534cccdc28037c1b1` (V2.1 Batch-1 CFG=0, `num_step=4`, `guidance_scale=0.0`)  
**Môi trường phần cứng**: AMD Ryzen 5 3450U (4C/8T), 16 GB DDR4 RAM, PyTorch 2.8.0+cpu, Python 3.13.15  

---

## 1. EXECUTIVE SUMMARY & OFFICIAL VERDICT

### Kết luận chính thức (Official Verdict)
```
================================================================================
FINAL VERDICT: READY AFTER FIXES
================================================================================
```

### Đánh giá cốt lõi:
1. **Kiến trúc Teacher & Cơ chế Unmasking (`[VERIFIED]`)**:
   - OmniVoice không phải mô hình Continuous Diffusion (DDPM/Flow Matching trên latent liên tục) mà là **Discrete Masked Token Generative Model** với 8 tầng RVQ codebooks (mỗi codebook kích thước 2048 token) chạy ở 25Hz.
   - Quá trình suy luận sử dụng thuật toán **Iterative Confidence-based Unmasking** với cosine schedule (`omnivoice/models/omnivoice.py:1360-1435`).
2. **Exposure Bias / Covariate Shift Discovery (`[CRITICAL FINDING]`)**:
   - Trong thiết kế chưng cất Teacher Trajectory tĩnh của V2.16, Student Step 1 được huấn luyện trên input là trạng thái token trung gian $x_{2/4}$ do Teacher sinh ra.
   - Tuy nhiên, trong thực tế suy luận (inference), Student tự chọn unmask các token dựa trên confidence của chính nó ($x_{\text{student}, 1}$). Nếu $x_{\text{student}, 1} \neq x_{2/4}$, Student bước vào Step 2 với một phân phối dữ liệu chưa từng thấy trong quá trình huấn luyện (covariate shift), dẫn đến suy giảm độ chính xác âm thanh và ngữ điệu.
   - **Bắt buộc khắc phục (Fix Required)**: Bổ sung cơ chế **Scheduled Sampling / Self-Rollout Mixing** trong GPU training loop trước khi huấn luyện chính thức.
3. **Tính toàn vẹn thuật toán & Gradient Isolation (`[VERIFIED - 6/6 TESTS PASSED]`)**:
   - Toàn bộ 6/6 bài kiểm tra pháp y độc lập trong test suite `tests/test_v2_17_distillation_forensic_audit.py` đều vượt qua: Teacher weights bất biến ($\Delta w = 0.0$), Invariant không phá hủy được bảo toàn, Teacher logits được detach hoàn toàn, Gradient cô lập tuyệt đối và micro-batch loss hội tụ giảm $79.9\%$.

---

## 2. PHASE 1 & PHASE 2 — ALGORITHMIC & TRAJECTORY AUDIT

### Trả lời chi tiết 10 câu hỏi kiểm toán mã nguồn

| # | Câu hỏi kiểm toán | Kết quả thanh tra thực tế | Trích dẫn mã nguồn (Code Citation) |
|---|---|---|---|
| **Q1** | Teacher dự đoán gì ở mỗi step? | Phân phối xác suất Categorical (Logits kích thước `[B, 8, T, 2048]`) cho 8 tầng RVQ codebook. | [`omnivoice/models/omnivoice.py:1120-1175`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/omnivoice/models/omnivoice.py#L1120-L1175) |
| **Q2** | Lịch trình unmasking tính như thế nào? | Cosine schedule: $N_{\text{unmask}}(t) = \lceil N_{\text{total}} \cdot \cos(\frac{t}{T} \frac{\pi}{2}) \rceil$. Với 4 bước: 29% -> 50% -> 71% -> 100%. | [`omnivoice/models/omnivoice.py:1375-1390`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/omnivoice/models/omnivoice.py#L1375-L1390) |
| **Q3** | Token đã unmask có bị thay đổi không? | **Bất biến không phá hủy (Non-destructive)**: Điểm số của token đã unmask được set bằng $-\infty$, đảm bảo không bao giờ bị remask hoặc ghi đè. | [`omnivoice/models/omnivoice.py:1410-1425`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/omnivoice/models/omnivoice.py#L1410-L1425) |
| **Q4** | Confidence score tính như thế nào? | $C_i = \max_{v} \text{Softmax}(\text{Logits}_i)$. Các token có confidence cao nhất trong số các token đang bị mask được unmask trước. | [`omnivoice/models/omnivoice.py:1400-1415`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/omnivoice/models/omnivoice.py#L1400-L1415) |
| **Q5** | Ánh xạ Teacher 4 bước -> Student 2 bước? | Teacher bước $\{0, 1, 2, 3\} \to$ Student bước $\{0, 1\}$. Student Step 0: $x_0 (100\% \text{ mask}) \to x_2 (50\% \text{ unmasked})$. Student Step 1: $x_2 \to x_4 (100\% \text{ unmasked})$. | [`benchmarks/run_v2_16_distillation_poc_dryrun.py:75-115`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/benchmarks/run_v2_16_distillation_poc_dryrun.py#L75-L115) |
| **Q6** | Có hiện tượng Exposure Bias không? | **CÓ (CRITICAL)**. Student suy luận tự chọn token bằng confidence riêng, tạo độ lệch $x_{\text{student}, 1} \neq x_{2/4}$. Cần Scheduled Sampling để bù đắp. | Phân tích tại Mục 3 báo cáo này. |
| **Q7** | Công thức tổn thất (Loss) có chuẩn xác? | Chuẩn xác: $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}} + (1-\alpha) \tau^2 \mathcal{L}_{\text{KD}}$. Loss chỉ tính trên các token đang được active unmask. | [`benchmarks/run_v2_16_distillation_poc_dryrun.py:120-145`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/benchmarks/run_v2_16_distillation_poc_dryrun.py#L120-L145) |
| **Q8** | Teacher Logits có được detach không? | Đã xác minh: `teacher_logits.detach()` và `torch.no_grad()` ngăn chặn rò rỉ gradient và tràn VRAM. | [`tests/test_v2_17_distillation_forensic_audit.py:85-115`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/tests/test_v2_17_distillation_forensic_audit.py#L85-L115) |
| **Q9** | Codebook 8 tầng có cần trọng số khác nhau? | Cần thiết: Tầng 0 (acoustic base) quan trọng nhất ($w_0 = 1.5$), các tầng cao bù chi tiết ($w_{1..7} \in [1.3, 0.8]$). | [`omnivoice/models/omnivoice.py:510-540`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/omnivoice/models/omnivoice.py#L510-L540) |
| **Q10** | Trọng số Teacher có bị đóng băng không? | Tuyệt đối an toàn: Toàn bộ tensor Teacher có `requires_grad = False` và không nằm trong Optimizer. | [`tests/test_v2_17_distillation_forensic_audit.py:40-60`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/tests/test_v2_17_distillation_forensic_audit.py#L40-L60) |

---

## 3. EXPOSURE BIAS & SCHEDULED SAMPLING RESOLUTION

```
[Teacher Trajectory Distillation (Static)]
Step 1: x_0 (Mask 100%) -------------> Teacher x_2 (Mask 50%)
                                            |
                                            v (Teacher Tokens fed into Step 2 during training)
Step 2:                               Teacher x_2 -------------> Teacher x_4 (Ground Truth)

[Actual Student Autonomous Inference]
Step 1: x_0 (Mask 100%) -------------> Student x_1 (Mask 50%, unmasked by Student's top-K confidence)
                                            |
                                            x (MISMATCH / DISTRIBUTION SHIFT!)
Step 2:                               Student x_1 -------------> Degraded Output (Error Accumulation)
```

### Giải pháp kỹ thuật bắt buộc (Mandatory Fix for GPU Phase):
Áp dụng **Scheduled Sampling Ratio $p_{\text{student}} = 0.5$**:
Trong $50\%$ số batch của Step 2, thay vì dùng token $x_{2/4}$ của Teacher, feed trực tiếp token do Student tự sinh ở Step 1 ($x_{\text{student}, 1}$ kèm `detach()`), sau đó tính loss đối chiếu với Ground Truth Teacher $x_{4/4}$.

---

## 4. PHASE 4 — INDEPENDENT FORENSIC TEST SUITE EXECUTION

Đã thực thi script kiểm toán pháp y: `tests/test_v2_17_distillation_forensic_audit.py`.

```
================================================================================
OMNIVOICE V2.17 — DISTILLATION FORENSIC AUDIT TEST SUITE
================================================================================
  [TEST 1 PASSED] Teacher Immutability verified: max weight delta = 0.000000.
  [TEST 2 PASSED] Mask Correctness & Non-destructive Invariant verified.
  [TEST 3 PASSED] Trajectory Alignment & Logit Detachment verified.
  [TEST 4 PASSED] Gradient Isolation verified: Teacher grad is None, Student grad is Finite.
  [TEST 5 PASSED] Overfit Tiny Batch verified: Loss dropped from 5.5422 -> 1.1145 (-79.9%).
  [TEST 6 PASSED] Bitwise Reproducibility verified: Identical seed produces identical loss.

[ALL AUDIT TESTS COMPLETED SUCCESSFULLY — 6/6 PASSED]
```

---

## 5. PHASE 5 — DECISION MATRIX & CLOUD GPU SPECIFICATIONS

| Hạng mục kỹ thuật | Trạng thái kiểm toán | Hành động triển khai bắt buộc |
|---|---|---|
| **1. Teacher Backbone** | `VERIFIED` | Giữ nguyên checkpoint `e5a3059` đã freeze. |
| **2. Trajectory Alignment** | `VERIFIED` | Trích xuất các snapshot $\{x_0, x_2, x_4\}$ từ rollout của Teacher. |
| **3. Phơi nhiễm sai lệch (Step 2)** | `ACTION REQUIRED` | Triển khai Scheduled Sampling ($p=0.5$) trong training loop. |
| **4. Công thức Loss (CE + Soft KD)** | `VERIFIED` | Duy trì $\alpha=0.5$, $\tau=2.0$, nhân hệ số $\tau^2=4.0$. |
| **5. Trọng số RVQ Codebook** | `OPTIMIZATION` | Phân rã trọng số tầng: $[1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8]$. |
| **6. Teacher Freezing** | `VERIFIED` | `requires_grad=False`, bọc teacher forward trong `torch.no_grad()`. |
| **7. Logit Detachment** | `VERIFIED` | Áp dụng `teacher_logits.detach()` trước khi tính KL divergence. |
| **8. Optimizer Isolation** | `VERIFIED` | Chỉ truyền `student.parameters()` vào `AdamW(lr=1e-4, weight_decay=1e-2)`. |
| **9. Tính tái lập (Reproducibility)** | `VERIFIED` | Cố định `torch.manual_seed(42)` và deterministic backend. |
| **10. Kế hoạch Cloud GPU POC** | `READY AFTER FIXES` | Chạy trên 1x NVIDIA A100 / RTX 4090 trong 1000 step (~15 phút). |

---

## 6. NEXT ACTION RECOMMENDATION

1. Giữ nguyên Golden Baseline `e5a3059` ở trạng thái nguyên vẹn trên local CPU.
2. Không thực hiện training nặng trên CPU Ryzen 5 3450U.
3. Khi người dùng sẵn sàng tài nguyên GPU (Google Colab A100 hoặc RunPod RTX 4090), đóng gói script training hoàn chỉnh (kèm Scheduled Sampling fix) để thực hiện 2-Step Student Distillation Proof-of-Concept.
