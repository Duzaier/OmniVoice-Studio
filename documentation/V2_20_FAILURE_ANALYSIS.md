# OMNIVOICE V2.20 — FAILURE ANALYSIS REPORT
## CROSS-MODEL FAILURE MODES & MITIGATION PROTOCOLS

**Trạng thái**: PHÂN TÍCH CHẾ ĐỘ THẤT BẠI VÀ CƠ CHẾ DỪNG SỚM  
**Ngày thực hiện**: 02/10/2026  
**Golden Baseline**: Git Commit `e5a3059e9f423fb34ffd34b534cccdc28037c1b1`  

---

## 1. MA TRẬN 12 CHẾ ĐỘ THẤT BẠI (FAILURE MODES MATRIX)

| STT | Chế độ thất bại (Failure Mode) | Cơ chế phát sinh | Triệu chứng nhận biết | Giải pháp xử lý trong V2.20 | Mức độ nghiêm trọng |
|---|---|---|---|---|---|
| **F1** | **Reference Audio quá dài ($>15\text{s}$)** | Nạp file WAV nguyên bản chưa cắt | Tràn bộ nhớ attention RVQ, loãng đặc trưng giọng | Sanity Checker từ chối ngay lập tức | `[CRITICAL]` |
| **F2** | **Reference Audio bị clipping méo tiếng** | File ghi âm bị bão hòa mic ($>0.2\%$ mẫu) | Giọng clone bị rè kim loại, nhiễu âm cao tần | Sanity Checker từ chối tự động | `[HIGH]` |
| **F3** | **Reference Audio nhiều tạp âm ($\text{SNR} < 14\text{dB}$)** | Ghi âm môi trường quán xá/xe cộ | Vocoder sinh tiếng sôi nền, lẫn giọng tạp | Sanity Checker từ chối | `[HIGH]` |
| **F4** | **Lẫn lộn dấu thanh tiếng Việt (Hỏi vs Ngã)** | Nén 2 bước làm phẳng đường F0 ở Tầng 0 | Đọc nhầm "sữa" thành "sửa", "nghĩ" thành "nghỉ" | Decoupled Codebook Loss ($w_0 \ge 1.5$) | `[CRITICAL]` |
| **F5** | **Nuốt phụ âm khép đuôi (-ng, -nh, -ch)** | Mất chi tiết phổ ở Tầng 1–3 | Đọc "bánh" thành "bách", "cánh" thành "cách" | Giám sát CER Whisper có trọng số phụ âm cuối | `[HIGH]` |
| **F6** | **Student tự tin thái quá vào token sai** | Lấy mẫu argmax ở Bước 1 chọn sai token | Token sai bị đóng băng vĩnh viễn ở Bước 2 | Scheduled Sampling ($p=0.75$) rèn luyện tự sửa | `[HIGH]` |
| **F7** | **Lệch giọng clone (Speaker Drift) đoạn dài** | Context attention bị trôi sau $>15\text{s}$ | Giọng biến đổi dần sang giọng mặc định | Giữ nguyên punctuation chunker & cross-fading | `[MEDIUM]` |
| **F8** | **Tràn bộ nhớ / Rò rỉ Gradient (VRAM Leak)** | Quên `detach()` logits hoặc intermediate tokens | VRAM tăng liên tục qua các epoch | Enforce `detach()` và kiểm tra gradient inactive | `[CRITICAL]` |
| **F9** | **Bất biến không phá hủy bị vi phạm** | Token đã unmask ở Bước 1 bị ghi đè ở Bước 2 | Phá vỡ cấu trúc âm học Bước 1 đã hội tụ | Enforce `scores.masked_fill_(-inf)` | `[CRITICAL]` |
| **F10** | **Đột biến Loss (Loss Explosion / NaN)** | Tỷ lệ học quá cao hoặc chia cho 0 trong Softmax KD | Loss nhảy lên $>10.0$ hoặc NaN | Cổng dừng sớm tự động (Loss Divergence Gate) | `[CRITICAL]` |
| **F11** | **Sai khác Tokenizer tiếng Việt** | Nhầm lẫn sang tokenizer nước ngoài | Mất ký tự UTF-8 có dấu (ư, ơ, đ, ă, â) | Giữ nguyên 100% Qwen Vietnamese tokenizer | `[FATAL]` |
| **F12** | **Mất cân bằng Codebook (Codebook Imbalance)** | Tầng 7 bị phạt quá nặng ($w_7 \to 0$) | Âm thanh bị đục (muffled), thiếu độ trong | Giới hạn ngưỡng dưới $w_c \ge 0.6$ | `[MEDIUM]` |

---

## 2. QUY TRÌNH AN TOÀN VÀ CƠ CHẾ DỪNG SỚM (SAFETY GATES PROTOCOL)

Khi chạy huấn luyện trên Cloud GPU, hệ thống tự động ngắt nếu vi phạm bất kỳ điều kiện nào sau đây:

1. **Gate 1 (NaN / Inf Check)**: Nếu bất kỳ tensor loss hoặc gradient nào chứa `NaN` hoặc `Inf` $\implies$ Stop ngay lập tức.
2. **Gate 2 (Loss Threshold)**: Nếu Loss trung bình trong 50 bước liên tiếp $> 10.0$ $\implies$ Early Stop.
3. **Gate 3 (Tone Accuracy Minimum)**: Nếu Tone Accuracy trên tập 100 minimal pairs $< 85.0\%$ $\implies$ Đánh dấu checkpoint lỗi.
4. **Gate 4 (Baseline Protection)**: Kiểm tra SHA-256 của toàn bộ file trong `omnivoice/models/` trước và sau mỗi epoch $\implies$ Nếu có bất kỳ sự thay đổi nào, rollback ngay lập tức.
