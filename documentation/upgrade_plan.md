# 🎙️ OmniVoice → F5-TTS ONNX: Nâng cấp toàn diện

## 📊 Đánh giá cấu hình máy

| Thành phần | Thông số | Đánh giá |
|---|---|---|
| CPU | AMD Ryzen 5 3450U, 4 cores / 8 threads, 2.1GHz | ⚠️ Entry-level laptop 2019 |
| GPU | AMD Radeon Vega Mobile (iGPU) — 2GB shared | ⚠️ iGPU tích hợp, dùng chung RAM |
| RAM | 16GB | ✅ Đủ để chạy model |

> [!CAUTION]
> **Kỳ vọng thực tế về "3 giây":** Với Ryzen 5 3450U và iGPU Vega, không có model voice cloning neural nào đạt 3 giây cho một câu dài (15–20 từ). Con số thực tế có thể đạt được là **8–20 giây** tùy model và cấu hình.
>
> Để đạt 3 giây, cần ít nhất GPU rời (RTX 3060, RX 6600 trở lên).

---

## 🏆 So sánh giải pháp phù hợp nhất

| Model | Voice Cloning | Tốc độ CPU (RTF) | Viet | Ghi chú |
|---|---|---|---|---|
| **OmniVoice** (hiện tại) | ✅ Xuất sắc | ~5–10 | ✅ 600+ ngôn ngữ | Diffusion, chậm |
| **F5-TTS + ONNX** | ✅ Rất tốt | ~0.3–0.8 | ✅ | **→ Tốt nhất cho máy này** |
| **Fish Speech 1.5** | ✅ Xuất sắc | ~1.5–3 | ✅ | Cần GPU rời |
| **Kokoro** | ❌ Không có | ~0.03 | ❌ | Chỉ fixed voices |

> [!IMPORTANT]
> **Lựa chọn tốt nhất cho cấu hình này: F5-TTS + ONNX Runtime + DirectML**
> - F5-TTS-ONNX hỗ trợ **Windows + AMD iGPU qua DirectML**
> - RTF ~0.3–0.8 trên CPU (4–15 giây thực tế cho câu 5 giây audio)
> - Voice cloning từ 10 giây audio tham chiếu
> - Hỗ trợ tiếng Việt tốt

---

## 🚀 Kế hoạch nâng cấp (3 giai đoạn)

```
Giai đoạn 1: Cài F5-TTS + ONNX Export
Giai đoạn 2: Kích hoạt DirectML cho AMD Vega
Giai đoạn 3: Build giao diện web thống nhất (Gradio)
```

---

## 📁 Cấu trúc dự án mới

```
C:\Users\Duzaier\.gemini\antigravity\scratch\VoiceClone\
├── f5tts/               ← F5-TTS (model gốc)
├── onnx_models/         ← ONNX exported models
│   ├── F5_Preprocess.onnx
│   ├── F5_Transformer.onnx
│   └── F5_Decode.onnx
├── ref_voices/          ← Thư mục lưu giọng tham chiếu
├── outputs/             ← Audio đầu ra
├── app.py               ← Gradio web UI thống nhất
└── requirements.txt
```

---

## ⚡ Tối ưu hóa áp dụng

1. **ONNX Runtime DirectML** — dùng AMD Vega iGPU thay CPU thuần
2. **FP16 Transformer** — giảm memory, tăng tốc ~30%
3. **Speaker embedding cache** — tính toán 1 lần, tái sử dụng
4. **Chunk inference** — xử lý từng đoạn, phát audio sớm hơn
5. **Inference Steps = 12** — cân bằng tốt giữa tốc độ/chất lượng
6. **Sway Sampling** — chiến lược sampling thông minh của F5-TTS

---

## 🎯 Kết quả kỳ vọng sau nâng cấp

| Kịch bản | Thời gian tạo audio |
|---|---|
| Câu ngắn (5 từ, ~1s audio) | **~3–5 giây** ✅ |
| Câu trung bình (15 từ, ~3s audio) | **~8–15 giây** |
| Câu dài (30 từ, ~6s audio) | **~15–30 giây** |

> [!TIP]
> Để đạt **< 3 giây** ổn định: cần upgrade lên GPU rời (AMD RX 6600 hoặc NVIDIA RTX 3060), có thể giảm xuống ~1–2 giây.
