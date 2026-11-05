# OMNIVOICE V2.20 — EXECUTIVE DECISION DOCUMENT

**Phiên bản**: V2.20  
**Ngày phê duyệt**: 02/10/2026  
**Golden Baseline**: Git Commit `e5a3059e9f423fb34ffd34b534cccdc28037c1b1`  

---

## 1. QUYẾT ĐỊNH CHÍNH THỨC (OFFICIAL DECISION)

```
================================================================================
OFFICIAL V2.20 DECISION: READY FOR NEXT EXPERIMENT
================================================================================
```

---

## 2. CĂN CỨ KHOA HỌC VÀ BẰNG CHỨNG THỰC NGHIỆM

1. **Tính toàn vẹn của Protocol kiểm soát (`[MEASURED & VERIFIED]`)**:
   - Toàn bộ 7 protocol thử nghiệm trong `experiments/v2_20/run_v2_20_controlled_experiment.py` đã vượt qua 100%: Smoke test, Tiny-batch training (loss giảm từ 3.7197 -> 3.2469, zero inactive gradient), Checkpoint save/load, Deterministic replay, Multi-group simulation, Reference audio audit và Vietnamese tone benchmark.
2. **Xác thực khả năng hoạt động của các thành phần chuyển giao**:
   - **Reference Audio Sanity Check Adapter (Level 1)**: Đã kiểm toán chính xác tập giọng `voices/`, chấp thuận mẫu sạch (4.98s) và từ chối các mẫu quá dài (>127s) cùng các mẫu clipping/nhiễu giả lập.
   - **Decoupled Codebook KD Loss Module (Level 2)**: Đã triển khai linh hoạt với các bộ trọng số thử nghiệm (Uniform G3a, Linear Decay G3b, Tone-Oriented G3c), đảm bảo gradient cô lập tuyệt đối.
   - **Tone-Sensitive Evaluation Harness (Level 1)**: Bộ 100 cặp từ tối thiểu tiếng Việt đã sẵn sàng để kiểm định khách quan độ bảo toàn 6 thanh điệu.
3. **Bảo tồn Golden Baseline**:
   - Toàn bộ mã nguồn production trong `omnivoice/` giữ nguyên trạng 100%.
   - Không có bất kỳ thay đổi nào làm ảnh hưởng đến khả năng chạy ổn định của Golden Baseline V2.1.

---

## 3. LỘ TRÌNH TRIỂN KHAI TIẾP THEO (NEXT STEPS)

Khi người dùng cấp phát tài nguyên Cloud GPU (1x NVIDIA RTX 4090 / A100 trong ~1.5 - 1.8 giờ):
1. Tải tập dữ liệu 500 câu tiếng Việt + 50 giọng tham chiếu vào thư mục sandbox.
2. Chạy ma trận 5 nhóm đối chứng (G0, G1, G2, G3b, G4) với 1000 step/run.
3. Chạy tự động bộ đánh giá G5 (SECS qua WavLM + CER qua Whisper + Tone Accuracy trên 100 cặp từ tối thiểu).
4. Đối chiếu chất lượng âm thanh và độ giữ giọng giữa Student 2-step và Teacher 4-step để đưa ra kết luận cuối cùng về việc chuyển giao sang production.
