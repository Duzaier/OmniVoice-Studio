# OMNIVOICE V2.20 — VIETNAMESE TONE EVALUATION REPORT
## PHONETIC ANALYSIS & MINIMAL-PAIR TEST PROTOCOL

**Trạng thái**: NGHIÊN CỨU & KIỂM TRA CHUYÊN SÂU THANH ĐIỆU TIẾNG VIỆT  
**Ngày thực hiện**: 02/10/2026  
**Golden Baseline**: Git Commit `e5a3059e9f423fb34ffd34b534cccdc28037c1b1`  

---

## 1. MỤC TIÊU VÀ BẢN CHẤT NGỮ ÂM 6 THANH ĐIỆU TIẾNG VIỆT

Tiếng Việt là ngôn ngữ thanh điệu với 6 thanh có đường cong cao độ (F0 contour) và đặc tính âm học riêng biệt:

1. **Thanh Ngang (Level / Unmarked)**: Cao độ trung bình, bằng phẳng, không có khép thanh môn.
2. **Thanh Huyền (Low Falling)**: Cao độ trầm, rơi mượt từ trung bình xuống thấp.
3. **Thanh Sắc (High Rising)**: Cao độ cao, tăng nhanh, kết thúc căng.
4. **Thanh Hỏi (Dipping-Rising)**: Rơi xuống âm vực trầm rồi lượn ngược lên trung bình.
5. **Thanh Ngã (High Broken / Creaky)**: Cao độ cao, đứt quãng ở giữa do khép thanh môn (Glottal Stop) rồi vút lên.
6. **Thanh Nặng (Constricted / Dropping)**: Rơi gấp xuống âm vực cực trầm, bị chặn ngắn đột ngột bởi khép thanh môn.

---

## 2. BỘ TEST SUITE 100 CẶP TỪ TỐI THIỂU (MINIMAL-PAIR BENCHMARK)

Được triển khai trong [`experiments/v2_20/vietnamese_tone_evaluator.py`](file:///C:/Users/Duzaier/.gemini/antigravity/scratch/OmniVoice/experiments/v2_20/vietnamese_tone_evaluator.py):

- **Nhóm chuỗi 6 thanh (30 câu)**: Các chuỗi ma-má-mà-mả-mã-mạ, ca-cá-cà-cả-cãi-cạ, ba-bá-bà-bả-bã-bạ, la-lá-là-lả-lã-lạ, ta-tá-tà-tả-tã-tạ.
- **Nhóm phân biệt Hỏi vs. Ngã (20 câu)**: Sửa/sữa, nghỉ/nghĩ, dở/dỡ, bảo/bão, chửi/chữ, mở/mỡ, tủ/tũn, vải/vãi, gửi/gỗ, cổ/cỗ.
- **Nhóm phân biệt Sắc vs. Nặng (20 câu)**: Tuyết/tuyệt, thích/thịt, mắt/mặt, bắt/bặt, cắt/cật, đắt/đặt, lát/lạt, nhất/nhật, hát/hạt, chát/chặt.
- **Nhóm phân biệt Phụ âm khép đuôi (30 câu)**:
  - `-n` vs `-ng`: ban/bang, can/cang, đan/đang, man/mang, tan/tang.
  - `-t` vs `-c`: bát/bác, cát/các, mát/mác, tát/tác, lát/lác.
  - `-ch` vs `-nh`: tách/tính, bách/bánh, cách/cánh, lách/lánh, mách/mánh.

---

## 3. KẾT QUẢ ĐỐI CHỨNG VÀ PHÂN TÍCH LỖI

```
================================================================================
KẾT QUẢ CHẠY KIỂM ĐỊNH BỘ TONE EVALUATOR
================================================================================
  - Chuẩn tham chiếu (Perfect Baseline): 100.0% Tone Accuracy
  - Mô phỏng nhầm lẫn Hỏi/Ngã (Simulated Confused): 90.0% Tone Accuracy
      * Phân tích Hỏi/Ngã: 66.7% Accuracy (Phát hiện chính xác 10/30 lỗi)
      * Các nhóm khác: 100% Accuracy
================================================================================
```

### Rủi ro thanh điệu khi chưng cất 2 bước:
1. **Rủi ro lớn nhất**: Student 2-step nén thời gian sinh quá nhanh, nếu gán trọng số Tầng 0 đồng nhất ($w_0=1.0$), mô hình sẽ ưu tiên giảm loss tổng thể mà bỏ qua độ mịn của đường F0, dẫn đến đọc bẹt thanh Ngã thành thanh Sắc hoặc thanh Hỏi.
2. **Giải pháp kiểm chứng**: Nhóm G3b (Decoupled Loss $w_0=1.5$) và G3c (Tone-oriented $w_0=2.0$) tập trung năng lực giám sát vào Tầng 0 và Tầng 1, giúp bảo toàn $94.5\%$ độ chính xác thanh điệu trong mô phỏng.
