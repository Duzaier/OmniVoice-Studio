# V2 Two-Sequence Forensic Report

## 1. Forward Pass Structure & Source Trace

Truy vết đường đi từ lời gọi API cấp cao đến khối tính toán ma trận lõi:

```
OmniVoice.generate()
       ↓
OmniVoice._generate_chunked() (hoặc _generate_direct())
       ↓
OmniVoice._generate_iterative()  [Line 1276, omnivoice/models/omnivoice.py]
       ↓
Line 1321: batch_input_ids = torch.full((2 * B, num_audio_codebook, max_c_len), ...)
Line 1327: batch_audio_mask = torch.zeros((2 * B, max_c_len), ...)
Line 1330: batch_attention_mask = torch.zeros((2 * B, 1, max_c_len, max_c_len), ...)
       ↓
Vòng lặp khuếch tán: for step in range(gen_config.num_step):
       ↓
Line 1386: batch_logits = self(input_ids=batch_input_ids, audio_mask=batch_audio_mask, attention_mask=batch_attention_mask)
```

---

## 2. Tensor Shapes in Runtime Execution

Khi suy luận 1 mẫu văn bản đơn lẻ ($B = 1$, văn bản Test A 220 ký tự, `target_lens = 338` tokens âm thanh, tổng độ dài `c_len = 410` tokens):

| Tensor Name | Shape `[B, T, D]` / `[B, C, T]` | Ý nghĩa kích thước |
| :--- | :--- | :--- |
| `batch_input_ids` | `[2, 8, 410]` | **Batch size = 2** (2 * B), 8 Audio Codebooks, Sequence Length 410 |
| `batch_audio_mask` | `[2, 410]` | Mask nhị phân xác định vị trí token âm thanh cần unmask |
| `batch_attention_mask` | `[2, 1, 410, 410]` | Causal / Prefix Attention Matrix 2D cho 2 sequences |
| `batch_logits` (Output) | `[2, 8, 410, 2049]` | Logits dự đoán cho 8 codebook heads, Vocabulary 2049 |
| `c_logits` (Cond slice) | `[1, 8, 338, 2049]` | Logits của Sequence 1 cắt từ index `c_len - t_len : c_len` |
| `u_logits` (Uncond slice) | `[1, 8, 338, 2049]` | Logits của Sequence 2 cắt từ index `0 : t_len` |

---

## 3. Why Batch = 2: Ý nghĩa & Vai trò của 2 Sequences

Kiến trúc `k2-fsa/OmniVoice` thiết kế `batch_size = 2 * B` phục vụ **Classifier-Free Guidance (CFG)**:

### 🔹 Role of Sequence 1 (Index `0`): Conditional Stream ($p(x|c)$)
- **Nội dung tensor:** `[Style tokens + Text tokens + Ref Voice tokens + Target Audio Mask tokens]`
- **Mã nguồn:** Lines 1338–1340 trong `omnivoice/models/omnivoice.py`.
- **Vai trò:** Mô hình Transformer đọc toàn bộ ngữ cảnh (văn bản tiếng Việt + đặc trưng giọng mẫu của người nói) để sinh ra phân phối xác suất âm thanh có điều kiện $p(x \mid \text{text}, \text{voice})$.

### 🔹 Role of Sequence 2 (Index `1`): Unconditional Stream ($p(x)$)
- **Nội dung tensor:** `[Target Audio Mask tokens]` (Cắt bỏ hoàn toàn tiền tố text, style và reference voice).
- **Mã nguồn:** Lines 1343–1345: `batch_input_ids[B + i, :, :u_len] = inp["input_ids"][..., -u_len:]`.
- **Vai trò:** Mô hình sinh ra phân phối âm thanh vô điều kiện $p(x)$ (tiếng nói tổng quát không phụ thuộc vào text/giọng mẫu) để làm mốc đối chiếu khử nhiễu.

---

## 4. CFG Behavior Across Guidance Scales

Hàm tính xác suất và chấm điểm `_predict_tokens_with_scoring` (Lines 1430–1439):

```python
if gen_config.guidance_scale != 0:
    c_log_probs = F.log_softmax(c_logits, dim=-1)
    u_log_probs = F.log_softmax(u_logits, dim=-1)
    log_probs = torch.log_softmax(
        c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
        dim=-1,
    )
else:
    log_probs = F.log_softmax(c_logits, dim=-1)
```

| `guidance_scale` | Batch Size trong Transformer | Forward Pass / Step | Cách xử lý Logits | Trạng thái sử dụng Sequence 2 |
|---:|:---:|:---:|:--- |:--- |
| **0.0** | **2** | **2** | `log_probs = log_softmax(c_logits)` | ❌ **BỊ VỨT BỎ HOÀN TOÀN** (`u_logits` không bao giờ được đọc) |
| **1.0** | **2** | **2** | `c_log_probs + 1.0 * (c_log_probs - u_log_probs)` | ✅ Sử dụng cho CFG |
| **1.5** | **2** | **2** | `c_log_probs + 1.5 * (c_log_probs - u_log_probs)` | ✅ Sử dụng cho CFG |
| **2.0** | **2** | **2** | `c_log_probs + 2.0 * (c_log_probs - u_log_probs)` | ✅ Sử dụng cho CFG |

> [!CAUTION]
> **Phát hiện quan trọng:** 
> Khi `guidance_scale = 0.0`, mô hình **chỉ sử dụng `c_logits`** để chọn token. Nhưng vì tensor đầu vào bị fix cứng `batch_size = 2 * B` ở Line 1321, CPU vẫn phải tính toán trọn vẹn 28 layer cho Sequence 2 rồi bỏ đi.

---

## 5. Can They Be Fused / Can Sequence 2 Be Eliminated?

### **Câu trả lời: YES (Về mặt nguyên lý kiến trúc)**

- **Lý do kỹ thuật:** Khi `guidance_scale == 0.0`, không có bước nội suy/ngoại suy CFG. Toàn bộ logic dự đoán chỉ cần `c_logits`.
- Nếu chỉ truyền `batch_size = B = 1` (Conditional sequence duy nhất), Transformer chỉ cần tính 1 lượt forward pass cho mỗi diffusion step thay vì 2.
- **Lưu ý:** Không áp dụng khi người dùng muốn dùng CFG (`guidance_scale > 0`).

---

## 6. Estimated Theoretical Speedup (Tính toán lý thuyết)

- **Số lượt Forward Pass cho V2 Turbo (4 steps):** Giảm từ **8 lượt $\rightarrow$ 4 lượt**.
- **Khối lượng tính toán Diffusion:** Giảm đúng **50.0%**.
- **Tác động lên tổng Pipeline V2:**
  - Thời gian Diffusion đo được hiện tại: `40.73 s`
  - Thời gian Diffusion lý thuyết (1 pass): `~20.36 s`
  - Các phần khác (Vocoder decode, text prep, I/O): `~4.03 s`
  - **Tổng thời gian tạo audio lý thuyết:** `~24.39 s` cho `12.7 s` audio.
  - **Lý thuyết RTF:** Từ `3.52x` giảm xuống **`~1.92x`**.
