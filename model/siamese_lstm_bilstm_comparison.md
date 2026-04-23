# So sánh Siamese LSTM và Siamese BiLSTM trên tập `data_ready`

Tài liệu tóm tắt cấu hình huấn luyện và metric retrieval trên **cùng pipeline dữ liệu** trong các notebook:

- `train-siamese-lstm-cosine.ipynb`
- `train-siamese-lstm-euclid.ipynb`
- `train-siamese-bilstm-cosine.ipynb`
- `train-siamese-bilstm-euclid.ipynb`

Các số **test** dưới đây lấy trực tiếp từ output `print('Test metrics:', ...)` đã ghi trong notebook (môi trường log: GPU Tesla T4, dữ liệu Kaggle `data_ready`). Nếu bạn train lại trên máy khác, metric có thể lệch nhẹ do seed / phiên bản thư viện.

---

## 1. Dữ liệu và tác vụ

Tất cả notebook dùng chung logic:

| Tập | Kích thước (theo output notebook) |
|-----|-------------------------------------|
| QA train / corpus train | 23 311 × 14 / 7 771 × 6 |
| QA val / corpus val | 2 841 × 14 / 947 × 6 |
| QA test / corpus test | 2 991 × 14 / 997 × 6 |

- Cột câu hỏi: `question`
- Cột đoạn văn: `article_content`
- Huấn luyện **triplet** (anchor = câu hỏi, positive = đoạn đúng, negative lấy ưu tiên cùng `macro_domain` nếu có)
- Tokenizer: ưu tiên `underthesea`; trên log Kaggle dùng **fallback regex** → vocab ~**5 800** từ tập train
- Giới hạn: `MAX_Q_LEN = 64`, `MAX_D_LEN = 256`, `MAX_VOCAB = 30 000` (thực tế nhỏ hơn do dữ liệu)
- Loader: batch **32**, tối ưu **Adam** `lr = 5e-4`, scheduler **ReduceLROnPlateau** theo MRR validation (subsample **1500** câu hỏi khi eval val)

**Đánh giá retrieval test:** encode toàn bộ `corpus_test`, với mỗi câu trong `qa_test` xếp hạng đoạn theo độ tương thích; báo cáo Recall@k và MRR (đủ **2 991** mẫu test).

---

## 2. Khác biệt kiến trúc và loss

| Thuộc tính | LSTM (1 chiều) | BiLSTM |
|------------|----------------|--------|
| Lớp `nn.LSTM` | `bidirectional=False` | `bidirectional=True` |
| `num_layers` (mặc định trong code) | **1** | **2** |
| `embed_dim` / `hidden_size` | 200 / 256 | 200 / 256 |
| Pooling | Trung bình có mask trên chuỗi output | Giống LSTM |
| Chuẩn hóa vector embedding | L2 (`F.normalize`) | L2 (`F.normalize`) |

**Loss và margin (theo từng notebook):**

| Variant | Hàm loss | `margin` | Cách xếp hạng lúc eval |
|---------|-----------|----------|-------------------------|
| *cosine* | `relu(margin - cos(a,p) + cos(a,n))` | **0.3** | Cosine → tích vô hướng với ma trận passage (cùng hướng vì vector đã L2) |
| *euclid* | `relu(margin + ‖a-p‖₂ - ‖a-n‖₂)` | **0.75** | Khoảng cách L2 (`torch.cdist` hoặc `pairwise_distance`) |

**Huấn luyện:** mixed precision (`GradScaler` + `autocast`), clip gradient 1.0, early stopping theo MRR val (lưu ý: notebook BiLSTM Euclidean đặt `PATIENCE = 3`, các notebook LSTM/BiLSTM Cosine dùng `PATIENCE = 5`).

---

## 3. Kết quả trên tập test (2991 truy vấn)

| Mô hình | Loss / metric | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---------|-----------------|----------|----------|----------|-----------|-----|
| **LSTM + Cosine** | Cosine triplet | **0.619** | **0.772** | **0.829** | **0.900** | **0.715** |
| LSTM + Euclidean | Euclidean triplet | 0.458 | 0.636 | 0.704 | 0.787 | 0.572 |
| BiLSTM + Cosine | Cosine triplet | 0.524 | 0.703 | 0.769 | 0.847 | 0.634 |
| BiLSTM + Euclidean | Euclidean triplet | 0.438 | 0.605 | 0.685 | 0.777 | 0.550 |

### Nhận xét ngắn

1. **Cosine vượt Euclidean** rõ rệt cho cả LSTM lẫn BiLSTM trên bộ legal QA này — phù hợp với việc embedding đã L2-normalize và retrieval dùng inner product tương đương cosine.
2. **LSTM một chiều (1 layer)** trong cấu hình notebook hiện tại **tổng quát tốt hơn BiLSTM hai layer hai chiều** trên metric test, dù BiLSTM có nhiều tham số hơn. Khả năng gồm: dữ liệu/tách từ tương đối nhỏ, hoặc mô hình sâu hơn dễ khớp tập train nhưng không cải thiện val/test.
3. **Không so sánh công bằng tuyệt đối về độ sâu:** LSTM dùng `num_layers=1`, BiLSTM dùng `num_layers=2`; muốn kết luận “do hướng hay do độ sâu” cần thêm thí nghiệm cố định `num_layers` và chiều ẩn tương đương số tham số.

---

## 4. Ghi chú về artifact trong repo

- Checkpoint trong `model/siamese_lstm_artifacts/` (ví dụ `siamese_bilstm_best.pt`, `train_history.csv`) có thể tương ứng **một lần chạy cụ thể**, không nhất thiết trùng hoàn toàn với bảng test ở trên nếu file đã bị ghi đè sau lần train khác.
- Để tái lập số liệu, nên chạy lại notebook tương ứng và đối chiếu `train_history.csv` + cell test cuối.

---

## 5. Tóm tắt lựa chọn thực tế

Trên tập `data_ready` và pipeline hiện tại, **Siamese LSTM + triplet cosine** cho MRR và Recall@k test cao nhất trong bốn cấu hình đã train. Nếu tiếp tục cải thiện BiLSTM, nên thử: giảm overfit (dropout, ít layer hơn), chỉnh `PATIENCE`/epoch, hoặc căn chỉnh margin và cách lấy negative — vẫn giữ **cosine** làm baseline ưu tiên.
