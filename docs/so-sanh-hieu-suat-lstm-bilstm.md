# So sánh hiệu suất các model Siamese LSTM/BiLSTM

Tài liệu này tổng hợp kết quả từ 4 notebook huấn luyện:

- `model/train-siamese-lstm-cosine.ipynb`
- `model/train-siamese-lstm-euclid.ipynb`
- `model/train-siamese-bilstm-cosine.ipynb`
- `model/train-siamese-bilstm-euclid.ipynb`

## 1) Bảng metric trên tập test

| Model | Loss/Distance | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| LSTM (traditional) | Cosine | **0.6088** | **0.7596** | **0.8134** | **0.8830** | **0.7020** |
| BiLSTM | Cosine | 0.5236 | 0.7031 | 0.7690 | 0.8472 | 0.6339 |
| BiLSTM | Euclidean | 0.4376 | 0.6055 | 0.6847 | 0.7767 | 0.5501 |
| LSTM (traditional) | Euclidean | 0.4266 | 0.6162 | 0.6857 | 0.7763 | 0.5457 |

## 2) Thông số huấn luyện và kiến trúc

### 2.1 Thông số dùng chung giữa các notebook

- `MAX_VOCAB = 30000`
- `MAX_Q_LEN = 64`
- `MAX_D_LEN = 256`
- `EPOCHS = 20`
- `MIN_DELTA = 1e-4`
- Tiền xử lý văn bản và pipeline dữ liệu retrieval là cùng một hướng thiết kế.

### 2.2 Thông số riêng của từng model

| Model | Kiến trúc encoder | embed_dim | hidden_size | num_layers | dropout | Loss + margin | Patience |
|---|---|---:|---:|---:|---:|---|---:|
| LSTM + Cosine | `SiameseLSTM` (1 chiều) | 200 | 698 | 1 | 0.3 | Triplet Cosine, `margin=0.3` | 5 |
| LSTM + Euclidean | `SiameseLSTM` (1 chiều) | 200 | 698 | 1 | 0.3 | Triplet Euclidean, `margin=0.75` | 5 |
| BiLSTM + Cosine | `SiameseBiLSTM` (2 chiều/bidirectional) | 200 | 256 | 2 | 0.3 | Triplet Cosine, `margin=0.3` | 5 |
| BiLSTM + Euclidean | `SiameseBiLSTM` (2 chiều/bidirectional) | 200 | 256 | 2 | 0.3 | Triplet Euclidean, `margin=0.75` | 3 |

### 2.3 Các điểm model khác nhau rõ nhất

1. **Backbone encoder**
   - Nhóm LSTM dùng `SiameseLSTM` (unidirectional).
   - Nhóm BiLSTM dùng `SiameseBiLSTM` (bidirectional).
2. **Kích thước hidden và số tầng**
   - LSTM: `hidden_size=698`, `num_layers=1`.
   - BiLSTM: `hidden_size=256`, `num_layers=2`.
3. **Hàm loss và margin**
   - Bản cosine: `triplet_loss_cosine(..., margin=0.3)`.
   - Bản euclidean: `triplet_loss_euclidean(..., margin=0.75)`.
4. **Early stopping patience**
   - Hầu hết là `PATIENCE=5`, riêng **BiLSTM + Euclidean** dùng `PATIENCE=3`.
5. **Các thông số còn lại**
   - `embed_dim=200`, `dropout=0.3`, thiết lập dữ liệu và cách đánh giá retrieval tương đối đồng nhất.

## 3) Nhận xét chính

- Với cùng backbone, **Cosine loss** vượt trội so với **Euclidean loss**:
  - LSTM: Recall@1 tăng từ 0.4266 -> 0.6088, MRR tăng từ 0.5457 -> 0.7020.
  - BiLSTM: Recall@1 tăng từ 0.4376 -> 0.5236, MRR tăng từ 0.5501 -> 0.6339.
- So sánh giữa hai backbone khi cùng dùng Cosine:
  - **LSTM (traditional) + Cosine** đang tốt hơn **BiLSTM + Cosine** trên toàn bộ Recall@k và MRR.
- Khi dùng Euclidean, BiLSTM và LSTM khá sát nhau:
  - BiLSTM nhỉnh hơn nhẹ ở Recall@1, Recall@10 và MRR.
  - LSTM nhỉnh hơn nhẹ ở Recall@3.

## 4) Xếp hạng theo MRR (cao -> thấp)

1. **LSTM + Cosine**: 0.7020  
2. **BiLSTM + Cosine**: 0.6339  
3. **BiLSTM + Euclidean**: 0.5501  
4. **LSTM + Euclidean**: 0.5457

## 5) Kết luận ngắn

- Model tốt nhất hiện tại trong các notebook đã chạy là **Siamese LSTM (traditional) + Triplet Cosine**.
- Nếu mục tiêu là tối đa hóa chất lượng retrieval, nên ưu tiên biến thể **Cosine** trước khi tinh chỉnh thêm kiến trúc.
