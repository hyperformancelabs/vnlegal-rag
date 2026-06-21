# VNLegal-RAG: Hệ truy hồi pháp lý tiếng Việt kết hợp TextCNN và Siamese BiLSTM

## Tóm tắt
Tài liệu này mô tả hiện trạng triển khai của dự án `vnlegal-rag`, một pipeline truy hồi văn bản pháp lý tiếng Việt phục vụ bài toán hỏi đáp theo hướng Retrieval-Augmented Generation (RAG). Hệ thống gồm ba khối chính: (i) xử lý dữ liệu từ corpus pháp luật và cặp QA, (ii) bộ phân loại chủ đề TextCNN để ước lượng xác suất `macro_domain`, và (iii) mô hình Siamese BiLSTM học biểu diễn đặc trưng dày để rerank truy hồi. Ở tầng đầu, TF-IDF được dùng để sinh tập ứng viên có độ bao phủ cao; ở tầng sau, các tín hiệu ngữ nghĩa và chủ đề được hợp nhất để cải thiện xếp hạng cuối. Kết quả huấn luyện hiện tại cho thấy mô hình Siamese hội tụ ổn định và cải thiện rõ các chỉ số truy hồi theo epoch.

## 1. Giới thiệu
Hỏi đáp pháp lý tiếng Việt có độ khó cao do đặc thù thuật ngữ chuyên ngành, văn bản dài, và chênh lệch biểu đạt giữa câu hỏi người dùng với ngôn ngữ điều luật. Phương pháp truy hồi thuần từ vựng thường bỏ sót trường hợp đồng nghĩa ngữ nghĩa, trong khi truy hồi thuần dense có thể chưa đủ mạnh với các thuật ngữ hiếm hoặc biểu thức pháp lý cứng.

Dự án áp dụng kiến trúc hybrid để tận dụng ưu điểm của nhiều tín hiệu:
- truy hồi lexical để đảm bảo recall đầu vào,
- xác suất chủ đề để ưu tiên miền pháp lý liên quan,
- độ tương đồng dense để cải thiện thứ hạng ngữ nghĩa.

Mục tiêu trực tiếp là tăng chất lượng top-k passage retrieval (Recall@k, MRR), làm nền cho bước sinh câu trả lời ở tầng RAG.

## 2. Phạm vi và đóng góp của dự án
Ở trạng thái hiện tại, repo đóng góp:
- Luồng dữ liệu đầy đủ từ dữ liệu thô đến tập train/val/test, có cơ chế chống rò rỉ theo nhóm tài liệu.
- Pipeline TextCNN tiếng Việt cho bài toán phân loại `macro_domain`.
- Pipeline Siamese BiLSTM theo triplet learning cho biểu diễn query-passage.
- Thiết kế huấn luyện có xét mất cân bằng lớp và tiêu chí lưu checkpoint theo chất lượng validation.
- Các kỹ thuật ổn định huấn luyện (chuẩn hóa embedding, regularization, mixed precision, scheduler).
- Cơ chế lưu artifact huấn luyện: checkpoint, metadata, từ điển token, lịch sử metric.
- Thiết kế truy hồi hybrid được ghi nhận trong `model/retrieval_results_comparison.md`.

## 3. Data Pipeline
### 3.1 Nguồn dữ liệu và quy trình xử lý
Luồng dữ liệu được triển khai trong `data/` theo ba bước:
- `data/load.ipynb`: chuẩn hóa schema, tạo định danh (`qa_id`, `passage_id`), hỗ trợ gán nhãn miền chủ đề.
- `data/eda.ipynb`: làm sạch dữ liệu (null, trùng lặp, lỗi định dạng), phân tích phân bố, xuất dữ liệu đã xử lý.
- `data/prepare-data.py`: sinh train/val/test bằng tách theo nhóm `doc_name` kết hợp stratification theo nhãn.

Về vận hành, pipeline trước hết bảo đảm tính nhất quán cấu trúc giữa bảng QA và bảng corpus (tên cột, liên kết id, hợp lệ trường văn bản), sau đó bổ sung metadata cần cho huấn luyện. Bước EDA có vai trò kép: kiểm định chất lượng dữ liệu và phân tích đặc trưng phân bố phục vụ thiết kế siêu tham số (độ dài chuỗi, tỷ lệ lớp, mật độ câu hỏi theo miền).

Điểm quan trọng nhất của bước split là chống đánh giá lạc quan do leakage. Thay vì tách ngẫu nhiên từng dòng, pipeline gom nhóm theo `doc_name` để đảm bảo các đoạn thuộc cùng tài liệu không xuất hiện chéo giữa train và test. Cách tách này phù hợp bối cảnh pháp lý, nơi nhiều câu hỏi có thể tham chiếu cùng một văn bản nguồn.

### 3.2 Bộ dữ liệu sẵn sàng huấn luyện
Các file đầu ra trong `data/data_ready/` gồm:
- `qa_train.csv`, `qa_val.csv`, `qa_test.csv`
- `corpus_train.csv`, `corpus_val.csv`, `corpus_test.csv`
- `corpus_full.csv`, `label_maps.json`

Cấu trúc này hỗ trợ hai chế độ sử dụng:
- **Phân loại chủ đề**: dùng `qa_*` để train/evaluate TextCNN.
- **Truy hồi passage**: dùng query từ `qa_*` và không gian ứng viên từ `corpus_*`.

`label_maps.json` đảm bảo ánh xạ nhãn-id nhất quán xuyên suốt train, lưu model và inference. `corpus_full.csv` phục vụ thí nghiệm truy hồi toàn kho, trong khi `corpus_{train,val,test}.csv` hỗ trợ đánh giá kiểm soát theo split.

## 4. Phương pháp (Methodology)
### 4.1 Bộ phân loại chủ đề: TextCNN
Nhánh TextCNN (tham chiếu `model/train-textcnn.ipynb`, `src/models/textcnn.py`) triển khai kiến trúc CNN đa kernel cho phân loại câu:
- tầng embedding token,
- nhiều nhánh `Conv1d` (ví dụ kernel 3/4/5),
- max-over-time pooling trên từng nhánh,
- dropout và lớp tuyến tính cuối để sinh logits đa lớp.

Mô hình xuất phân phối xác suất trên `macro_domain`; phân phối này được dùng như topic prior trong truy hồi hybrid.

Về mặt trực giác, TextCNN phù hợp cho truy vấn pháp lý vì mô hình bắt tốt các cụm từ mẫu và mẫu n-gram ngắn/trung bình xuất hiện lặp lại trong câu hỏi. Max pooling giúp tạo biểu diễn gọn, bền vững hơn trước biến thiên độ dài câu. Trong retrieval, điểm từ TextCNN không thay thế xếp hạng chính mà đóng vai trò tín hiệu bổ trợ để ưu tiên passage cùng miền chủ đề.

Một đóng góp quan trọng so với bối cảnh paper gốc là chiến lược embedding. Thay vì phụ thuộc embedding pretrained phổ biến (chủ yếu tối ưu cho tiếng Anh hoặc miền tổng quát), dự án dùng embedding khởi tạo ngẫu nhiên và cho phép mô hình học trực tiếp từ dữ liệu pháp lý tiếng Việt (CNN-rand). Cách này giúp không gian biểu diễn thích nghi tốt hơn với thuật ngữ chuyên ngành và cách diễn đạt đặc thù của tập dữ liệu.

Ngoài ra, phần loss cũng được điều chỉnh theo thực tế dữ liệu mất cân bằng. Trong khi baseline thường dùng cross-entropy chuẩn, triển khai hiện tại hỗ trợ weighted cross-entropy và focal loss để giảm thiên lệch về lớp phổ biến, cải thiện khả năng học cho các `macro_domain` ít mẫu.

### 4.2 Bộ truy hồi/rerank dense: Siamese BiLSTM
Nhánh Siamese (tham chiếu `model/train-siamese-bilstm.ipynb`, `src/models/siamese.py`) sử dụng:
- encoder BiLSTM dùng chung trọng số cho anchor (query), positive passage và negative passage,
- phép pooling để thu vector biểu diễn mức câu/đoạn,
- chuẩn hóa L2 trên embedding đầu ra,
- triplet loss theo khoảng cách Euclid:

\[
\mathcal{L} = \max(0, m + d(a,p) - d(a,n))
\]

trong đó \(m\) là margin, \(d\) là khoảng cách Euclid, và \((a,p,n)\) là bộ anchor-positive-negative.

Encoder dùng chung buộc query và passage cùng nằm trong một không gian biểu diễn, còn triplet objective học thứ tự tương đối (đúng gần hơn sai ít nhất một biên). Cách học này phù hợp trực tiếp với mục tiêu top-k retrieval hơn so với objective phân loại thuần. BiLSTM giúp nắm bắt phụ thuộc ngữ cảnh hai chiều trong câu pháp lý dài; L2-normalization giúp ổn định thang đo và tăng tính nhất quán khi so sánh khoảng cách.

So với mô tả Siamese BiLSTM truyền thống, thực nghiệm hiện tại có một số điều chỉnh quan trọng: dùng mean pooling trên token hợp lệ (thay cho max pooling), đặt margin ở mức `0.75`, bổ sung gradient clipping để ổn định tối ưu, và triển khai biến thể cosine trong `train-siamese-bilstm-cosine.ipynb` để so sánh trực tiếp với Euclidean. Khi cosine cho kết quả tốt hơn theo cùng điều kiện đánh giá, đây được xem là đóng góp thực nghiệm của nhóm.

### 4.3 Chiến lược truy hồi hybrid
Thiết kế truy hồi hiện tại gồm bốn bước:
1. Sinh ứng viên bằng TF-IDF trên corpus.
2. Tính topic prior từ TextCNN theo mức tương thích `macro_domain`.
3. Tính điểm dense từ embedding Siamese.
4. Hợp nhất điểm và rerank để lấy top-k passage.

Mô hình hybrid cân bằng ba mục tiêu: độ phủ ứng viên (lexical), định hướng miền pháp lý (topic prior), và sát nghĩa ngữ nghĩa (dense rerank).

### 4.4 Các đóng góp kỹ thuật bổ sung của TextCNN và Siamese BiLSTM
Ngoài kiến trúc tổng quan, triển khai hiện tại còn có các đóng góp thực dụng quan trọng cho dữ liệu pháp lý:

**Nhánh TextCNN**
- Cơ chế giảm ảnh hưởng mất cân bằng lớp qua thiết lập tối ưu có trọng số, giúp các lớp `macro_domain` hiếm không bị chìm bởi lớp phổ biến.
- Embedding khởi tạo ngẫu nhiên, học trực tiếp trên dữ liệu pháp lý tiếng Việt (hướng CNN-rand) để giảm lệch miền so với embedding ngoài.
- Điều chỉnh hàm mất mát từ cross-entropy chuẩn sang weighted cross-entropy/focal loss trong bối cảnh dữ liệu imbalance.
- Tổ hợp regularization (dropout ở nhiều vị trí và ràng buộc trọng số) để hạn chế overfit trên các cụm từ pháp lý lặp.
- Chiến lược chọn checkpoint dựa trên hành vi validation thay vì chỉ nhìn loss train, giúp tăng khả năng tổng quát hóa.
- Thiết kế đầu ra theo dạng topic prior tích hợp trực tiếp vào bước fusion retrieval, thay vì dùng classifier như một mô-đun tách rời.

**Nhánh Siamese BiLSTM**
- Thiết kế metric learning với encoder dùng chung cho query/passage, tạo không gian embedding tối ưu cho xếp hạng.
- Dùng mean pooling (trung bình theo token hợp lệ) như lựa chọn phù hợp cho chuỗi pháp lý dài.
- Chuẩn hóa L2 trước khi tính khoảng cách để ổn định thang đo và tăng tính nhất quán giữa batch.
- Cấu hình margin `0.75` kết hợp gradient clipping để hạn chế dao động gradient trong huấn luyện triplet.
- Cách tạo triplet theo liên kết thực (positive theo passage_id, negative theo lấy mẫu thực dụng và định hướng miền) giúp bài học gần thực tế truy hồi hơn random negative thuần.
- Hard negative sampling theo cùng `macro_domain`, tạo negative khó hơn và cải thiện năng lực phân biệt passage gần nghĩa.
- Nhánh thử nghiệm cosine (`train-siamese-bilstm-cosine.ipynb`) để đánh giá lựa chọn metric ngoài Euclidean.
- Vòng đánh giá trong huấn luyện theo Recall@k và MRR, nên quyết định tối ưu bám sát mục tiêu retrieval downstream.

Những điểm này giúp hệ thống vượt khỏi mức mô hình mẫu, tiến gần hơn tới một pipeline truy hồi pháp lý có tính ứng dụng.

## 5. Tổ chức triển khai trong repository
### 5.1 Các module tái sử dụng (`src/`)
- `src/datasets/`: lớp dataset cho classification và triplet retrieval.
- `src/models/`: định nghĩa `TextCNN`, `BiLSTMEncoder`, `SiameseBiLSTM`.
- `src/encoders/tfidf.py`: bộ mã hóa lexical baseline.
- `src/training/`: khung vòng lặp train và utility metric.

### 5.2 Notebook thí nghiệm (`model/`)
- `train-textcnn.ipynb`
- `train-siamese-bilstm.ipynb`
- `train-siamese-bilstm-cosine.ipynb`
- `test-pipeline.ipynb`

### 5.3 Artifact huấn luyện
Các đầu ra chính gồm:
- checkpoint mô hình (`*.pt`),
- metadata huấn luyện (`*_meta.json`),
- từ điển tokenizer (`tokenizer_vocab.json`),
- lịch sử train (`train_history.csv`).

## 6. Quan sát thực nghiệm (trạng thái hiện tại)
Theo log huấn luyện đang có ở nhánh Siamese:
- loss giảm ổn định theo epoch,
- metric truy hồi tăng dần trong quá trình train,
- điểm tốt nhất quan sát được xấp xỉ:
  - `MRR ≈ 0.5794`
  - `R@1 ≈ 0.4633`
  - `R@5 ≈ 0.7173`

Kết quả này cho thấy không gian embedding đã học được quan hệ ngữ nghĩa có ích cho truy hồi pháp lý. Giá trị cụ thể có thể dao động theo seed, split và chiến lược lấy negative.

## 7. Thảo luận
### 7.1 Điểm mạnh
- Kiến trúc hybrid tăng độ bền so với truy hồi một tín hiệu đơn.
- Topic prior từ classifier hữu ích trong kho dữ liệu có cấu trúc miền rõ.
- Tách dữ liệu theo nhóm tài liệu giúp đánh giá đáng tin cậy hơn.

### 7.2 Hạn chế hiện tại
- Quy trình còn phụ thuộc notebook, chưa tối ưu cho CI/CD tự động.
- Một số thành phần pipeline CLI còn ở mức khung.
- Đánh giá hiện tập trung vào retrieval, chưa chuẩn hóa đầy đủ metric end-to-end cho generated answer.

## 8. Hướng phát triển
- Chuẩn hóa pipeline huấn luyện/suy luận thành script hoặc CLI.
- Thực hiện ablation có kiểm soát: TF-IDF only, dense only, hybrid + topic prior.
- Benchmark cosine và Euclidean trong cùng điều kiện chuẩn hóa.
- Bổ sung đánh giá đầu-cuối RAG (độ trung thực, độ chính xác trích dẫn, nhất quán pháp lý).

## 9. Kết luận
`vnlegal-rag` là nền tảng thực nghiệm khả thi cho truy hồi pháp lý tiếng Việt trong hệ RAG. Bằng cách kết hợp truy hồi lexical, prior theo chủ đề và rerank dense theo Siamese, dự án hướng tới cải thiện chất lượng top-k passage trong bối cảnh dữ liệu pháp lý thực tế. Cấu trúc repo và artifact hiện tại đã đủ để lặp thí nghiệm, so sánh mô hình, và tiến tới pipeline sản xuất.

## 10. Ablation Study (đề xuất thực nghiệm)
Để lượng hóa rõ đóng góp của từng thành phần, nên chạy ma trận ablation sau:
- **A0 (baseline lexical)**: chỉ dùng TF-IDF.
- **A1**: TF-IDF + topic prior từ TextCNN.
- **A2**: TF-IDF + Siamese rerank (Euclidean).
- **A3**: TF-IDF + Siamese rerank (Cosine).
- **A4 (hybrid đầy đủ)**: TF-IDF + TextCNN prior + Siamese rerank.

Ablation riêng cho TextCNN:
- **T0**: embedding pretrained/tổng quát + cross-entropy chuẩn.
- **T1**: CNN-rand + cross-entropy chuẩn.
- **T2**: CNN-rand + weighted cross-entropy.
- **T3**: CNN-rand + focal loss.

Ablation riêng cho Siamese:
- **S0**: max pooling + random negative + Euclidean.
- **S1**: mean pooling + random negative + Euclidean.
- **S2**: mean pooling + hard negative cùng `macro_domain` + Euclidean.
- **S3**: mean pooling + hard negative + Cosine.

Các chỉ số chính nên báo cáo gồm `Recall@1`, `Recall@5`, `MRR` và độ trễ suy luận. Thiết kế này giúp truy vết trực tiếp tác động của (i) embedding thích nghi miền, (ii) loss xử lý imbalance, (iii) hard negative sampling và (iv) lựa chọn metric khoảng cách.

## 11. Threats to Validity (rủi ro hợp lệ thực nghiệm)
Một số yếu tố có thể ảnh hưởng độ tin cậy của kết quả:
- **Độ nhạy theo split**: hiệu năng có thể biến động theo cách chia nhóm tài liệu, đặc biệt ở domain ít mẫu.
- **Thiên lệch negative sampling**: chất lượng hard negative phụ thuộc mạnh vào tập ứng viên và độ đúng của nhãn miền.
- **Trạng thái ẩn trong notebook**: chạy cell không tuần tự có thể tạo sai lệch nếu seed/config không được reset nhất quán.
- **Rò rỉ metric**: nếu lọc corpus/split không chặt, Recall@k và MRR có thể bị đánh giá cao giả tạo.
- **Giới hạn độ phủ miền pháp lý**: tập hiện tại có thể chưa bao trùm hết các tiểu miền hoặc cách diễn đạt pháp lý mới.

Để giảm rủi ro, cần chạy lặp nhiều seed, kiểm toán split định kỳ và chuẩn hóa script đánh giá.

## 12. Reproducibility Checklist
Để tăng khả năng tái lập:
- Cố định và ghi lại seed cho Python/NumPy/PyTorch.
- Lưu đầy đủ cấu hình train (mô hình, loss, margin, metric, pooling, chế độ lấy negative).
- Version hóa artifact (`*.pt`, `*_meta.json`, `train_history.csv`, vocab) kèm tag thí nghiệm.
- Báo cáo cả điểm trung bình và điểm tốt nhất trên nhiều seed.
- Duy trì tập đánh giá cố định và script chuẩn để tính lại `Recall@k`, `MRR` từ checkpoint đã lưu.
- Ghi rõ bối cảnh phần cứng/thời gian chạy (GPU, batch size, AMP bật/tắt, thời lượng train).

## Tài liệu tham khảo
[1] Y. Kim, "Convolutional Neural Networks for Sentence Classification," EMNLP, 2014.  
[2] Các tài liệu metric learning theo hướng triplet loss và Siamese networks.  
[3] Tài liệu nội bộ repo: `MO_TA_CAU_TRUC_DU_AN.md`, `MO_TA_KIEN_TRUC_TEXTCNN.md`, `model/retrieval_results_comparison.md`.
