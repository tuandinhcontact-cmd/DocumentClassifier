# Dự Án Phân Loại Tin Tức Đa Lớp Cân Bằng (News Classification Project)

Dự án này ghi lại hành trình từ việc xây dựng mô hình học máy cơ bản, thử nghiệm các kỹ thuật xử lý dữ liệu khác nhau, giải quyết mất cân bằng dữ liệu cực kỳ nghiêm trọng, cho đến việc thiết kế cấu trúc phân loại tùy chỉnh **Hierarchical Cascade Classifier** để tối ưu hóa hiệu năng và tốc độ trên tập dữ liệu hơn **182,000 dòng** (gồm 14 danh mục tin tức).

---

## 1. Tổng Quan Quy Trình Thử Nghiệm

Dự án đã trải qua 4 giai đoạn thử nghiệm chính với kết quả như sau:

| Thử nghiệm | Phương pháp Trích xuất | Thuật toán Phân loại | Quy mô Dữ liệu | Độ chính xác (Accuracy) | Ưu/Nhược điểm |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Thử nghiệm 1** | TF-IDF (1,500 features) | Ensemble (Soft Voting) gồm: LR, Gaussian NB, SVC, Random Forest | 5,000 mẫu | **42.10%** | **Nhược**: Gaussian NB hoạt động rất kém trên TF-IDF, kéo tụt hiệu năng Ensemble. |
| **Thử nghiệm 2** | Word2Vec (100 dims, trained from scratch) | Ensemble (Soft Voting) gồm: LR, Multinomial NB (MinMax scaled), SVC, RF | 10,000 mẫu | **22.75%** | **Nhược**: Dữ liệu train quá ít để Word2Vec học ngữ nghĩa sâu. |
| **Thử nghiệm 3** | TF-IDF (2,500 features) | So sánh: Baseline vs Class Weighting vs SMOTE | 25,000 mẫu | **62.52% - 72.52%** | **Nhận xét**: SMOTE và Class Weighting tăng F1-score lớp thiểu số. Class Weighting chạy nhanh gấp 7 lần SMOTE. |
| **Thử nghiệm 4 (Tối ưu)** | **TF-IDF (30,000 features, Bigrams)** | **Hierarchical Cascade Classifier** (Logistic Regression + Class Weight) | **182,081 mẫu (Full)** | **78.18%** | **Ưu**: Tách lớp lớn trước, lớp nhỏ sau. Tốc độ cực nhanh (1.5s), hiệu năng cao nhất. |

---

## 2. Chi Tiết Phương Pháp & Các Tệp Tin Mã Nguồn

### Bước 1: Thu Thập & Gộp Dữ Liệu Sạch (`merge_and_clean.py`)
Chúng tôi gộp 3 nguồn dữ liệu khác nhau thành một tập dữ liệu chuẩn duy nhất có tên `merged_cleaned_dataset.csv` (182,081 dòng):
* **`final_data_14_custom_labels.csv`**: Dữ liệu gốc chứa 14 nhãn (52,482 dòng).
* **`BBC_dataset.csv`**: Bộ dữ liệu tin tức BBC tiếng Anh (2,225 dòng, đọc dưới dạng encoding `cp1252`).
* **`news.csv`**: Bộ dữ liệu tin tức bổ sung (127,393 dòng).

**Quy tắc xử lý văn bản**:
1. Ghép trường `headline` và `short_description` thành một chuỗi duy nhất.
2. Chuyển về chữ thường, loại bỏ các ký tự đặc biệt, chữ số, và dấu câu.
3. Loại bỏ stopword tiếng Anh (sử dụng danh sách chuẩn từ scikit-learn).
4. Ánh xạ các nhãn khác nhau về đúng 14 danh mục thống nhất.

*Mã nguồn thực hiện*: [merge_and_clean.py](file:///Users/gtuan/.gemini/antigravity/scratch/ensemble_learning/merge_and_clean.py)

---

### Bước 2: Thử Nghiệm Xử Lý Mất Cân Bằng (`train_imbalanced.py`)
Do dữ liệu mất cân bằng cực kỳ nghiêm trọng (`Tech & Science` chiếm ~78k mẫu trong khi `Education` chỉ có ~1.8k mẫu), chúng tôi đã so sánh 3 phương pháp trên ma trận đặc trưng TF-IDF:
1. **Baseline**: Huấn luyện thông thường.
2. **Class Weighting (`class_weight='balanced'`)**: Phạt nặng hơn khi dự đoán sai lớp thiểu số.
3. **SMOTE**: Sinh thêm dữ liệu nhân tạo cho lớp thiểu số.

**Kết luận**: **Class Weighting** tối ưu hơn hẳn **SMOTE** đối với dữ liệu văn bản vì cho độ chính xác tương đương (Macro F1-score ~43.9%) nhưng thời gian chạy nhanh gấp 7 lần và không tốn bộ nhớ RAM.

*Mã nguồn thực hiện*: [train_imbalanced.py](file:///Users/gtuan/.gemini/antigravity/scratch/ensemble_learning/train_imbalanced.py)

---

### Bước 3: Giải Pháp Tối Ưu - Hierarchical Cascade Classifier (`train_cascade.py`)
Mô hình phân tầng hoạt động theo cơ chế huấn luyện một chuỗi các bộ phân loại nhị phân (One-vs-Rest) nối tiếp nhau theo thứ tự phân bổ số lượng mẫu từ lớn đến nhỏ:
* **Model 1**: Dự đoán xem văn bản có thuộc lớp lớn nhất `Tech & Science` hay không. Nếu có, dừng dự đoán; nếu không, chuyển dữ liệu còn lại cho mô hình tiếp theo.
* **Model 2**: Dự đoán xem văn bản có thuộc lớp lớn nhì `Politics and society` hay không.
* Cứ như thế bóc tách dần cho tới lớp cuối cùng.

Cơ chế này giúp loại bỏ sự áp đảo của các lớp đa số lên lớp thiểu số, mang lại độ chính xác ấn tượng **78.18%** trên toàn bộ tập dữ liệu chỉ trong **1.50 giây** huấn luyện.

*Mã nguồn thực hiện*: [train_cascade.py](file:///Users/gtuan/.gemini/antigravity/scratch/ensemble_learning/train_cascade.py)

---

### Bước 4: Hướng Phát Triển Tương Lai - Deep Learning (`train_bert.py`)
Nếu bạn muốn nâng độ chính xác vượt ngưỡng **80% - 85%**, chúng tôi cung cấp tệp cấu hình để fine-tune mô hình học sâu **DistilBERT** tiền huấn luyện của Hugging Face.

*Mã nguồn thực hiện*: [train_bert.py](file:///Users/gtuan/.gemini/antigravity/scratch/ensemble_learning/train_bert.py)

---

## 3. Hướng Dẫn Cài Đặt & Chạy Chương Trình

### 1. Cài đặt các thư viện yêu cầu:
```bash
/usr/bin/python3 -m pip install --user numpy pandas scikit-learn matplotlib imbalanced-learn gensim torch transformers datasets accelerate
```

### 2. Gộp và làm sạch dữ liệu:
Đảm bảo các file dữ liệu gốc (`final_data_14_custom_labels.csv`, `BBC_dataset.csv`, `news.csv`) nằm trong cùng thư mục, sau đó chạy:
```bash
/usr/bin/python3 merge_and_clean.py
```

### 3. Huấn luyện mô hình tối ưu (Cascade Classifier):
```bash
/usr/bin/python3 train_cascade.py
```

### 4. Thử nghiệm huấn luyện BERT (Yêu cầu GPU/MPS để chạy nhanh):
```bash
/usr/bin/python3 train_bert.py
```
