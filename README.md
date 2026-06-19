# 📰 Document Classifier: 3-Step Hybrid Cascade System

Dự án phân loại chủ đề tin tức văn bản tự động (News Classification), được phát triển dựa trên việc thiết kế cấu trúc phân tầng dữ liệu (Data Flow) thay vì phụ thuộc hoàn toàn vào các thư viện đóng gói sẵn. 

Hệ thống có khả năng phân loại bài báo vào **14 chủ đề khác nhau** với điểm nhấn kỹ thuật là toàn bộ thuật toán Toán học lõi được **xây dựng thủ công (Code From Scratch)** bằng NumPy, đồng thời áp dụng kiến trúc thác nước 3 tầng để khắc phục triệt để sự mất cân bằng dữ liệu (Imbalanced Data).

---

## 🚀 Tính năng nổi bật & Thành tựu
- **Thuật toán tự phát triển 100%:** Các thuật toán `CustomLogisticRegression` (với Adam Optimizer) và `CustomMultinomialNB` (với Log-Sum-Exp Trick) được lập trình hoàn toàn từ con số 0.
- **Kiến trúc 3-Step Cascade:** Cô lập và "bóc tách" dần các nhãn áp đảo (Tech & Politics chiếm >70% dữ liệu), giúp mô hình tập trung bảo vệ độ chính xác cho các nhãn thiểu số còn lại.
- **Ensemble OVR Soft-Voting:** Kết hợp ma trận xác suất L1 Normalization của mô hình NB và mô hình LR đa lớp (nhân bản 12 OVR) để bù trừ sai số.
- **Hiệu năng xuất sắc:** 
  - Đạt độ chính xác **76.77%** trên 182,000 bài báo chỉ với phương pháp TF-IDF truyền thống. 
  - **Đánh bại thuật toán tối ưu C++** của thư viện `scikit-learn` về cả tốc độ hội tụ (train xong trong vòng 90 giây) lẫn độ chính xác.
- **Giao diện Web Trực quan (Flask):** Cho phép người dùng theo dõi tận mắt từng bước đi của bài báo qua hệ thống Cascade.

---

## 📁 Cấu trúc thư mục (Chuẩn công nghiệp)
```text
DocumentClassifier/
│
├── data/                       # Chứa file CSV dataset (Đã tiền xử lý, gộp Headline và Short Desc)
├── custom_models/              # Mã nguồn Toán học cốt lõi (Logistic Regression, Naive Bayes...)
├── static/ & templates/        # Giao diện Web HTML/CSS/JS
│
├── archive_experiments/        # Gom các file chạy thử nghiệm, test thuật toán cũ (Bert, Sklearn...)
├── old_models/                 # Nơi lưu trữ các file trọng số (.pkl, .npy) cũ
│
├── train_3step_cascade.py      # File chính để đào tạo mô hình (Train Model)
├── app.py                      # File chính để chạy Web Server
├── cascade_model.pkl           # Trọng số mô hình đỉnh cao nhất hiện tại (F1 56.71%)
└── Bao_cao_Mo_hinh.md          # Báo cáo lý thuyết và kiến trúc chi tiết
```

---

## 🛠 Hướng dẫn cài đặt và chạy thử

### 1. Cài đặt thư viện
Bạn cần cài đặt các thư viện cơ bản sau (Khuyến nghị dùng môi trường ảo `venv` hoặc `conda`):
```bash
pip install pandas numpy scikit-learn flask
```

### 2. Huấn luyện mô hình
Nếu bạn muốn tự tay huấn luyện lại mô hình từ tập dữ liệu `data/merged_cleaned_dataset.csv`, hãy chạy lệnh sau:
```bash
python train_3step_cascade.py
```
*Quá trình này sẽ diễn ra rất nhanh (chỉ mất chưa tới 2 phút trên CPU).*

### 3. Khởi chạy Giao diện Web
Để test thử khả năng phân loại bài báo của mô hình qua Web UI trực quan, hãy chạy:
```bash
python app.py
```
Sau đó mở trình duyệt và truy cập vào đường link: `http://127.0.0.1:5000`

---

## 📚 Báo cáo chuyên sâu
Toàn bộ tư duy thuật toán, cách xử lý lỗi tràn số toán học, công thức Adam Optimizer cũng như các thử nghiệm thất bại đều được ghi chép và phân tích cẩn thận tại file: `Bao_cao_Mo_hinh.md`. Mời bạn đọc để hiểu chi tiết vòng đời kiến trúc phần mềm này!
