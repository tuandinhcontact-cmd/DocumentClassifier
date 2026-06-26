# 📰 Document Classifier: Flat Soft Voting Ensemble System

Dự án phân loại chủ đề tin tức văn bản tự động (News Classification), được phát triển dựa trên việc thiết kế cấu trúc phẳng biểu quyết mềm (Flat Soft Voting Ensemble) từ nhiều mô hình học máy thành phần tự code.

Hệ thống có khả năng phân loại bài báo vào **13 chủ đề khác nhau** với điểm nhấn kỹ thuật là toàn bộ thuật toán Toán học lõi được **xây dựng thủ công (Code From Scratch)** bằng NumPy.

---

## 🚀 Tính năng nổi bật & Thành tựu
- **Thuật toán tự phát triển 100%:** Các thuật toán `CustomLogisticRegression` (với Adam Optimizer), `CustomLinearSVM` (với Platt Scaling), và `CustomMultinomialNB` (với Log-Sum-Exp Trick) được lập trình hoàn toàn từ con số 0.
- **Ensemble Soft-Voting:** Kết hợp ma trận xác suất của 3 mô hình tự code độc lập: MultinomialNB, Logistic Regression OVR (Adam), và Linear SVM OVR (Platt Scaling) để đưa ra phán quyết cuối cùng hoàn hảo nhất.
- **Hiệu năng xuất sắc:** 
  - Đạt độ chính xác **70.03%** (Macro F1-Score **68.17%**) trên dữ liệu tin tức gộp HuffPost, BBC, và 4 tập bổ sung (Education, Technology, Sports, Business).
  - **Đánh bại thuật toán tối ưu C++** của thư viện `scikit-learn` về cả tốc độ hội tụ lẫn độ chính xác trên không gian đặc trưng thưa của TF-IDF.
- **Giao diện Web Trực quan (Flask):** Cho phép người dùng theo dõi tận mắt xác suất biểu quyết của từng mô hình thành phần trong hội đồng (Ensemble Voting Path).

---

## 📁 Cấu trúc thư mục (Chuẩn công nghiệp)
```text
DocumentClassifier/
│
├── data/                       # Chứa file CSV dataset (BBC, HuffPost, Education, Tech, Sports, Business)
├── custom_models/              # Mã nguồn Toán học cốt lõi (Logistic Regression, Naive Bayes...)
├── static/                     # Giao diện Web HTML/CSS/JS (biểu đồ phân bố...)
├── models/                     # Nơi lưu trữ các tệp trọng số mô hình đã huấn luyện (.pkl)
├── training_scripts/           # Gom các file huấn luyện mô hình, grid search
├── logs/                       # Chứa các tệp ghi nhật ký huấn luyện (train logs)
├── scratch/                    # Các file chạy nháp, thực nghiệm phụ trợ
│
├── app.py                      # File chính để chạy Web Server
├── README.md                   # Hướng dẫn dự án
├── Bao_cao_Mo_hinh.md          # Báo cáo lý thuyết và kiến trúc chi tiết
└── Bao_cao_Chuan_bi_Du_lieu.md # Báo cáo tiền xử lý và chuẩn bị dữ liệu
```

---

## 🛠 Hướng dẫn cài đặt và chạy thử

### 1. Cài đặt thư viện
Bạn cần cài đặt các thư viện cơ bản sau (Khuyến nghị dùng môi trường ảo `venv` hoặc `conda`):
```bash
pip install pandas numpy scikit-learn flask matplotlib seaborn
```

### 2. Huấn luyện mô hình tối ưu
Nếu bạn muốn huấn luyện lại mô hình tối ưu nhất và xuất ra file chạy cho Web App:
```bash
PYTHONPATH=. python training_scripts/train_optimal_model_to_production.py
```
*Quá trình này huấn luyện cực nhanh (chỉ mất chưa tới 20 giây trên CPU).*

### 3. Khởi chạy Giao diện Web
Để test thử khả năng phân loại bài báo của mô hình qua Web UI trực quan, hãy chạy:
```bash
python app.py
```
Sau đó mở trình duyệt và truy cập vào đường link: `http://127.0.0.1:5000`

---

## 📚 Báo cáo chuyên sâu
Toàn bộ tư duy thuật toán, cách xử lý lỗi tràn số toán học, công thức Adam Optimizer cũng như các thử nghiệm đều được ghi chép và phân tích cẩn thận tại file: `Bao_cao_Mo_hinh.md`. Mời bạn đọc để hiểu chi tiết vòng đời kiến trúc phần mềm này!
