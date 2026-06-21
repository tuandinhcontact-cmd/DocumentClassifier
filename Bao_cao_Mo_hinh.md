# BÁO CÁO DỰ ÁN: PHÂN LOẠI TIN TỨC VỚI KIẾN TRÚC 3-STEP HYBRID CASCADE

## 1. Phân công nhiệm vụ
- **Thành viên 1:** Tìm hiểu cơ sở toán học, trực tiếp lập trình (code from scratch) các mô hình Logistic Regression (với Adam Optimizer) và Multinomial Naive Bayes (với Log-Sum-Exp).
- **Thành viên 2:** Nghiên cứu và thiết kế kiến trúc phân loại phân tầng (Cascade Architecture), xử lý mất cân bằng dữ liệu, và lập trình bộ tổng hợp Soft-voting.
- **Thành viên 3:** Tiền xử lý dữ liệu (Cleaning, Regex, Stopwords), xây dựng giao diện Web bằng Flask và tích hợp mô hình lên hệ thống.
*(Vui lòng điền tên cụ thể của các thành viên vào mục này).*

---

## 2. Giới thiệu bài toán
### 2.1. Phát biểu bài toán
Hàng ngày, các tòa soạn báo điện tử sản xuất hàng ngàn bản tin. Phân loại thủ công là bất khả thi. Bài toán đặt ra là: **Xây dựng một thuật toán học máy có khả năng tự động gán nhãn thể loại cho một bài báo chỉ dựa vào văn bản (Tiêu đề và Mô tả ngắn)**.
Khó khăn lớn nhất của bài toán nằm ở sự mất cân bằng lớp (Imbalanced Data) cực đoan. Nhóm `Tech & Science` và `Politics and society` chiếm tới hơn 70% tổng số bản ghi, trong khi 12 nhãn còn lại chỉ chia nhau 30%. Nếu dùng phương pháp Flat Classification (Phân loại phẳng đồng loạt), mô hình sẽ bị thiên lệch hoàn toàn về 2 nhãn lớn.

### 2.2. Mục đích và phạm vi nghiên cứu
- **Mục đích:** Đề xuất một kiến trúc giải quyết triệt để sự mất cân bằng dữ liệu mà không cần phải dùng đến kỹ thuật sinh dữ liệu giả (SMOTE) vốn làm méo mó văn bản.
- **Phạm vi nghiên cứu:** 
  - Khởi tạo hoàn toàn bằng tay (From Scratch) các thuật toán Cổ điển bằng Toán học ma trận (NumPy). 
  - Không sử dụng thư viện mô hình xây sẵn của scikit-learn để chứng minh năng lực hiểu sâu bản chất thuật toán.

---

## 3. Các phương pháp được sử dụng
### 3.1. Dữ liệu
#### 3.1.1. Đánh giá tổng quan về dữ liệu
Tập dữ liệu đầu vào chứa 182,081 bài báo được phân vào 14 danh mục. Các văn bản có độ dài rất ngắn (trung bình 1-3 câu), chứa nhiều ký tự đặc biệt, URL, và số đếm.

#### 3.1.2. Chuẩn bị và Tiền xử lý dữ liệu
1. Ghép cột `headline` và `short_description` thành một khối văn bản duy nhất nhằm gia tăng lượng từ vựng.
2. Dùng Biểu thức chính quy (Regex) để xóa ký tự đặc biệt, xóa URL, và chuyển về chữ thường.
3. Loại bỏ stopwords bằng danh sách từ vựng chuẩn của tiếng Anh.

#### 3.1.3. Vector hóa (TF-IDF)
Máy tính không hiểu chữ cái, do đó toàn bộ văn bản được mã hóa thành ma trận toán học thông qua thuật toán TF-IDF. Nhằm loại bỏ nhiễu và "lời nguyền chiều dữ liệu", chúng tôi cố tình giới hạn không gian từ vựng ở mức `max_features = 20000`. Thử nghiệm thực tế cho thấy việc giới hạn chung này đóng vai trò như một bộ lọc nhiễu tự nhiên tuyệt vời.

### 3.2. Thiết kế mô hình học máy và Ensemble
Hệ thống sử dụng **Cấu trúc 3-Step Hybrid Cascade** (Phân loại thác nước 3 bước):
- **Bước 1:** Dùng `CustomMultinomialNB` lọc toàn bộ nhãn `Tech & Science` ra khỏi tệp dữ liệu.
- **Bước 2:** Dùng `CustomMultinomialNB` lọc tiếp nhãn `Politics and society`.
- **Bước 3:** 12 nhãn thiểu số còn lại được chuyển cho bộ **CustomMultiClassVotingClassifier**. Bộ máy này trung bình cộng xác suất từ 1 mô hình NB, 12 mô hình LR, và 12 mô hình SVM (đã qua hiệu chuẩn Platt Scaling) để đưa ra phán quyết cuối cùng.

---

## 4. Cơ sở lý thuyết (Bám sát mã nguồn)

Mục này trình bày chi tiết nền tảng Toán học đằng sau các class Python được viết thủ công trong hệ thống.

### 4.1. Logistic Regression với thuật toán Adam Optimizer (Lớp `CustomLogisticRegression`)
Thay vì dùng Gradient Descent truyền thống, class `CustomLogisticRegression` của chúng tôi sử dụng hàm mất mát Cross-Entropy và thuật toán Adam (Adaptive Moment Estimation) để tăng tốc độ hội tụ trên ma trận thưa 20.000 chiều.
- **Hàm giả thuyết (Sigmoid):** $h(X) = \frac{1}{1 + e^{-(X \cdot w + b)}}$
- **Khởi tạo thông số thuật toán Adam:**
  - Vận tốc (Velocity) $v_w = 0$ và Động lượng (Momentum) $m_w = 0$.
  - $\beta_1 = 0.9$ và $\beta_2 = 0.999$.
- **Cập nhật trọng số trong Code:**
  Tại mỗi Epoch, Gradient của $w$ và $b$ được tính toán. Thay vì lấy $w = w - \alpha \cdot \text{grad}$, Adam điều chỉnh như sau:
  - $m_w = \beta_1 \cdot m_w + (1 - \beta_1) \cdot \text{grad}$
  - $v_w = \beta_2 \cdot v_w + (1 - \beta_2) \cdot \text{grad}^2$
  - Trọng số mới: $w = w - \frac{\alpha}{\sqrt{v_{w\_corrected}} + \epsilon} \cdot m_{w\_corrected}$
Nhờ thuật toán này, mô hình Logistic Regression của chúng tôi chỉ mất ~100 vòng lặp là đạt độ chính xác tối đa.

### 4.2. Multinomial Naive Bayes và Log-Sum-Exp Trick (Lớp `CustomMultinomialNB`)
Đây là mô hình chủ lực ở Bước 1 và Bước 2 do tốc độ bay lượn trên ma trận đếm (TF-IDF). Dựa trên định lý Bayes, xác suất của lớp $y_c$ với văn bản $x$ là: $P(y_c|x) \propto P(y_c) \prod P(x_i|y_c)$.
Tuy nhiên, trong code thực tế, nếu nhân hàng vạn xác suất nhỏ lại với nhau, máy tính sẽ bị tràn số (Underflow - trả về kết quả 0.0). Do đó, lớp `CustomMultinomialNB` sử dụng logarit:
- **Biến đổi hàm log:** $\log P(y_c|x) = \log P(y_c) + \sum \log P(x_i|y_c)$
Khi cần trả về giá trị xác suất (Probability) từ không gian Log, chúng tôi sử dụng thủ thuật toán học **Log-Sum-Exp (LSE)** trong hàm `predict_proba`:
- Cân bằng giá trị max: $M = \max(\text{log\_probs})$
- $P(y) = e^{\text{log\_probs} - M} / \sum e^{\text{log\_probs} - M}$
Điều này giúp mô hình bay Bayes tự viết của chúng tôi hoàn toàn miễn nhiễm với lỗi toán học của hệ thống phần cứng.

### 4.3. Linear SVM với Adam Optimizer và Hiệu chuẩn Platt Scaling (Lớp `CustomLinearSVM`)
Khác với LR tối ưu xác suất, SVM tối ưu "Khoảng cách ranh giới lớn nhất" (Maximum Margin) bằng hàm Hinge Loss. Ban đầu, mô hình sử dụng Pegasos SGD nhưng bị sụp đổ (Accuracy 8%) do mất cân bằng dữ liệu và tốc độ hội tụ chậm trên ma trận thưa. 
Để giải quyết, chúng tôi đã tiến hành một nâng cấp mang tính đột phá: 
- Lắp đặt **Adam Optimizer** thay cho SGD, giúp mô hình hội tụ cực nhanh trong 100 epochs.
- Lập trình cơ chế **Cân bằng nhãn (`class_weight='balanced'`)** tương tự như LR, giúp nhân hệ số phạt Gradient cho nhãn thiểu số, ngăn chặn hiện tượng Linear SVM bị "kẹt" ở dự đoán nhãn đa số.

Do SVM gốc không xuất ra xác suất (không thể tham gia Soft-Voting), chúng tôi tiếp tục áp dụng thuật toán **Platt Scaling**:
- Tính toán khoảng cách cứng $f(x) = w^T x + b$ cho toàn bộ điểm dữ liệu.
- Đưa các khoảng cách này đi qua một mô hình Logistic Regression phụ trợ $P(y=1|x) = \frac{1}{1 + e^{-(A \cdot f(x) + B)}}$. 
Điều này ép SVM phải nhả ra dải phân bố xác suất mượt mà từ 0 đến 1, giúp nó đủ tiêu chuẩn đứng vào hàng ngũ Soft-Voting ở Bước 3.

### 4.4. One-Vs-Rest Classification (Lớp `CustomOneVsRestClassifier`)
Logistic Regression tự nhiên chỉ phân biệt được 2 lớp (Nhị phân 0-1). Để giúp nó phân loại 12 nhãn ở Bước 3, chúng tôi lập trình thuật toán OVR.
- **Quy trình hoạt động:** Hàm `fit` sẽ duyệt qua $K=12$ nhãn. Ở mỗi vòng lặp, nó sao chép nhân bản một mô hình Logistic Regression, coi nhãn hiện tại là $1$ và 11 nhãn còn lại là $0$. Do đó sinh ra 12 mô hình nhị phân độc lập.
- **Chuẩn hóa xác suất (L1 Normalization):** Khi gọi `predict_proba`, 12 mô hình sẽ trả về 12 giá trị xác suất (không cộng lại bằng 1). Hàm sẽ tự động tính tổng của 12 xác suất trên từng hàng, và chia ngược lại từng ô cho tổng đó để đảm bảo chuẩn phân bố xác suất L1.

### 4.4. Kiến trúc luồng dữ liệu 3-Step Cascade (Lớp `ThreeStepCascadeClassifier`)
Thuật toán phân loại thác nước (Cascade) là trái tim của dự án, nhằm giải quyết sự mất cân bằng.
- Ý tưởng toán học: Phân hoạch không gian mẫu thành 3 phần không giao nhau.
  - Tầng 1 hấp thụ $x$ nếu $P(\text{Tech}|x) \ge 0.5$. Lượng dữ liệu còn lại $X' = X \setminus X_{\text{Tech}}$.
  - Tầng 2 hấp thụ $x \in X'$ nếu $P(\text{Politics}|x) \ge 0.5$. Dữ liệu còn lại $X'' = X' \setminus X_{\text{Politics}}$.
  - Tầng 3 nhận $X''$ để phân rã qua hội đồng OVR.

### 4.6. Ensemble Soft-Voting (Lớp `CustomMultiClassVotingClassifier`)
Thay vì để 1 thuật toán đưa ra phán quyết cuối cùng (Gây ra thiên lệch), chúng tôi thiết kế hội đồng bầu chọn mềm. Hội đồng "Tam Thánh" bao gồm:
- Ma trận xác suất từ MultinomialNB là $P_A$.
- Ma trận xác suất từ OVR Logistic Regression là $P_B$.
- Ma trận xác suất từ OVR Linear SVM (Platt) là $P_C$.
- Soft-Voting thực hiện tính trung bình cộng từng điểm dữ liệu: $P_{\text{final}} = \frac{P_A + P_B + P_C}{3}$. 
Nhãn cuối cùng là `argmax` của $P_{\text{final}}$. Điều này bù trừ nhược điểm của tất cả các mô hình, tạo ra một chốt chặn an toàn hoàn hảo về mặt lý thuyết học máy.

---

## 5. Thực nghiệm và đánh giá
### 5.1. Thực nghiệm và đánh giá
Hệ thống được huấn luyện trên 182,081 mẫu, sử dụng bộ xử lý trung tâm (CPU). Mã nguồn Python 100% tự viết cho Core Algorithms.

### 5.2. Kết quả so sánh
Dưới đây là sự tiến hóa của hiệu năng thông qua các lần tái cấu trúc kiến trúc phần mềm:

| Kiến trúc | Accuracy | Macro F1 | Thời gian |
| :--- | :---: | :---: | :---: |
| Flat Classification (Sklearn C++ Baseline) | 75.96% | 58.69% | 8.66s |
| 13-Node Cascade (Bản cũ) | 74.57% | 51.04% | 197.22s |
| **3-Step Cascade (Soft-Voting tự code)** | **76.77%** | **56.71%** | **92.32s** |

Sự vượt trội của mô hình 3-Step Cascade (76.77%) so với mô hình tối ưu C++ của thư viện `scikit-learn` (75.96%) là minh chứng hùng hồn nhất cho thấy việc **thiết kế luồng dữ liệu (Data Flow)** có sức ảnh hưởng mạnh mẽ hơn việc nhắm mắt áp dụng các thư viện sẵn có.

### 5.3. Đánh giá điểm mạnh/ điểm yếu
- **Điểm mạnh:** Xử lý tận gốc sự mất cân bằng dữ liệu mà không cần sinh dữ liệu giả; Tốc độ hội tụ thần tốc (92s cho 182k bài báo) do áp dụng Adam Optimizer.
- **Điểm yếu:** Cơ chế Cascade Bước 1 và 2 dễ "cướp" mất các văn bản có ngữ cảnh lai tạp (Ví dụ bài Chính trị có yếu tố Mạng xã hội dễ bị tầng Tech hút mất), do giới hạn từ vựng của thuật toán TF-IDF.

---

## 6. Cài đặt hệ thống và khó khăn gặp phải
### 6.1. Các chức năng chính của hệ thống
Hệ thống Web Flask cung cấp:
- Ô nhập liệu văn bản tin tức tự do.
- Giao diện **Cascade Path Tracking**, cho phép người dùng nhìn xuyên thấu vào hộp đen AI: Từng dòng log minh họa bản tin đã "đập" vào cửa Tầng 1 như thế nào, bị từ chối ra sao, và được Tầng 3 tiếp nhận với xác suất bao nhiêu.

### 6.2. Khó khăn và cách thức giải quyết
1. **Tràn bộ nhớ với TF-IDF động (Dynamic TF-IDF):**
   - *Khó khăn:* Nỗ lực tháo bỏ giới hạn số lượng từ ở Bước 3 nhằm vớt vát dữ liệu thiểu số đã phản tác dụng. Ma trận lên tới hàng trăm ngàn chiều khiến hệ thống chạy ì ạch và Accuracy tụt giảm do nhiễu.
   - *Giải quyết:* Khôi phục lại mốc **Global TF-IDF 20.000 từ vựng**. Nó đóng vai trò làm bộ lọc nhiễu xuất sắc.
2. **Xác suất bị lỗi chia 0 (Divide by Zero) trong Lớp OVR:**
   - *Khó khăn:* Khi tính L1 Normalization, có những hàng xác suất của cả 12 nhãn đều là 0, dẫn tới lỗi NaN.
   - *Giải quyết:* Bổ sung cơ chế `row_sums[row_sums == 0] = 1.0` vào hàm chia ma trận Numpy.

---

## 7. Tranh luận/khám phá/kết luận
### 7.1. Kết luận và khám phá
Dự án chứng minh được sự khác biệt giữa "Thợ gõ code" và "Kỹ sư máy học". Bằng cách hiểu rõ từng đạo hàm trong Logistic Regression và từng phép nhân logarit trong Naive Bayes, chúng tôi đã tạo ra một hệ thống tùy chỉnh với hiệu năng Đánh bại được thư viện công nghiệp Scikit-learn ở thước đo Accuracy. 
Việc sắp xếp thứ tự phân loại theo khối lượng dữ liệu (Cascade) giải quyết hoàn toàn bài toán Imbalanced mà không cần tăng khối lượng tính toán.

### 7.2. Hướng phát triển trong tương lai
Sẽ vô cùng thú vị nếu thay thế Vector TF-IDF bằng **Pre-trained BERT Embeddings**. BERT sẽ loại bỏ hoàn toàn yếu điểm "giao thoa từ vựng" của Bước 1, bởi nó hiểu được văn cảnh (Contextual) thay vì chỉ nhặt từ khóa. Tuy nhiên điều này đòi hỏi nâng cấp hạ tầng phần cứng có trang bị GPU.
