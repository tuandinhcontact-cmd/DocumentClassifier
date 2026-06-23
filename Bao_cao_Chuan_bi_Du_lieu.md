# BÁO CÁO KỸ THUẬT: QUY TRÌNH CHUẨN BỊ VÀ TIỀN XỬ LÝ DỮ LIỆU TIN TỨC
*(Data Preparation & Preprocessing Pipeline)*

Báo cáo này trình bày chi tiết quy trình xử lý dữ liệu từ các nguồn thô ban đầu, các chiến lược khử nhiễu vật lý, phân rã nhãn chồng chéo ngữ nghĩa, tiền xử lý văn bản và cơ chế cắt tỉa để cân bằng phân phối dữ liệu trước khi đưa vào huấn luyện mô hình 3-Step Hybrid Cascade.

---

## 1. Khảo sát nguồn dữ liệu thô (Raw Data Profiling)

Hệ thống sử dụng dữ liệu từ ba nguồn độc lập với các đặc tính vật lý và mức độ chất lượng khác nhau:

| Nguồn dữ liệu | Định dạng thô | Quy mô mẫu | Số lượng nhãn gốc | Đặc điểm & Chất lượng |
| :--- | :---: | :---: | :---: | :--- |
| **HuffPost Dataset** | CSV (từ JSON gốc) | 182,081 | 42 nhãn | Tin tức từ báo HuffPost. Văn bản cực ngắn (Headline + Short Description). Dữ liệu rất sạch, ít lỗi chính tả, nhãn chính xác. |
| **BBC News Dataset** | CSV | 2,225 | 5 nhãn | Bộ dữ liệu benchmark NLP chuẩn của BBC. Bài viết có độ dài trung bình lớn, văn cảnh phong phú, nhãn rất chuẩn. |
| **news.csv** | CSV | 127,393 | 3 nhãn | Tin tức thời sự cào thô từ Internet. Chứa nhiều dòng trống, tin rác quá ngắn, trùng lặp và lỗi mâu thuẫn nhãn nghiêm trọng. |

---

## 2. Chiến lược Ánh xạ và Đồng bộ hóa nhãn (Label Harmonization)

Để tối ưu hóa khả năng phân loại, hệ thống thực hiện đồng bộ hóa toàn bộ không gian nhãn thô từ các nguồn dữ liệu khác nhau về **13 nhãn chuẩn** thống nhất:

```text
[42 Nhãn HuffPost] + [5 Nhãn BBC] + [3 Nhãn news.csv]
                        │
                        ▼ (Ánh xạ nhãn)
                  [13 Nhãn Chuẩn]
```

### Chi tiết ánh xạ các nhóm nhãn chính từ dữ liệu gốc HuffPost/BBC/news.csv:
* **Politics and society**: POLITICS, IMPACT, WORLD NEWS, THE WORLDPOST, WORLDPOST, CRIME (HuffPost); politics (BBC); politic (news.csv).
* **Tech & Science**: TECH, SCIENCE (HuffPost); tech (BBC); science, technology (news.csv).
* **Entertainment**: ENTERTAINMENT, COMEDY, MEDIA, WEIRD NEWS (HuffPost); entertainment (BBC).
* **Lifestyle**: TRAVEL, STYLE & BEAUTY, HOME & LIVING, STYLE, FIFTY, GOOD NEWS (HuffPost).
* **Business**: BUSINESS, MONEY (HuffPost); business (BBC).
* **Sports**: SPORTS (HuffPost); sport (BBC).
* **Health**: WELLNESS, HEALTHY LIVING (HuffPost).
* **Family**: PARENTING, PARENTS, WEDDINGS, DIVORCE (HuffPost).
* **Community**: QUEER VOICES, BLACK VOICES, LATINO VOICES, WOMEN, RELIGION (HuffPost).
* **Food & drinks**: FOOD & DRINK, TASTE (HuffPost).
* **Environment**: ENVIRONMENT, GREEN (HuffPost).
* **Arts & Culture**: ARTS & CULTURE, CULTURE & ARTS, ARTS (HuffPost).
* **Education**: EDUCATION, COLLEGE (HuffPost).

*Lưu ý: Nhãn `U.S. NEWS` gốc bị loại bỏ khỏi dữ liệu.*

---

## 3. Quy trình khử nhiễu sâu cho tệp news.csv (Physical & Label Noise Cleaning)

Do tệp `news.csv` cào thô chứa lượng nhiễu lớn ảnh hưởng trực tiếp đến chất lượng tối ưu hóa biên phân lớp, chúng tôi thiết lập một bộ lọc tiền xử lý riêng thông qua [clean_news_csv.py](file:///Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/scratch/clean_news_csv.py):

1. **Loại bỏ dòng trống**: Loại bỏ hoàn toàn các dòng không chứa nội dung văn bản (18 dòng).
2. **Loại bỏ tin tức quá ngắn**: Lọc bỏ các dòng có độ dài **dưới 8 từ** (5,097 dòng). Đây là các chuỗi ký tự rác hoặc thông tin phụ (ví dụ: *"Read more..."*, *"Update..."*) không mang giá trị từ khóa đặc trưng.
3. **Giải quyết mâu thuẫn nhãn (Label Conflict Resolution)**: Loại bỏ hoàn toàn các dòng có nội dung văn bản giống hệt nhau nhưng bị gán nhãn khác nhau (2,698 dòng bị xóa). Việc này ngăn chặn tình trạng thuật toán Logistic Regression và Naive Bayes bị hội tụ lệch hoặc không thể phân chia biên giới hạn đặc trưng do cùng một vector đầu vào có nhiều nhãn đối nghịch.
4. **Khử trùng lặp (Deduplication)**: Xóa bỏ các dòng trùng lặp nội dung văn bản có cùng nhãn (1,799 dòng).

> **Kết quả định lượng**: Quy mô tệp `news.csv` sau làm sạch giảm từ **127,393 dòng** xuống còn **117,781 dòng sạch**.

---

## 4. Pipeline Tiền xử lý văn bản chung (Text Preprocessing)

Sau khi đồng bộ hóa nhãn, toàn bộ văn bản từ cả 3 nguồn đều được đi qua một pipeline chuẩn hóa văn bản tự động trước khi vector hóa:

1. **Chuẩn hóa chữ thường (Lowercasing)**: Chuyển đổi toàn bộ văn bản về dạng viết thường để mô hình không phân biệt chữ viết hoa đầu câu hay viết hoa danh từ riêng.
2. **Lọc ký tự bằng Biểu thức chính quy (Regex Cleaning)**: Áp dụng biểu thức chính quy để loại bỏ hoàn toàn các dấu câu, ký tự đặc biệt, chữ số và liên kết URL. Chỉ giữ lại bảng chữ cái Latinh và các khoảng trắng phân cách:
   `[^a-zA-Záàảãạ...đ\s]`
3. **Loại bỏ từ dừng (Stopwords Removal)**: Tách văn bản thành các từ đơn và lọc bỏ danh sách từ dừng tiếng Anh chuẩn (`ENGLISH_STOP_WORDS`). Đây là những từ ngữ có tần suất xuất hiện cao nhưng mang ít giá trị ngữ nghĩa phân loại (ví dụ: *the, is, at, which, on...*).

---

## 5. Cơ chế Cắt tỉa và Cân bằng dữ liệu (Data Capping & Balancing)

Để giải quyết bài toán mất cân bằng lớp cực đoan (class imbalance) — nguyên nhân gây sụp đổ F1-score của các lớp thiểu số, hệ thống áp dụng cơ chế cắt tỉa 2 giai đoạn:

### Giai đoạn 1: Cắt tỉa chọn lọc trước khi gộp (Selective Capping)
Áp dụng giới hạn tối đa `JSON_MAX_SAMPLES = 3500` mẫu cho mỗi nhãn trong số 11 nhãn thiểu số của tập dữ liệu Huffpost. Riêng hai nhãn đa số `Tech & Science` và `Politics and society` trong tập HuffPost được giữ nguyên tỷ lệ tự nhiên ở giai đoạn này để thu thập tối đa sự phong phú từ vựng.

### Giai đoạn 2: Cắt tỉa sau khi gộp (Post-merge Capping)
Sau khi gộp chung cả 3 nguồn dữ liệu đã làm sạch lại với nhau, hệ thống tiến hành giới hạn số lượng mẫu tối đa của hai lớp đa số:
* `TECH_CAP = 40000` mẫu cho lớp `Tech & Science`.
* `POL_CAP = 40000` mẫu cho lớp `Politics and society`.

> [!TIP]
> Cơ chế này giúp thu hẹp khoảng cách quy mô lớp (từ chênh lệch gấp 40 lần xuống còn tối đa gấp 10 lần so với các lớp thiểu số), bảo toàn tốt ranh giới phân loại của các lớp nhỏ ở Bước 3 mà không làm mất đi các tin tức chất lượng cao từ các nguồn bổ sung.

---

## 6. Phân phối dữ liệu sạch cuối cùng (Final Dataset Distribution)

Tập dữ liệu gộp sạch cuối cùng được lưu trữ tại tệp [merged_cleaned_dataset.csv](file:///Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/merged_cleaned_dataset.csv) với tổng số lượng **117,983 dòng** và có phân bố nhãn chi tiết như sau:

| STT | Danh mục mục tiêu (Category) | Số lượng mẫu sạch | Tỷ lệ phần trăm (%) |
| :---: | :--- | :---: | :---: |
| 1 | **Politics and society** | 40,000 | 33.90% |
| 2 | **Tech & Science** | 40,000 | 33.90% |
| 3 | Sports | 4,011 | 3.40% |
| 4 | Business | 4,010 | 3.40% |
| 5 | Entertainment | 3,886 | 3.29% |
| 6 | Family | 3,500 | 2.97% |
| 7 | Lifestyle | 3,500 | 2.97% |
| 8 | Community | 3,500 | 2.97% |
| 9 | Food & drinks | 3,500 | 2.97% |
| 10 | Health | 3,500 | 2.97% |
| 11 | Environment | 3,488 | 2.96% |
| 12 | Arts & Culture | 3,265 | 2.77% |
| 13 | Education | 1,823 | 1.55% |
| **Tổng** | **13 danh mục** | **117,983** | **100%** |
