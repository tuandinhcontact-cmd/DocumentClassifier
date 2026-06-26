# BÁO CÁO KỸ THUẬT: QUY TRÌNH CHUẨN BỊ VÀ TIỀN XỬ LÝ DỮ LIỆU TIN TỨC
*(Data Preparation & Preprocessing Pipeline)*

Báo cáo này trình bày chi tiết quy trình xử lý dữ liệu từ các nguồn thô ban đầu, chiến lược ánh xạ đồng bộ hóa nhãn, tiền xử lý làm sạch văn bản và cơ chế cắt tỉa để cân bằng phân phối dữ liệu trước khi đưa vào huấn luyện mô hình Flat Soft Voting Ensemble.

---

## 1. Khảo sát nguồn dữ liệu thô (Raw Data Profiling)

Hệ thống sử dụng dữ liệu từ **3 nhóm nguồn** độc lập được lưu trữ trong thư mục `data/` với các đặc tính vật lý và chất lượng khác nhau:

| Nhóm nguồn dữ liệu | Định dạng thô | Quy mô mẫu | Số lượng nhãn gốc | Đặc điểm & Chất lượng |
| :--- | :---: | :---: | :---: | :--- |
| **1. HuffPost News** | JSON / CSV | 188,437 | 40+ nhãn | Tin tức từ báo HuffPost. Văn bản cực ngắn (Headline + Short Description). Dữ liệu rất sạch, nhãn chính xác. |
| **2. BBC News** | CSV | 2,225 | 5 nhãn | Bộ dữ liệu benchmark NLP chuẩn của BBC. Bài viết có độ dài trung bình lớn (báo dài), văn cảnh phong phú, nhãn rất chuẩn. |
| **3. Tập bổ sung** | CSV (4 tệp) | 8,000 | 4 nhãn | 4 tệp dữ liệu sạch chuyên biệt (`education_data.csv`, `technology_data.csv`, `sports_data.csv`, `business_data.csv`). Văn bản ngắn, cân bằng 2,000 mẫu/tệp. |

---

### Khảo sát chi tiết từng nhóm nguồn dữ liệu thô:

#### 📰 Nhóm 1: HuffPost News Dataset v3
* **Nguồn gốc**: Cào từ trang tin tức HuffPost từ năm 2012 đến 2022 (gồm tiêu đề `headline` và mô tả ngắn `short_description`).
* **Đặc tính văn bản**: Văn bản dạng cực ngắn, trung bình ~31.4 từ/bản ghi.
* **Đặc điểm phân bố**: Mất cân bằng cực kỳ nghiêm trọng (Politics chiếm tới 46.5k mẫu, trong khi Education chỉ có 1.8k mẫu).
* **Vai trò trong dự án (Role in Project)**:
  * **Là khung xương của hệ thống**: Đây là nguồn dữ liệu nền tảng và duy nhất cung cấp tin tức cho các lớp không có trong các tập dữ liệu bổ sung (như `Lifestyle`, `Arts & Culture`, `Food & drinks`, `Family`, `Community`, `Environment`, `Health`, v.v.).
  * **Đồng bộ hóa không gian nhãn**: Nhờ ánh xạ 42 nhãn gốc HuffPost về 13 nhãn chuẩn, hệ thống mới có khả năng phân loại đa lớp đồng nhất trên toàn bộ các nguồn dữ liệu tin tức.

#### 📡 Nhóm 2: BBC News Dataset
* **Nguồn gốc**: Bộ dữ liệu benchmark NLP chuẩn gồm các bài báo chất lượng cao từ trang tin BBC.
* **Đặc tính văn bản**: Văn bản dài, đầy đủ nội dung bài viết (`news`), trung bình chứa tới 384 từ (gấp 12 lần HuffPost).
* **Đặc điểm phân bố**: Cân bằng tương đối tốt trên 5 thể loại chính (Sport, Business, Politics, Tech, Entertainment).

#### 📚 Nhóm 3: Các tập dữ liệu bổ sung chuyên biệt (Sports / Business / Tech / Education)
* **Nguồn gốc (Origin)**: 
  * Trích xuất từ tập dữ liệu công cộng **"News Articles Classification Dataset for NLP & ML"** trên Kaggle đóng góp bởi tác giả **Banuprakash Vellingiri**.
  * Dữ liệu được cào tự động (web scraping) trực tiếp từ trang báo điện tử nổi tiếng của Ấn Độ là ***"The Indian Express"*** (https://indianexpress.com/) dưới giấy phép mã nguồn mở **Apache 2.0**.
* **Định dạng ban đầu (Initial Format)**: Gồm 4 tệp tin CSV riêng biệt (`sports_data.csv`, `business_data.csv`, `technology_data.csv`, `education_data.csv`), mỗi tệp chứa đúng **2,000 bản ghi** độc nhất (tổng cộng **8,000 mẫu**).
* **Cấu trúc các trường (Field Structure)**: Gồm 5 cột dữ liệu chính:
  1. `Headlines`: Tiêu đề bài viết.
  2. `Description`: Tóm tắt nội dung bài viết.
  3. `Content`: Toàn bộ nội dung bài báo chi tiết (Full text).
  4. `URL`: Link nguồn bài viết gốc trên trang The Indian Express.
  5. `Category`: Nhãn thể loại tương ứng (`sports`, `business`, `technology`, `education`).
* **Đặc điểm văn bản (Text Characteristics)**:
  * **Ngôn ngữ**: 100% Tiếng Anh chuẩn báo chí chính thống Ấn Độ.
  * **Ghép nối**: Dự án ghép `Headlines` + `Description` thành chuỗi văn bản đầu vào để đồng dạng với tin tức dạng ngắn của HuffPost. Độ dài trung bình sau ghép là **39.4 từ/bản ghi** (tối đa **84 từ**).
  * **Từ vựng**: Mang đậm các thuật ngữ chuyên ngành (như cricket, tài chính ngân hàng Ấn Độ, công nghệ thông tin và chính sách giáo dục).
* **Vai trò trong dự án (Role in Project)**:
  * **Tăng cường mẫu (Data Augmentation)** cho các lớp thiểu số (ví dụ: lớp Education tăng vọt thêm 2,000 mẫu, tăng **+110%**).
  * **Cân bằng lớp (Class Balancing)** và cải thiện đáng kể Macro F1-score tổng thể (tăng vọt từ ~64% lên vượt mốc **68%**).
  * **Tăng tính tổng quát hóa (Generalization)** và giảm thiểu hiện tượng quá khớp (overfitting).

---

## 2. Chiến lược Ánh xạ và Đồng bộ hóa nhãn (Label Harmonization)

Để tối ưu hóa khả năng phân loại đa lớp, hệ thống thực hiện đồng bộ hóa toàn bộ không gian nhãn thô từ các nguồn dữ liệu khác nhau về **13 nhãn chuẩn** thống nhất:

```text
[40+ Nhãn HuffPost] + [5 Nhãn BBC] + [4 Nhãn Tập bổ sung]
                        │
                        ▼ (Ánh xạ đồng bộ)
                  [13 Nhãn Chuẩn]
```

### Chi tiết ánh xạ các nhóm nhãn chính từ dữ liệu gốc về 13 nhãn chuẩn:
* **Politics and society**: POLITICS, IMPACT, WORLD NEWS, THE WORLDPOST, WORLDPOST, CRIME (HuffPost); politics (BBC).
* **Tech & Science**: TECH, SCIENCE (HuffPost); tech (BBC); Technology (từ `technology_data.csv`).
* **Entertainment**: ENTERTAINMENT, COMEDY, MEDIA, WEIRD NEWS (HuffPost); entertainment (BBC).
* **Lifestyle**: TRAVEL, STYLE & BEAUTY, HOME & LIVING, STYLE, FIFTY, GOOD NEWS (HuffPost).
* **Business**: BUSINESS, MONEY (HuffPost); business (BBC); Business (từ `business_data.csv`).
* **Sports**: SPORTS (HuffPost); sport (BBC); Sports (từ `sports_data.csv`).
* **Health**: WELLNESS, HEALTHY LIVING (HuffPost).
* **Family**: PARENTING, PARENTS, WEDDINGS, DIVORCE (HuffPost).
* **Community**: QUEER VOICES, BLACK VOICES, LATINO VOICES, WOMEN, RELIGION (HuffPost).
* **Food & drinks**: FOOD & DRINK, TASTE (HuffPost).
* **Environment**: ENVIRONMENT, GREEN (HuffPost).
* **Arts & Culture**: ARTS & CULTURE, CULTURE & ARTS, ARTS (HuffPost).
* **Education**: EDUCATION, COLLEGE (HuffPost); Education (từ `education_data.csv`).


---

## 3. Phân tích sâu quy trình Chuẩn bị và Tích hợp dữ liệu (In-depth Data Preparation & Integration)

Trước khi đi vào quy trình tiền xử lý làm sạch ký tự văn bản, hệ thống thực thi một chuỗi các bước chuẩn bị dữ liệu cấu trúc (Data Preparation) ở mức DataFrame nhằm đồng bộ hóa cấu trúc, tích hợp nguồn và giải quyết bài toán mất cân bằng phân phối lớp:

### 3.1. Đồng nhất hóa cấu trúc đầu vào (Structural Alignment)
Do dữ liệu đầu vào đến từ 3 nguồn độc lập có cấu trúc cột hoàn toàn khác biệt, hệ thống thực thi bước đồng bộ cấu trúc trường văn bản thành một cột đại diện duy nhất `text_raw`:
* **Đối với HuffPost**: Kết hợp tiêu đề và mô tả ngắn: $\text{text\_raw} = \text{headline} + \text{" "} + \text{short\_description}$. Chúng tôi sử dụng phương thức `.fillna('')` trước khi ghép nối để ngăn ngừa hiện tượng lan truyền giá trị trống (`NaN propagation`). Nếu không xử lý, sự xuất hiện của bất kỳ giá trị `NaN` nào ở một trong hai trường sẽ biến toàn bộ chuỗi kết quả ghép thành `NaN`.
* **Đối với BBC News**: Sử dụng trực tiếp trường `news` làm đại diện cho `text_raw` (đóng vai trò là tin tức dạng dài).
* **Đối với 4 tập dữ liệu bổ sung (Kaggle)**: Đồng dạng hóa bằng cách ghép nối tiêu đề và tóm tắt: $\text{text\_raw} = \text{Headlines} + \text{" "} + \text{Description}$ để tạo ra tin tức dạng ngắn tương thích với HuffPost.

### 3.2. Ánh xạ và Đồng bộ hóa nhãn cấp DataFrame (DataFrame-level Label Harmonization)
Trước khi thực hiện gộp (merge) các nguồn dữ liệu, hệ thống bắt buộc phải đồng bộ hóa nhãn của từng nguồn về cùng một không gian nhãn đích (13 nhãn chuẩn). Việc này được thực hiện ở mức DataFrame:
* **Đối với HuffPost**: Sử dụng phương thức `.map(category_mapping)` trong Pandas để chuyển đổi 42 nhãn ban đầu (ví dụ: `WELLNESS`, `HEALTHY LIVING` về nhãn chuẩn `Health`; `TECH`, `SCIENCE` về `Tech & Science`). Các bản ghi có nhãn không nằm trong danh mục quan tâm hoặc có giá trị nhãn trống sẽ bị lọc bỏ bằng phương thức `.dropna(subset=['category'])`.
* **Đối với BBC News**: Chuyển đổi cột loại tin tức `type` sang nhãn chuẩn tương ứng thông qua `bbc_mapping` (ví dụ: `sport` ➔ `Sports`, `business` ➔ `Business`, `politics` ➔ `Politics and society`).
* **Đối với 4 tập bổ sung**: Gán cứng nhãn trực tiếp cho từng tệp tin dựa trên bản chất dữ liệu đầu vào:
  * `education_data.csv` gán nhãn `Education`
  * `technology_data.csv` gán nhãn `Tech & Science`
  * `sports_data.csv` gán nhãn `Sports`
  * `business_data.csv` gán nhãn `Business`

### 3.3. Tích hợp và làm sạch dữ liệu đa nguồn (Multi-source Merging & Safety Filtering)
* **Gộp dữ liệu**: Sử dụng thư viện Pandas với hàm `pd.concat([df_huff, df_bbc, df_edu, df_tech, df_sports, df_business], ignore_index=True)` để hợp nhất 6 luồng dữ liệu thành một DataFrame duy nhất với chỉ mục (index) liên tục.
* **Lọc bỏ giá trị thiếu (Null Filtering)**: Loại bỏ triệt để các bản ghi bị thiếu thông tin ở trường văn bản hoặc trường nhãn bằng cách chạy hàm `.dropna(subset=['text_raw', 'category'])`.
* **Loại bỏ dòng trống (Empty Row Elimination)**: Một số trang tin cào bị lỗi chỉ thu về khoảng trắng. Hệ thống loại trừ các bản ghi này bằng bộ lọc logic:
  `merged_raw[merged_raw['text_raw'].str.strip() != '']`

### 3.4. Cơ chế Cắt tỉa kiểm soát mất cân bằng (Selective Capping Strategy)
* **Đặt vấn đề**: Nhãn đa số lớn nhất (Politics) có 46,979 mẫu trong khi nhãn thiểu số lớn nhất (Arts & Culture) chỉ có 3,265 mẫu (lệch gấp **14.4 lần**). Nếu huấn luyện trực tiếp, mô hình sẽ tối ưu hóa hàm lỗi bằng cách dự đoán nhãn đa số cho hầu hết các mẫu tin, dẫn đến sự sụp đổ Recall của các lớp nhỏ.
* **Lý do chọn ngưỡng 20,000**:
  * Nếu chọn ngưỡng quá nhỏ (ví dụ 5,000): Sẽ gây lãng phí dữ liệu và làm mất đi lượng từ vựng tin tức cực kỳ phong phú của tập HuffPost.
  * Nếu chọn ngưỡng quá lớn (ví dụ 40,000): Không giải quyết được mất cân bằng, tỷ lệ chênh lệch vẫn vượt quá 10 lần.
  * Ngưỡng **20,000** mẫu/nhãn là "điểm ngọt" (sweet spot) tối ưu qua thực nghiệm, giảm độ chênh lệch tối đa xuống còn **6.1 lần** nhưng vẫn giữ lại tổng lượng dữ liệu lớn (156k mẫu) giúp SVM và Logistic Regression hội tụ tốt.
* **Tại sao thực hiện capping trước khi chia Train/Test (Pre-split Capping)**:
  * Việc áp dụng capping trên tập dữ liệu tổng giúp chúng ta chia tập Train/Test theo tỷ lệ 80/20 phân tầng chính xác. Đồng thời, nó phản ánh đúng hiệu năng thực tế của mô hình biểu quyết trên một tập kiểm thử đại diện ổn định, thay vì chỉ capping trên tập Train khiến tập Test bị biến động.

---


## 4. Quy trình Tiền xử lý dữ liệu và Phân chia tập dữ liệu (Data Preprocessing & Dataset Splitting Pipeline)

Sau khi dữ liệu cấu trúc đã được tích hợp và cân bằng quy mô qua cơ chế Capping, hệ thống chuyển sang giai đoạn tiền xử lý dữ liệu. Quy trình này hoạt động như một pipeline tuần tự, chuyển đổi văn bản thô thành biểu diễn ma trận đặc trưng số học. 

Trong thiết kế hệ thống học máy thực tế, **phần chia tách tập Train/Test được xếp vào quy trình Tiền xử lý dữ liệu** bởi nó đóng vai trò là "chốt chặn phân định trạng thái" (stateful transition point). Nó chia quy trình tiền xử lý thành các bước độc lập với ngữ cảnh (Stateless) và các bước phụ thuộc vào ngữ cảnh toàn tập dữ liệu (Stateful), từ đó ngăn ngừa triệt để hiện tượng rò rỉ dữ liệu (Data Leakage).

```text
               [Dữ liệu sau Capping]
                         │
                         ▼
        [1. Stateless Cleaning (Làm sạch)]
        (Lowercasing, Regex, Stopwords)
                         │
                         ▼
       =====================================
       [2. Stratified Train/Test Split 80/20]  <--- CHỐT CHẶN NGĂN CHẶN LEAKAGE
       =====================================
            │                         │
            ▼ (Huấn luyện)            ▼ (Kiểm thử)
      [Tập Train Raw]           [Tập Test Raw]
            │                         │
            ▼                         │
     [Tập Train Vectorized]           │ (Ánh xạ 1 chiều)
     (fit_transform TF-IDF) ──────────┼───────► [Tập Test Vectorized]
                                      │         (transform TF-IDF)
                                      ▼
```

---

### 4.1. Giai đoạn 1: Làm sạch văn bản thô độc lập (Stateless Text Cleaning)
Các bước này được áp dụng độc lập trên từng tài liệu (document-wise), không phụ thuộc vào từ vựng hay tần suất của các tài liệu khác trong toàn bộ tập dữ liệu, nên hoàn toàn an toàn khi chạy trước bước phân tách Train/Test.

#### 1. Ghép cột văn bản (Text Concatenation)
* **Cách thực hiện**: 
  * Đối với tin tức dạng ngắn (HuffPost, 4 tệp bổ sung): Kết hợp tiêu đề và mô tả ngắn: $\text{text\_raw} = \text{headline} + \text{" "} + \text{description}$.
  * Đối với tin tức dạng dài (BBC News): Sử dụng trực tiếp nội dung chi tiết bài viết.
* **Lý do kỹ thuật**: Tiêu đề cung cấp từ khóa mang tính chủ đề cực kỳ mạnh (ví dụ: *"Stock market crash"*), trong khi mô tả ngắn cung cấp thêm các thực thể và ngữ cảnh bổ trợ (ví dụ: *"Wall Street"*). Việc ghép nối giúp mở rộng độ dài trung bình của tin ngắn từ ~30 từ lên ~39.4 từ, tạo ra mật độ đặc trưng dày hơn cho các bộ phân loại tuyến tính học từ khóa hiệu quả hơn.

#### 2. Chuẩn hóa chữ thường (Lowercasing)
* **Cách thực hiện**: Áp dụng phương thức `.lower()` trên toàn bộ chuỗi văn bản.
* **Lý do kỹ thuật**: Tiếng Anh viết hoa các từ đầu câu và danh từ riêng (ví dụ: *"Apple"* ở đầu câu hay *"Apple Inc."*). Nếu không chuẩn hóa, thuật toán trích xuất đặc trưng sẽ coi *"Apple"*, *"apple"*, và *"APPLE"* là ba token độc lập, dẫn đến sự bùng nổ không gian từ vựng (Vocabulary Explosion) vô nghĩa, làm loãng ma trận đặc trưng và tăng nguy cơ quá khớp (overfitting).

#### 3. Lọc ký tự nhiễu bằng Biểu thức chính quy (Regex Character Cleaning)
* **Cách thực hiện**: Sử dụng biểu thức chính quy `re.sub(r'[^a-zA-Z\s]', ' ', text)`.
* **Lý do kỹ thuật**: Dấu câu (dấu chấm, phẩy, hỏi chấm) và chữ số thường là nhiễu đối với các mô hình phân loại văn bản dạng túi từ (Bag-of-Words) hay TF-IDF. Nếu giữ lại, chúng sẽ tạo ra các từ khóa lỗi như *"politics,"* (dính dấu phẩy) khác với *"politics"*.
* **An toàn ranh giới từ (Word Boundary Safety)**: 
  > [!IMPORTANT]
  > Việc thay thế các ký tự không hợp lệ bằng **khoảng trắng `' '`** thay vì chuỗi rỗng `''` là cực kỳ quan trọng. 
  > * Nếu thay bằng chuỗi rỗng: Từ ghép ngăn cách bởi dấu gạch chéo hoặc gạch ngang như *"state/nation"* hay *"COVID-19"* sẽ bị dính liền thành *"statenation"* (từ vô nghĩa) và *"COVID19"*.
  > * Thay bằng khoảng trắng giúp bảo toàn ranh giới từ: *"state/nation"* ➔ *"state nation"*, *"COVID-19"* ➔ *"COVID 19"*, giữ nguyên ngữ nghĩa cốt lõi của hai từ đơn.

#### 4. Tách từ và Loại bỏ từ dừng (Tokenization & Stopwords Removal)
* **Cách thực hiện**: Tách chuỗi văn bản sạch thành các token riêng lẻ qua phương thức `.split()`, sau đó đối chiếu với danh sách từ dừng tiếng Anh chuẩn (`sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`) và loại bỏ chúng.
* **Lý do kỹ thuật**: Từ dừng (như *the, and, of, in, to, is...*) xuất hiện với tần suất cực kỳ cao trong mọi văn bản nhưng lại không mang thông tin phân loại chủ đề (chúng xuất hiện ở cả tin Thể thao lẫn tin Chính trị). Nếu giữ lại, chúng sẽ áp đảo tần suất xuất hiện (Term Frequency) của các từ khóa mang thông tin thực sự (như *election, court, football, software*). Mặc dù IDF có thể giảm trọng số của chúng, việc chủ động loại bỏ từ đầu giúp giảm đáng kể kích thước chiều của ma trận đặc trưng, tiết kiệm bộ nhớ RAM và tăng tốc độ tính toán cho các mô hình SVM và Logistic Regression.

---

### 4.2. Giai đoạn 2: Phân chia tập dữ liệu huấn luyện/kiểm thử phân tầng (Stratified Train/Test Split)
Đây là bước bản lề, chuyển đổi từ tiền xử lý stateless sang stateful.

* **Cách thực hiện**: Sử dụng thuật toán phân chia phân tầng với tỷ lệ 80% cho tập huấn luyện (Train - 125,412 mẫu) và 20% cho tập kiểm thử (Test - 31,353 mẫu) cùng hạt giống ngẫu nhiên cố định:
  `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
* **Toán học của phân chia phân tầng (Stratified Split)**:
  * Mặc dù đã áp dụng capping trần 20,000 mẫu/nhãn, phân phối nhãn của chúng ta vẫn mất cân bằng tự nhiên do các nhãn thiểu số (như Arts & Culture chỉ có 3,265 mẫu, Education chỉ có 3,823 mẫu, lệch khoảng **6.1 lần** so với nhãn đa số 20,000 mẫu).
  * Nếu chia tách ngẫu nhiên thông thường (Random Split), có thể xảy ra sai số chọn mẫu (sampling bias). Ví dụ, tập Test có thể ngẫu nhiên nhận quá nhiều hoặc quá ít mẫu của nhãn thiểu số Arts & Culture, dẫn đến việc tính toán chỉ số F1-score và Accuracy trên tập Test bị biến động lớn và thiếu tin cậy.
  * Việc thiết lập `stratify=y` đảm bảo tỷ lệ của toàn bộ 13 lớp nhãn trong tập huấn luyện (Train) và tập kiểm thử (Test) được bảo toàn chính xác tuyệt đối theo tỷ lệ gốc (ví dụ nhãn Politics luôn chiếm đúng 12.76% ở cả tập Train và Test, nhãn Arts & Culture luôn chiếm đúng 2.08% ở cả hai tập).
* **Bản chất ngăn chặn Rò rỉ dữ liệu (Data Leakage Prevention)**:
  > [!WARNING]
  > **Tại sao chia tách tập Train/Test phải được thực hiện NGAY TẠI ĐÂY, trước bước TF-IDF Vectorization?**
  > * Bước trích xuất đặc trưng TF-IDF là một bước tiền xử lý **phụ thuộc trạng thái (Stateful Preprocessing)**. Nó cần duyệt qua toàn bộ kho văn bản (corpus) để xây dựng từ điển từ vựng (vocabulary) và tính toán trọng số nghịch đảo tần suất tài liệu (IDF) cho từng từ.
  > * Nếu ta thực hiện `fit_transform` TF-IDF trên toàn bộ dữ liệu trước khi split, bộ trích xuất sẽ học được phân phối từ vựng và tần suất IDF của cả tập Test. Trọng số đặc trưng của tập Train lúc này đã bị "nhiễm độc" bởi thông tin từ tập Test.
  > * Hiện tượng này gọi là **Rò rỉ dữ liệu (Data Leakage)**. Khi xảy ra rò rỉ, mô hình sẽ đạt hiệu năng đánh giá (Accuracy, F1-score) cao một cách giả tạo trong quá trình kiểm thử, nhưng sẽ suy giảm nghiêm trọng (overfitting nặng) khi đối mặt với dữ liệu thực tế ngoài production.
  > * Do đó, bắt buộc phải split dữ liệu trước. Chỉ dùng tập Train để học từ điển và trọng số IDF (`fit_transform`), sau đó áp dụng một chiều (`transform`) lên tập Test để biến tập Test thành dữ liệu hoàn toàn mới và độc lập.

---

### 4.3. Giai đoạn 3: Trích xuất đặc trưng văn bản (Stateful Feature Extraction & TF-IDF Vectorization)
Bước cuối cùng trong pipeline tiền xử lý là chuyển đổi văn bản đã làm sạch thành các vector số học đại diện.

* **Cách thực hiện**:
  * Trên tập Train: `X_train_vec = vectorizer.fit_transform(X_train_raw)`
  * Trên tập Test: `X_test_vec = vectorizer.transform(X_test_raw)`
* **Chi tiết cấu hình tham số tối ưu hóa**:
  Bộ trích xuất đặc trưng sử dụng các tham số tối ưu đã được tinh chỉnh qua thực nghiệm:
  1. `ngram_range=(1, 3)`: Trích xuất các từ đơn (unigram), từ ghép đôi (bigram) và cụm ba từ (trigram). Việc trích xuất n-gram giúp mô hình học được các cụm từ mang ý nghĩa đặc trưng như *"white house"* (Chính trị), *"artificial intelligence"* (Công nghệ), *"world cup"* (Thể thao) thay vì chỉ nhìn nhận các từ đơn lẻ *"white"*, *"house"*, *"artificial"*, *"intelligence"*.
  2. `max_features=50000`: Giới hạn kích thước từ điển ở mức 50,000 đặc trưng quan trọng nhất có tần suất cao. Điều này giúp ngăn ngừa hiện tượng bùng nổ chiều đặc trưng (curse of dimensionality), loại bỏ các từ cực kỳ hiếm (ví dụ các từ gõ sai chính tả chỉ xuất hiện 1 lần) và kiểm soát bộ nhớ RAM khi huấn luyện SVM.
  3. `sublinear_tf=True`: Áp dụng công thức quy đổi tần suất từ (Term Frequency) theo thang đo logarithmic:
     $$\text{TF}_{\text{sublinear}} = 1 + \log(\text{TF})$$
     *Lý do toán học*: Trong một văn bản, nếu từ khóa *"election"* xuất hiện 20 lần thì không có nghĩa là tài liệu đó quan trọng gấp 20 lần tài liệu chỉ chứa từ *"election"* 1 lần. Công thức logarithmic giúp giảm bớt ảnh hưởng của các từ lặp lại nhiều lần trong cùng một bài báo, làm cho trọng số đặc trưng trở nên thực tế và ổn định hơn.
  4. `min_df=2`: Loại bỏ hoàn toàn các từ chỉ xuất hiện ở đúng 1 tài liệu trong toàn bộ tập Train. Các từ này thường là lỗi gõ chữ hoặc thuật ngữ quá cá biệt, không có khả năng tổng quát hóa cho mô hình dự báo.

---

## 5. Phân phối dữ liệu sạch cuối cùng (Final Dataset Distribution)

Tập dữ liệu gộp sạch cuối cùng sau khi tích hợp 6 nguồn dữ liệu thô và áp dụng cơ chế Capping 20k đạt quy mô tổng cộng **156,765 mẫu** và có phân bố nhãn chi tiết như sau:

| STT | Danh mục mục tiêu (Category) | Số lượng mẫu trước capping | Số lượng mẫu sau Capping 20k | Tỷ lệ phần trăm (%) | Trạng thái cắt tỉa |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Politics and society** | 46,979 | 20,000 | 12.76% | ▼ Cắt xuống 20,000 |
| 2 | **Lifestyle** | 27,188 | 20,000 | 12.76% | ▼ Cắt xuống 20,000 |
| 3 | **Entertainment** | 24,522 | 20,000 | 12.76% | ▼ Cắt xuống 20,000 |
| 4 | **Health** | 23,208 | 20,000 | 12.76% | ▼ Cắt xuống 20,000 |
| 5 | Family | 19,425 | 19,425 | 12.39% | — Giữ nguyên |
| 6 | Community | 15,864 | 15,864 | 10.12% | — Giữ nguyên |
| 7 | Business | 9,397 | 9,397 | 6.00% | — Giữ nguyên |
| 8 | Food & drinks | 8,271 | 8,271 | 5.28% | — Giữ nguyên |
| 9 | Sports | 6,925 | 6,925 | 4.42% | — Giữ nguyên |
| 10 | Tech & Science | 6,307 | 6,307 | 4.02% | — Giữ nguyên |
| 11 | Education | 3,823 | 3,823 | 2.44% | — Giữ nguyên |
| 12 | Environment | 3,488 | 3,488 | 2.22% | — Giữ nguyên |
| 13 | Arts & Culture | 3,265 | 3,265 | 2.08% | — Giữ nguyên |
| | **Tổng** | **198,662** | **156,765** | **100%** | ▼ Giảm 21% tổng mẫu |
