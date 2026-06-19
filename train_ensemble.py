import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("=== Training Ensemble Soft Voting Classifier on News Dataset ===")
    
    # Cấu hình tham số kiểm soát hiệu năng
    USE_SUBSET = True         # Đặt thành False nếu muốn huấn luyện trên TOÀN BỘ dữ liệu
    SUBSET_SIZE = 10000        # Số lượng mẫu lấy ngẫu nhiên để chạy nhanh
    MAX_FEATURES = 1500       # Số lượng từ vựng tối đa trong TF-IDF (giúp tránh tràn RAM)
    
    dataset_path = "cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}. Vui lòng chạy process_dataset.py trước.")
        return

    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv(dataset_path)
    
    # Điền giá trị rỗng cho cột văn bản nếu có lỗi NaN
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # 2. Lấy mẫu dữ liệu nếu được kích hoạt
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Đang lấy ngẫu nhiên phân lớp {SUBSET_SIZE} mẫu để huấn luyện nhanh...")
        df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)
    else:
        print(f"Sử dụng toàn bộ dữ liệu ({len(df)} mẫu).")
        
    X = df['cleaned_text']
    y = df['category']
    
    # 3. Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Trích xuất đặc trưng văn bản bằng TF-IDF
    print(f"Đang Vector hóa văn bản bằng TF-IDF (max_features={MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Đối với GaussianNB, chúng ta cần dữ liệu ở dạng mảng dày đặc (dense array)
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()
    
    # 5. Khởi tạo các mô hình thành phần
    print("Đang khởi tạo các mô hình phân loại...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    gnb = GaussianNB()
    
    # SVC cần probability=True để soft voting hoạt động, giới hạn max_iter=2000 để tránh lặp vô hạn
    svc = SVC(probability=True, random_state=42, max_iter=2000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # 6. Khởi tạo Soft Voting Ensemble
    ensemble_clf = VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('gnb', gnb),
            ('svc', svc),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    # Danh sách các mô hình
    models = {
        'Logistic Regression': (log_reg, False), # Sử dụng sparse matrix
        'Gaussian Naive Bayes': (gnb, True),      # Sử dụng dense matrix
        'Support Vector Machine (SVC)': (svc, False), # Sử dụng sparse matrix
        'Random Forest': (rf, False),             # Sử dụng sparse matrix
        'Ensemble (Soft Voting)': (ensemble_clf, True) # Sử dụng dense matrix vì có GaussianNB
    }
    
    # Lưu kết quả
    results = {}
    
    # 7. Huấn luyện và đánh giá từng mô hình
    for name, (model, needs_dense) in models.items():
        print(f"\n--- Đang huấn luyện & đánh giá: {name} ---")
        start_time = time.time()
        
        # Chọn dữ liệu phù hợp (dense cho GaussianNB, sparse cho các mô hình khác)
        X_tr = X_train_dense if needs_dense else X_train_tfidf
        X_te = X_test_dense if needs_dense else X_test_tfidf
        
        # Train
        model.fit(X_tr, y_train)
        
        # Predict
        y_pred = model.predict(X_te)
        
        elapsed_time = time.time() - start_time
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'Accuracy': acc, 'Time (s)': elapsed_time}
        
        print(f"Thời gian huấn luyện: {elapsed_time:.2f} giây")
        print(f"Accuracy: {acc:.4f}")
        
    # 8. In bảng tổng kết kết quả
    print("\n" + "="*50)
    print("BẢNG TỔNG KẾT KẾT QUẢ")
    print("="*50)
    print(f"{'Mô hình':<30} | {'Độ chính xác (Accuracy)':<25} | {'Thời gian chạy (s)':<15}")
    print("-"*75)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:.4%}{'':<17} | {metrics['Time (s)']:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()
