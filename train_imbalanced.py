import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

def main():
    print("=== So sánh các phương pháp xử lý mất cân bằng dữ liệu ===")
    
    # Do SMOTE trên ma trận TF-IDF rất nặng và tốn RAM, 
    # chúng ta sẽ lấy mẫu ngẫu nhiên có phân lớp (stratified subset) để demo nhanh hiệu quả.
    USE_SUBSET = True
    SUBSET_SIZE = 25000       # Số lượng mẫu huấn luyện demo
    MAX_FEATURES = 2500       # Giới hạn đặc trưng TF-IDF để SMOTE hoạt động ổn định
    
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu sạch đã gộp...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Lấy mẫu
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Lấy mẫu ngẫu nhiên phân lớp {SUBSET_SIZE} dòng...")
        # Sử dụng stratify trên category để giữ nguyên tỷ lệ mất cân bằng dữ liệu
        df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text']
    y = df['category']
    
    print("\nPhân phối nhãn trong mẫu huấn luyện:")
    print(y.value_counts())
    
    # 2. Chia tập Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Vector hóa bằng TF-IDF
    print(f"\nVector hóa văn bản bằng TF-IDF (max_features={MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # ==========================================
    # Cấu hình 1: Baseline (Không xử lý mất cân bằng)
    # ==========================================
    print("\n--- 1. Đang chạy Baseline Logistic Regression... ---")
    start = time.time()
    clf_baseline = LogisticRegression(max_iter=1000, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    time_baseline = time.time() - start
    
    # ==========================================
    # Cấu hình 2: Class Weighting (Cân bằng bằng trọng số phạt)
    # ==========================================
    print("\n--- 2. Đang chạy Logistic Regression với Class Weights... ---")
    start = time.time()
    clf_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_weighted.fit(X_train, y_train)
    y_pred_weighted = clf_weighted.predict(X_test)
    time_weighted = time.time() - start
    
    # ==========================================
    # Cấu hình 3: SMOTE (Oversampling dữ liệu thiểu số)
    # ==========================================
    print("\n--- 3. Đang chạy SMOTE + Logistic Regression... ---")
    start = time.time()
    
    # Áp dụng SMOTE lên tập Train
    print("Đang áp dụng SMOTE để sinh dữ liệu nhân tạo (Oversampling)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Số lượng mẫu sau khi áp dụng SMOTE: {X_train_resampled.shape[0]} (so với ban đầu {X_train.shape[0]})")
    
    clf_smote = LogisticRegression(max_iter=1000, random_state=42)
    clf_smote.fit(X_train_resampled, y_train_resampled)
    y_pred_smote = clf_smote.predict(X_test)
    time_smote = time.time() - start
    
    # ==========================================
    # Báo cáo tổng kết
    # ==========================================
    results = {
        'Baseline': (y_pred_baseline, time_baseline),
        'Class Weighting': (y_pred_weighted, time_weighted),
        'SMOTE': (y_pred_smote, time_smote)
    }
    
    print("\n" + "="*80)
    print("BẢNG SO SÁNH KẾT QUẢ")
    print("="*80)
    print(f"{'Phương pháp':<25} | {'Accuracy':<10} | {'Macro F1-Score':<15} | {'Thời gian chạy':<15}")
    print("-"*80)
    for name, (y_pred, t) in results.items():
        acc = accuracy_score(y_test, y_pred)
        # Sử dụng macro F1 vì nó đánh giá công bằng hiệu năng trên cả các lớp thiểu số
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{name:<25} | {acc:.4%}  | {macro_f1:.4%}       | {t:.2f}s")
    print("="*80)
    
    # In chi tiết báo cáo F1 của từng lớp cho thấy sự khác biệt của các thuật toán
    print("\nBÁO CÁO F1-SCORE CHI TIẾT CỦA PHƯƠNG PHÁP CLASS WEIGHTING:")
    print(classification_report(y_test, y_pred_weighted))

if __name__ == "__main__":
    main()
