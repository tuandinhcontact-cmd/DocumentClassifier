import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("=== Huấn luyện Mô hình Tối ưu hóa trên TOÀN BỘ 189,814 dữ liệu ===")
    
    dataset_path = "cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu sạch...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    X = df['cleaned_text']
    y = df['category']
    
    print(f"Tổng số mẫu: {len(df)}")
    
    # 2. Chia tập train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Vector hóa TF-IDF với Unigrams + Bigrams (từ đơn + cụm 2 từ)
    print("Đang trích xuất đặc trưng TF-IDF (N-grams: 1-2, Max Features: 40,000)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Sử dụng SGDClassifier để huấn luyện cực nhanh trên dữ liệu lớn
    # loss='log_loss' là Logistic Regression tối ưu bằng SGD
    print("Đang huấn luyện mô hình SGD Classifier (Logistic Regression)...")
    start_time = time.time()
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)
    
    # Đánh giá
    y_pred = clf.predict(X_test_tfidf)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nThời gian huấn luyện: {elapsed_time:.2f} giây")
    print(f"Độ chính xác (Accuracy) trên toàn bộ dữ liệu: {acc:.4%}")
    
    # In báo cáo chi tiết cho một số lớp tiêu biểu
    print("\nBáo cáo chi tiết phân loại (Classification Report):")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
