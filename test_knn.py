import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

def main():
    print("=== Thử nghiệm K-Nearest Neighbors (K-NN) với TF-IDF ===")
    
    # 1. Đọc dữ liệu
    dataset_path = "data/merged_cleaned_dataset.csv"
    print(f"Đang đọc dữ liệu từ {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # ⚠️ K-NN tính toán khoảng cách rất chậm trên tập dữ liệu lớn (Dự đoán O(N * D))
    # Nên lấy ngẫu nhiên một mẫu vừa đủ (VD: 30,000 bài báo) để chạy trong vài phút.
    # Nếu chạy trên toàn bộ 182k bài báo, thời gian Predict có thể lên tới hàng tiếng đồng hồ.
    sample_size = 30000
    print(f"\nDo thuật toán K-NN có độ phức tạp dự đoán là O(N*D), lấy ngẫu nhiên {sample_size} mẫu để kiểm tra độ chính xác...")
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    X = df['cleaned_text']
    y = df['category']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Vector hóa TF-IDF (20,000 features)
    print("Đang Vector hóa TF-IDF (20,000 features)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 3. Chạy K-NN
    # Dùng n_neighbors = 5 mặc định, n_jobs=-1 để chạy đa luồng
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric='cosine')
    
    print("\n-> Đang train K-NN (K-NN không có quá trình train thực sự, chỉ lưu dữ liệu vào RAM)...")
    start_train = time.time()
    knn.fit(X_train, y_train)
    print(f"   Thời gian Fit: {time.time() - start_train:.2f} giây")
    
    print("\n-> Đang dự đoán (Quá trình này cực kỳ tốn thời gian tính toán khoảng cách)...")
    start_pred = time.time()
    y_pred = knn.predict(X_test)
    pred_time = time.time() - start_pred
    print(f"   Thời gian Predict trên {len(y_test)} mẫu: {pred_time:.2f} giây")
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n🌟 KẾT QUẢ CỦA K-NN:")
    print(f"   Accuracy: {acc:.4%}")
    print(f"   Macro F1: {f1:.4%}")

if __name__ == "__main__":
    main()
