import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from train_cascade import HierarchicalCascadeClassifier

def main():
    print("=== Huấn luyện và Đóng gói mô hình Cascade Classifier ===")
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # Đọc dữ liệu
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Huấn luyện trên 50,000 mẫu để mô hình đạt độ chính xác cao và chạy nhanh
    TRAIN_SIZE = 50000
    if len(df) > TRAIN_SIZE:
        print(f"Lấy mẫu phân lớp {TRAIN_SIZE} dòng để huấn luyện...")
        df, _ = train_test_split(df, train_size=TRAIN_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text']
    y = df['category']
    
    # Vector hóa TF-IDF với Bigrams
    print("Vector hóa dữ liệu...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=25000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Khởi tạo và huấn luyện Cascade
    clf = HierarchicalCascadeClassifier(
        base_estimator_class=LogisticRegression,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_tfidf, y)
    
    # Lưu mô hình, vectorizer và danh sách các lớp
    model_data = {
        'model': clf,
        'vectorizer': vectorizer,
        'classes_order': clf.classes_order
    }
    
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
        
    print("\nĐã đóng gói mô hình thành công vào tệp 'cascade_model.pkl'!")

if __name__ == "__main__":
    main()
