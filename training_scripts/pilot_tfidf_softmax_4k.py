import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from custom_models.softmax_regression import CustomSoftmaxRegression

# Stopwords
stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def main():
    print("=" * 80)
    print(" GIAI ĐOẠN 1: THỬ NGHIỆM ĐẶC TRƯNG TF-IDF + SOFTMAX REGRESSION (4K CAP)")
    print("=" * 80)

    dataset_path = "data/merged_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy {dataset_path}!")
        print("Vui lòng chạy trước: python data/data_preparation/merge_and_clean.py")
        return

    # 1. Đọc dữ liệu
    print("1. Đọc dữ liệu từ file...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text_raw", "category"])
    df = df[df["text_raw"].str.strip() != ""]
    print(f"   Tổng số mẫu: {len(df):,} | Số nhãn: {df['category'].nunique()}")

    # 2. Tiền xử lý: lowercase + remove stopwords
    print("\n2. Tiền xử lý văn bản (lowercase + remove stopwords)...")
    t0 = time.time()
    df["cleaned_text"] = df["text_raw"].apply(clean_text)
    print(f"   Xong trong {time.time() - t0:.1f}s")

    X_raw = df["cleaned_text"].values
    y     = df["category"].values

    # 3. Chia Train / Test (80/20 Stratified Split)
    print("\n3. Chia Train/Test (80/20 Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Kích thước tập Train gốc: {len(X_train_raw):,} | Kích thước tập Test: {len(X_test_raw):,}")

    # 4. Undersampling tập Train (Capping tối đa 4,000 mẫu/nhãn)
    print("\n4. Undersampling tập Train (Cap tối đa 4,000 mẫu/nhãn)...")
    df_train = pd.DataFrame({'text': X_train_raw, 'category': y_train})
    sampled_list = []

    for cat, group in df_train.groupby('category'):
        n_samples = min(len(group), 4000)
        sampled_list.append(group.sample(n_samples, random_state=42))

    df_train_sampled = pd.concat(sampled_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_train_raw_sampled = df_train_sampled['text'].values
    y_train_sampled = df_train_sampled['category'].values
    print(f"   => Kích thước tập Train sau Undersample: {len(X_train_raw_sampled):,} mẫu")

    # 5. Trích xuất đặc trưng TF-IDF
    print("\n5. Trích xuất đặc trưng TF-IDF...")
    tfidf_params = {
        'max_features': 50000,
        'ngram_range': (1, 3),
        'sublinear_tf': True,
        'min_df': 2
    }
    t_start_extract = time.time()
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_vec = vectorizer.fit_transform(X_train_raw_sampled)
    X_test_vec = vectorizer.transform(X_test_raw)
    t_extract = time.time() - t_start_extract
    print(f"   Hoàn thành TF-IDF trong {t_extract:.2f}s | Shape Train: {X_train_vec.shape}")

    # 6. Huấn luyện mô hình Softmax Regression cơ sở
    print("\n6. Huấn luyện Softmax Regression...")
    softmax_model = CustomSoftmaxRegression(
        lr=0.001, 
        epochs=100, 
        C=1.0, 
        class_weight='balanced', 
        batch_size=1024,
        verbose=True
    )
    t_start_train = time.time()
    softmax_model.fit(X_train_vec, y_train_sampled)
    t_train = time.time() - t_start_train
    print(f"   Huấn luyện xong trong {t_train:.2f}s")

    # 7. Đánh giá mô hình
    print("\n7. Đánh giá mô hình trên Test Set...")
    y_pred = softmax_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 60)
    print(" KẾT QUẢ ĐẶC TRƯNG TF-IDF:")
    print("=" * 60)
    print(f" Accuracy               : {acc:.4%}")
    print(f" Macro F1               : {f1:.4%}")
    print(f" Thời gian trích xuất   : {t_extract:.2f}s")
    print(f" Thời gian huấn luyện   : {t_train:.2f}s")
    print(f" Tổng thời gian chạy    : {t_extract + t_train:.2f}s")
    print("=" * 60)
    print("\nChi tiết Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
