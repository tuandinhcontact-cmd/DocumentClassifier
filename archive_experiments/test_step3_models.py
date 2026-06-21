import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from train_3step_cascade import CustomOneVsRestClassifier

def main():
    print("=== ĐÁNH GIÁ ĐỘC LẬP CÁC MÔ HÌNH CHO BƯỚC 3 (12 Nhãn thiểu số) ===")
    
    dataset_path = "data/merged_cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Lọc bỏ 2 nhãn lớn (Tech & Science, Politics and society) để giả lập môi trường Bước 3
    print("\n1. Lọc dữ liệu cho Bước 3...")
    df_step3 = df[~df['category'].isin(['Tech & Science', 'Politics and society'])]
    print(f"Số lượng bài báo còn lại cho Bước 3: {len(df_step3)} bài báo.")
    
    X = df_step3['cleaned_text']
    y = df_step3['category']
    
    # Chia Train/Test (Đảm bảo giống hệt tỷ lệ của luồng Cascade)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n2. Vector hóa TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print("\n3. Khởi tạo 3 Mô hình ứng cử viên...")
    mnb = CustomMultinomialNB(alpha=1.0)
    lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced'))
    svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight='balanced'))
    
    models = {
        "MultinomialNB (Xác suất Độc lập)": mnb,
        "LogisticRegression OVR (Adam Optimizer)": lr_ovr,
        "LinearSVM OVR (Hinge Loss + Platt)": svm_ovr
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n-> Đang huấn luyện và dự đoán: {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        elapsed_time = time.time() - start_time
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results.append((name, acc, f1, elapsed_time))
        print(f"   Hoàn thành trong {elapsed_time:.2f} giây | Accuracy: {acc:.4%} | Macro F1: {f1:.4%}")
        
    print("\n=== TỔNG KẾT BẢNG XẾP HẠNG BƯỚC 3 ===")
    results.sort(key=lambda x: x[1], reverse=True) # Sort by Accuracy
    for i, (name, acc, f1, t) in enumerate(results):
        print(f"Top {i+1}: {name}")
        print(f"         Accuracy: {acc:.4%} | Macro F1: {f1:.4%} | Thời gian: {t:.2f}s")

if __name__ == "__main__":
    main()
