import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import time

def document_vector(tokenized_doc, w2v_model, vector_size):
    """
    Tạo vector cho văn bản bằng cách lấy trung bình cộng 
    các vector từ của những từ có trong từ điển Word2Vec.
    """
    words = [word for word in tokenized_doc if word in w2v_model.wv]
    if len(words) >= 1:
        return np.mean(w2v_model.wv[words], axis=0)
    else:
        return np.zeros(vector_size)

def main():
    print("=== Training Ensemble Soft Voting with Word2Vec & Multinomial NB ===")
    
    # Cấu hình tham số hiệu năng
    USE_SUBSET = True
    SUBSET_SIZE = 10000
    VECTOR_SIZE = 100 # Độ dài vector Word2Vec (100 dimensions)
    
    dataset_path = "cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}. Vui lòng chạy process_dataset.py trước.")
        return

    # 1. Đọc dữ liệu
    print("Đang đọc dữ liệu...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # Lấy mẫu dữ liệu
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Đang lấy ngẫu nhiên {SUBSET_SIZE} mẫu để huấn luyện nhanh...")
        df = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)
    else:
        print(f"Sử dụng toàn bộ dữ liệu ({len(df)} mẫu).")
        
    # Tách từ (Tokenize) dữ liệu văn bản
    print("Đang tách từ dữ liệu...")
    tokenized_texts = [text.split() for text in df['cleaned_text']]
    y = df['category']
    
    # 2. Chia tập train/test
    X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
        tokenized_texts, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Huấn luyện mô hình Word2Vec trên tập Train
    print(f"Đang huấn luyện mô hình Word2Vec (vector_size={VECTOR_SIZE})...")
    w2v_model = Word2Vec(
        sentences=X_train_tokens, 
        vector_size=VECTOR_SIZE, 
        window=5, 
        min_count=1, 
        workers=4,
        seed=42
    )
    
    # 4. Chuyển đổi các văn bản thành các vector trung bình
    print("Đang chuyển đổi văn bản sang vector Word2Vec...")
    X_train_w2v = np.array([document_vector(doc, w2v_model, VECTOR_SIZE) for doc in X_train_tokens])
    X_test_w2v = np.array([document_vector(doc, w2v_model, VECTOR_SIZE) for doc in X_test_tokens])
    
    # 5. MinMaxScaler: MultinomialNB KHÔNG chấp nhận giá trị âm. 
    # Do các vector Word2Vec chứa cả giá trị âm, chúng ta phải scale chúng về khoảng không âm [0, 1].
    print("Đang chuẩn hóa vector sang khoảng không âm [0, 1] bằng MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_w2v)
    X_test_scaled = scaler.transform(X_test_w2v)
    
    # 6. Khởi tạo các mô hình phân loại
    print("Đang khởi tạo các mô hình phân loại...")
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    mnb = MultinomialNB()
    
    # SVC cần probability=True để soft voting hoạt động
    svc = SVC(probability=True, random_state=42, max_iter=2000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Khởi tạo Ensemble Soft Voting
    ensemble_clf = VotingClassifier(
        estimators=[
            ('lr', log_reg),
            ('mnb', mnb),
            ('svc', svc),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    # Danh sách các mô hình (Tất cả đều dùng chung dữ liệu đã scale không âm)
    models = {
        'Logistic Regression': log_reg,
        'Multinomial Naive Bayes': mnb,
        'Support Vector Machine (SVC)': svc,
        'Random Forest': rf,
        'Ensemble (Soft Voting)': ensemble_clf
    }
    
    results = {}
    
    # 7. Huấn luyện và đánh giá
    for name, model in models.items():
        print(f"\n--- Đang huấn luyện & đánh giá: {name} ---")
        start_time = time.time()
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        elapsed_time = time.time() - start_time
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'Accuracy': acc, 'Time (s)': elapsed_time}
        
        print(f"Thời gian huấn luyện: {elapsed_time:.2f} giây")
        print(f"Accuracy: {acc:.4f}")
        
    # 8. In bảng tổng kết kết quả
    print("\n" + "="*50)
    print("BẢNG TỔNG KẾT KẾT QUẢ (Word2Vec + Multinomial NB)")
    print("="*50)
    print(f"{'Mô hình':<30} | {'Độ chính xác (Accuracy)':<25} | {'Thời gian chạy (s)':<15}")
    print("-"*75)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:.4%}{'':<17} | {metrics['Time (s)']:.2f}s")
    print("="*50)

if __name__ == "__main__":
    main()
