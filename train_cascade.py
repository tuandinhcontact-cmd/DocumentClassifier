import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

class HierarchicalCascadeClassifier:
    def __init__(self, base_estimator_class=LogisticRegression, **estimator_params):
        self.base_estimator_class = base_estimator_class
        self.estimator_params = estimator_params
        self.models = {}
        self.classes_order = []
        
    def fit(self, X, y):
        # Sắp xếp các lớp theo thứ tự số lượng mẫu giảm dần (lớp nhiều mẫu nhất xếp đầu)
        class_counts = pd.Series(y).value_counts()
        self.classes_order = class_counts.index.tolist()
        
        current_X = X
        current_y = np.array(y)
        
        print("\n--- Bắt đầu huấn luyện chuỗi phân cấp (Cascade) ---")
        # Huấn luyện chuỗi model cho N-1 lớp đầu tiên
        for i, class_name in enumerate(self.classes_order[:-1]):
            # Tạo nhãn nhị phân: 1 nếu là lớp hiện tại, 0 nếu thuộc về các lớp còn lại
            y_binary = (current_y == class_name).astype(int)
            
            # Khởi tạo và huấn luyện model nhị phân
            model = self.base_estimator_class(**self.estimator_params)
            model.fit(current_X, y_binary)
            
            # Lưu model
            self.models[class_name] = model
            
            # Lọc bớt dữ liệu thuộc lớp hiện tại ra khỏi tập huấn luyện của bước tiếp theo
            remaining_mask = (y_binary == 0)
            
            # Nếu không còn dữ liệu thuộc các lớp khác, dừng sớm
            if not remaining_mask.any():
                break
                
            # Cập nhật dữ liệu cho model tiếp theo
            if hasattr(current_X, "tocsr"): # Nếu là sparse matrix của scipy
                current_X = current_X[remaining_mask]
            else:
                current_X = current_X[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Model {i+1}: Đã huấn luyện xong lớp '{class_name}'. Số mẫu còn lại cho bước sau: {len(current_y)}")
            
        print("Huấn luyện thành công toàn bộ chuỗi phân cấp!")
        
    def predict(self, X):
        num_samples = X.shape[0]
        # Mặc định kết quả ban đầu là lớp cuối cùng trong chuỗi
        predictions = np.full(num_samples, self.classes_order[-1], dtype=object)
        
        # Mảng đánh dấu các dòng đã được phân loại xong
        classified = np.zeros(num_samples, dtype=bool)
        
        # Duyệt qua từng mô hình nhị phân trong chuỗi
        for class_name in self.classes_order[:-1]:
            if class_name not in self.models:
                continue
                
            # Chỉ dự đoán trên các mẫu chưa được phân loại
            unclassified_indices = np.where(~classified)[0]
            if len(unclassified_indices) == 0:
                break
                
            X_unclass = X[unclassified_indices]
            model = self.models[class_name]
            
            # Dự đoán nhị phân (1: thuộc lớp này, 0: thuộc các lớp còn lại)
            preds_binary = model.predict(X_unclass)
            
            # Cập nhật kết quả cho các mẫu được dự đoán là 1
            is_class_indices = unclassified_indices[preds_binary == 1]
            predictions[is_class_indices] = class_name
            classified[is_class_indices] = True
            
        return predictions

def main():
    print("=== Thử nghiệm mô hình Hierarchical Cascade Classifier ===")
    
    # Cấu hình thử nghiệm
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 30000
    
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Lấy mẫu ngẫu nhiên phân lớp {SUBSET_SIZE} dòng...")
        df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text']
    y = df['category']
    
    # 2. Chia tập Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. TF-IDF vectorization
    print("Vector hóa văn bản bằng TF-IDF (N-grams: 1-2)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 4. Huấn luyện mô hình Cascade tùy chỉnh
    print("\nHuấn luyện mô hình Cascade...")
    start_time = time.time()
    cascade_clf = HierarchicalCascadeClassifier(
        base_estimator_class=LogisticRegression, 
        max_iter=1000, 
        class_weight='balanced', # Sử dụng class_weight nhị phân để tối ưu hóa việc phân chia One vs Rest
        random_state=42
    )
    cascade_clf.fit(X_train, y_train)
    
    # Dự đoán
    y_pred_cascade = cascade_clf.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc_cascade = accuracy_score(y_test, y_pred_cascade)
    macro_f1_cascade = f1_score(y_test, y_pred_cascade, average='macro')
    
    # 5. Huấn luyện mô hình Logistic Regression đa lớp thông thường để so sánh
    print("\nHuấn luyện mô hình Logistic Regression đa lớp thông thường (Multi-class)...")
    clf_multi = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf_multi.fit(X_train, y_train)
    y_pred_multi = clf_multi.predict(X_test)
    
    acc_multi = accuracy_score(y_test, y_pred_multi)
    macro_f1_multi = f1_score(y_test, y_pred_multi, average='macro')
    
    # 6. In bảng so sánh kết quả
    print("\n" + "="*80)
    print("BẢNG SO SÁNH KẾT QUẢ")
    print("="*80)
    print(f"{'Mô hình':<40} | {'Accuracy':<12} | {'Macro F1-Score':<15} | {'Thời gian chạy':<15}")
    print("-"*80)
    print(f"{'Multi-class Logistic Regression':<40} | {acc_multi:.4%}   | {macro_f1_multi:.4%}       | {elapsed_time:.2f}s")
    print(f"{'Hierarchical Cascade Classifier (Custom)':<40} | {acc_cascade:.4%}   | {macro_f1_cascade:.4%}       | {elapsed_time:.2f}s")
    print("="*80)
    
    print("\nBÁO CÁO CHI TIẾT CỦA MÔ HÌNH HIERARCHICAL CASCADE CLASSIFIER:")
    print(classification_report(y_test, y_pred_cascade))

if __name__ == "__main__":
    main()
