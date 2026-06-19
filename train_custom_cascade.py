import os
import pickle
import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Import 4 thuật toán tự viết từ đầu
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.gaussian_nb import CustomGaussianNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.random_forest import CustomRandomForest

# 1. Định nghĩa bộ phân lớp Soft Voting Ensemble từ đầu
class CustomVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators # Danh sách tuple: (name, estimator_object)
        
    def fit(self, X, y):
        for name, est in self.estimators:
            # Khởi chạy fit của từng mô hình nhị phân tự viết
            est.fit(X, y)
        return self
        
    def predict_proba(self, X):
        # Tính toán xác suất dự đoán lớp và lấy trung bình cộng (Soft Voting)
        probs = [est.predict_proba(X) for name, est in self.estimators]
        return np.mean(probs, axis=0)
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# 2. Định nghĩa mô hình Cascade từ đầu (sử dụng copy.deepcopy để nhân bản mô hình tự viết)
class CustomHierarchicalCascadeClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = {}
        self.classes_order = []
        
    def fit(self, X, y):
        # Sắp xếp danh mục từ nhiều mẫu đến ít mẫu
        class_counts = pd.Series(y).value_counts()
        self.classes_order = class_counts.index.tolist()
        
        current_X = X
        current_y = np.array(y)
        
        print("\n--- Bắt đầu huấn luyện chuỗi Custom Cascade (4 thuật toán tự viết) ---")
        for i, class_name in enumerate(self.classes_order[:-1]):
            # Tạo nhãn nhị phân
            y_binary = (current_y == class_name).astype(int)
            
            # Sử dụng deepcopy để clone mô hình tùy chỉnh thuần Python
            model = copy.deepcopy(self.base_estimator)
            
            start_step = time.time()
            model.fit(current_X, y_binary)
            step_time = time.time() - start_step
            
            self.models[class_name] = model
            
            remaining_mask = (y_binary == 0)
            if not remaining_mask.any():
                break
                
            if hasattr(current_X, "tocsr"):
                current_X = current_X[remaining_mask]
            else:
                current_X = current_X[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Bước {i+1} [{class_name}]: Hoàn tất huấn luyện trong {step_time:.2f}s. Còn lại: {len(current_y)} mẫu.")
            
        print("Huấn luyện thành công toàn bộ chuỗi Custom Cascade!")
        
    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.full(num_samples, self.classes_order[-1], dtype=object)
        classified = np.zeros(num_samples, dtype=bool)
        
        for class_name in self.classes_order[:-1]:
            if class_name not in self.models:
                continue
                
            unclassified_indices = np.where(~classified)[0]
            if len(unclassified_indices) == 0:
                break
                
            X_unclass = X[unclassified_indices]
            model = self.models[class_name]
            
            preds_binary = model.predict(X_unclass)
            
            is_class_indices = unclassified_indices[preds_binary == 1]
            predictions[is_class_indices] = class_name
            classified[is_class_indices] = True
            
        return predictions

def main():
    print("=== Huấn luyện Mô hình Cascade với 4 thuật toán tự viết từ đầu ===")
    
    # Cấu hình tham số hiệu năng
    USE_SUBSET = False
    SUBSET_SIZE = 10000
    MAX_FEATURES = 25000
    
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Lấy mẫu phân lớp ngẫu nhiên {SUBSET_SIZE} dòng...")
        df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text']
    y = df['category']
    
    # 2. Chia tập Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. TF-IDF vectorization
    print("Vector hóa văn bản bằng TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 4. Khởi tạo 4 mô hình tự viết nhị phân
    print("Đang khởi tạo các mô hình tự viết nhị phân...")
    lr = CustomLogisticRegression(lr=0.5, epochs=100, C=1.0, class_weight='balanced')
    gnb = CustomGaussianNB()
    svm = CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20)
    # Tốc độ cây quyết định thuần python chậm hơn, nên chúng ta giới hạn 3 cây quyết định, độ sâu 4
    rf = CustomRandomForest(n_estimators=3, max_depth=4, min_samples_split=2)
    
    custom_ensemble = CustomVotingClassifier(
        estimators=[
            ('lr', lr),
            ('gnb', gnb),
            ('svm', svm),
            ('rf', rf)
        ]
    )
    
    # 5. Khởi tạo và huấn luyện chuỗi Cascade
    start_time = time.time()
    cascade_custom = CustomHierarchicalCascadeClassifier(base_estimator=custom_ensemble)
    cascade_custom.fit(X_train, y_train)
    
    # 6. Dự đoán và đánh giá
    print("\nĐang dự đoán trên tập kiểm thử...")
    y_pred = cascade_custom.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTổng thời gian huấn luyện: {elapsed_time:.2f} giây")
    print(f"Độ chính xác (Accuracy): {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    # 7. Đóng gói mô hình ghi đè lên 'cascade_model.pkl' để giao diện Web chạy bằng mô hình mới
    print("\nĐóng gói mô hình mới cho giao diện Web...")
    model_data = {
        'model': cascade_custom,
        'vectorizer': vectorizer,
        'classes_order': cascade_custom.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Ghi đè thành công tệp 'cascade_model.pkl'!")

if __name__ == "__main__":
    main()
