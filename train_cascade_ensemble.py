import os
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.base import clone

class HierarchicalCascadeEnsembleClassifier:
    def __init__(self, base_ensemble_estimator):
        self.base_ensemble_estimator = base_ensemble_estimator
        self.models = {}
        self.classes_order = []
        
    def fit(self, X, y):
        # Sắp xếp các lớp theo thứ tự số lượng mẫu giảm dần
        class_counts = pd.Series(y).value_counts()
        self.classes_order = class_counts.index.tolist()
        
        current_X = X
        current_y = np.array(y)
        
        print("\n--- Bắt đầu huấn luyện chuỗi phân cấp Ensemble Cascade (One-vs-Rest) ---")
        for i, class_name in enumerate(self.classes_order[:-1]):
            # Tạo nhãn nhị phân: 1 nếu thuộc lớp hiện tại, 0 nếu thuộc các lớp còn lại
            y_binary = (current_y == class_name).astype(int)
            
            # Clone mô hình Ensemble để tạo bản sao chưa huấn luyện cho bước hiện tại
            model = clone(self.base_ensemble_estimator)
            
            start_step = time.time()
            model.fit(current_X, y_binary)
            step_time = time.time() - start_step
            
            # Lưu model
            self.models[class_name] = model
            
            # Lọc dữ liệu còn lại cho bước sau
            remaining_mask = (y_binary == 0)
            if not remaining_mask.any():
                break
                
            if hasattr(current_X, "tocsr"):
                current_X = current_X[remaining_mask]
            else:
                current_X = current_X[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Bước {i+1} [{class_name}]: Hoàn tất huấn luyện trong {step_time:.2f}s. Còn lại: {len(current_y)} mẫu.")
            
        print("Huấn luyện thành công toàn bộ chuỗi Ensemble Cascade!")
        
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
    print("=== Thử nghiệm mô hình Ensemble Soft Voting Cascade ===")
    
    # Huấn luyện trên toàn bộ dữ liệu 182,081 mẫu
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 25000
    
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy tệp {dataset_path}.")
        return

    # 1. Đọc dữ liệu
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Lấy mẫu ngẫu nhiên {SUBSET_SIZE} dòng...")
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
    
    # 4. Định nghĩa 4 mô hình thành phần cho Soft Voting nhị phân
    print("Khởi tạo bộ phân loại Soft Voting Ensemble (4 mô hình)...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    mnb = MultinomialNB()
    
    # Sử dụng LinearSVC được hiệu chuẩn xác suất qua CalibratedClassifierCV
    # giúp SVM chạy siêu tốc trên dữ liệu lớn (nhanh gấp ngàn lần so với SVC truyền thống)
    svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual=False))
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    binary_ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('mnb', mnb),
            ('svm', svm),
            ('rf', rf)
        ],
        voting='soft'
    )
    
    # 5. Khởi tạo và huấn luyện mô hình Cascade với mô hình cơ sở là Ensemble
    start_time = time.time()
    cascade_ensemble_clf = HierarchicalCascadeEnsembleClassifier(base_ensemble_estimator=binary_ensemble)
    cascade_ensemble_clf.fit(X_train, y_train)
    
    # Dự đoán
    print("\nĐang dự đoán trên tập kiểm thử...")
    y_pred = cascade_ensemble_clf.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTổng thời gian huấn luyện: {elapsed_time:.2f} giây")
    print(f"Độ chính xác (Accuracy): {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))
    
    # 6. Đóng gói mô hình mới này ghi đè lên 'cascade_model.pkl' để giao diện Web tự động sử dụng mô hình mới
    print("\nĐang đóng gói mô hình mới cho giao diện Web sử dụng...")
    model_data = {
        'model': cascade_ensemble_clf,
        'vectorizer': vectorizer,
        'classes_order': cascade_ensemble_clf.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Ghi đè thành công lên 'cascade_model.pkl'!")

if __name__ == "__main__":
    main()
