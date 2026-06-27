import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE


from custom_models.softmax_regression import CustomSoftmaxRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.flat_ensemble import CustomOneVsRestClassifier, FlatSoftVotingClassifier

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
    print(" HUẤN LUYỆN VÀ XUẤT MÔ HÌNH TỐI ƯU NHẤT RA PRODUCTION")
    print(" (Softmax + NB + LinearSVM | SMOTE | 100 Epochs)")
    print("=" * 80)

    # ── 1. Đọc dữ liệu từ file đã chuẩn bị ──────────────────────────────────
    dataset_path = "data/merged_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"⚠️  Không tìm thấy {dataset_path}!")
        print("    Vui lòng chạy trước: python data/data_preparation/merge_and_clean.py")
        return

    print(f"\n1. Đọc dữ liệu từ {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text_raw", "category"])
    df = df[df["text_raw"].str.strip() != ""]
    print(f"   Tổng: {len(df):,} dòng | {df['category'].nunique()} nhãn")
    print("   Phân bố nhãn:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"     - {cat:<25}: {cnt:,} mẫu")

    # ── 2. Tiền xử lý: lowercase + remove stopwords ───────────────────────────
    print("\n2. Tiền xử lý văn bản (lowercase + remove stopwords)...")
    t_start = time.time()
    df["cleaned_text"] = df["text_raw"].apply(clean_text)
    print(f"   Xong trong {time.time() - t_start:.1f}s")

    X = df["cleaned_text"].values
    y  = df["category"].values

    # ── 3. Train/Test split (80/20 Stratified) ────────────────────────────────
    print("\n3. Chia Train/Test (80/20 Stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Kích thước tập Train: {len(X_train_raw):,} | Tập Test: {len(X_test_raw):,}")

    # ── 4. Đọc params tốt nhất từ Grid Search (nếu có) ────────────────────────
    params_path = "models/best_params.json"
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        tfidf_p   = best_params.get("tfidf",   {"max_features": 50000, "ngram_range": [1, 3], "sublinear_tf": True, "min_df": 2})
        nb_p      = best_params.get("nb",      {"alpha": 0.1})
        softmax_p = best_params.get("softmax", {"lr": 0.001, "epochs": 100})
        svm_p     = best_params.get("svm",     {"lambda_param": 0.1, "epochs": 100})
        print(f"\n4. Đã tìm thấy {params_path} — dùng params từ Grid Search:")
        print(f"   TF-IDF   : {tfidf_p}")
        print(f"   NB       : {nb_p}")
        print(f"   Softmax  : {softmax_p}")
        print(f"   SVM      : {svm_p}")
    else:
        print("\n4. Không có best_params.json — dùng tham số mặc định.")
        print("   (Chạy train_individual_models_comparison.py trước để tìm params tối ưu)")
        tfidf_p   = {"max_features": 50000, "ngram_range": [1, 3], "sublinear_tf": True, "min_df": 2}
        nb_p      = {"alpha": 0.1}
        softmax_p = {"lr": 0.001, "epochs": 100}
        svm_p     = {"lambda_param": 0.1, "epochs": 100}

    # ── 5. TF-IDF ───────────────────────────────────────────────────────
    print("\n5. Trích xuất đặc trưng TF-IDF...")
    t0 = time.time()
    # ngram_range trong JSON lưu dưới dạng list — cần convert sang tuple
    tfidf_p["ngram_range"] = tuple(tfidf_p["ngram_range"])
    vectorizer = TfidfVectorizer(**tfidf_p)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec  = vectorizer.transform(X_test_raw)
    print(f"   Xong trong {time.time() - t0:.1f}s | Số đặc trưng: {X_train_vec.shape[1]:,}")


    # ── 6. SMOTE ─────────────────────────────────────────────────────────
    print("\n6. Áp dụng SMOTE để cân bằng tập Train...")
    t_smote = time.time()
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)
    print(f"   Xong trong {time.time() - t_smote:.1f}s | Kích thước mới: {X_train_smote.shape[0]:,} mẫu")

    # ── 7. Huấn luyện các mô hình sản phẩm ──────────────────────────────
    print("\n7. Huấn luyện các mô hình sản phẩm (Ensemble bộ đôi & Linear SVM độc lập)...")
    t0 = time.time()

    # A. Bộ đôi Ensemble (NB + Softmax)
    print("   -> Huấn luyện Ensemble (NB + Softmax)...")
    nb_est = CustomMultinomialNB(**nb_p)
    lr_est = CustomSoftmaxRegression(
        lr=softmax_p["lr"], epochs=softmax_p["epochs"],
        C=1.0, class_weight=None, beta1=0.9, beta2=0.999, verbose=True
    )
    ensemble = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_est),
        ('SoftmaxRegression', lr_est),
    ])
    ensemble.fit(X_train_smote, y_train_smote)
    
    # B. Linear SVM (Chạy độc lập)
    print("   -> Huấn luyện Linear SVM OVR độc lập...")
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(
            lr=0.01, lambda_param=svm_p["lambda_param"], epochs=svm_p["epochs"],
            class_weight=None, beta1=0.8, beta2=0.999
        )
    )
    svm_est.fit(X_train_smote, y_train_smote)
    
    print(f"   Huấn luyện xong tất cả trong {time.time() - t0:.1f}s")
    
    # Đánh giá trên tập Test
    y_pred = ensemble.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n" + "=" * 60)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)
    print(f" Accuracy : {acc:.4%}")
    print(f" Macro F1 : {f1:.4%}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 60)
    
    # Lưu mô hình vào models/flat_gridsearch_model.pkl
    out_path = "models/flat_gridsearch_model.pkl"
    print(f"   Đang xuất mô hình ra {out_path}...")
    model_data = {
        'vectorizer': vectorizer,
        'model': ensemble,
        'svm_model': svm_est,
        'best_params': {
            'tfidf': tfidf_p,
            'nb': nb_p,
            'softmax': softmax_p,
            'svm': svm_p
        },
        'classes': list(np.unique(y))
    }
    
    with open(out_path, "wb") as f:
        pickle.dump(model_data, f)
    print("   ✅ Đã lưu mô hình thành công!")

if __name__ == "__main__":
    main()
