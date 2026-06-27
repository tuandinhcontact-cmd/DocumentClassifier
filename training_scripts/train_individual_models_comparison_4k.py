import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
import pandas as pd
import time
import copy
from itertools import product
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score


def clean_text(text):
    """Tiền xử lý văn bản: lowercase + loại ký tự đặc biệt + remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    return " ".join(w for w in words if w not in ENGLISH_STOP_WORDS)


from custom_models.softmax_regression import CustomSoftmaxRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.flat_ensemble import CustomOneVsRestClassifier, FlatSoftVotingClassifier


# ─────────────────────────────────────────────────────
# Grid Search Helper
# ─────────────────────────────────────────────────────
def grid_search_cv(model_builder, param_grid, X, y, cv=3, label=""):
    """
    model_builder: function taking parameters and returning a model instance
    param_grid: dict of parameter lists
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    print(f"\n[Grid Search] {label}: {len(combos)} combinations × {cv} folds = {len(combos)*cv} fits")
    best_score = -1
    best_params = None
    results = []

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = model_builder(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro', zero_division=0)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        results.append((params, mean_score))

        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"    [{param_str}] → CV Macro F1 = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"  => Best {label}: {best_params} (CV Macro F1 = {best_score:.4f})")
    return best_params, best_score, results


# ─────────────────────────────────────────────────────
# Main Experiment Script
# ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" THỬ NGHIỆM SO SÁNH INDIVIDUAL MODELS vs SOFT VOTING ENSEMBLE")
    print(" Dữ liệu: merged_dataset.csv | Cap: 4,000 mẫu/nhãn | CV: 3-Fold")
    print("=" * 70)

    dataset_path = "data/merged_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Không tìm thấy {dataset_path}!")
        print("Vui lòng chạy trước: python data/data_preparation/merge_and_clean.py")
        return

    # 1. Đọc dữ liệu
    print("1. Đọc dữ liệu từ file...")
    t_start = time.time()
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text_raw", "category"])
    df = df[df["text_raw"].str.strip() != ""]
    print(f"   Tổng số mẫu: {len(df):,} | Số nhãn: {df['category'].nunique()}")
    print("   Phân bố nhãn ban đầu:")
    for cat, count in df["category"].value_counts().items():
        print(f"     - {cat}: {count:,} mẫu")

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

    print("   Phân bố tập Train sau khi Undersample:")
    for cat, group in df_train.groupby('category'):
        n_samples = min(len(group), 4000)
        sampled_list.append(group.sample(n_samples, random_state=42))
        print(f"     - {cat:25s} : {len(group):,} mẫu -> {n_samples:,} mẫu")

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
    print(f"   Tham số TF-IDF: {tfidf_params}")
    t0 = time.time()
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_vec = vectorizer.fit_transform(X_train_raw_sampled)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"   Hoàn thành TF-IDF ({time.time() - t0:.1f}s) | Shape Train: {X_train_vec.shape}")


    # 5. Grid Search tìm siêu tham số tốt nhất cho từng mô hình đơn lẻ
    print("\n5. Bắt đầu Grid Search cho từng mô hình...")

    # A. Multinomial Naive Bayes
    nb_param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
    best_nb_params, best_nb_score, _ = grid_search_cv(
        lambda alpha: CustomMultinomialNB(alpha=alpha),
        nb_param_grid,
        X_train_vec, y_train_sampled,
        cv=3, label="MultinomialNB"
    )

    # B. Softmax Regression (Đa lớp bản địa)
    softmax_param_grid = {
        'lr': [0.001, 0.01],
        'epochs': [50, 100]
    }
    best_softmax_params, best_softmax_score, _ = grid_search_cv(
        lambda lr, epochs: CustomSoftmaxRegression(lr=lr, epochs=epochs, C=1.0, class_weight='balanced'),
        softmax_param_grid,
        X_train_vec, y_train_sampled,
        cv=3, label="SoftmaxRegression"
    )

    # C. Linear SVM (OVR)
    svm_param_grid = {
        'lambda_param': [0.001, 0.01, 0.1],
        'epochs': [50, 100]
    }
    best_svm_params, best_svm_score, _ = grid_search_cv(
        lambda lambda_param, epochs: CustomOneVsRestClassifier(
            CustomLinearSVM(lr=0.01, lambda_param=lambda_param, epochs=epochs, class_weight='balanced')
        ),
        svm_param_grid,
        X_train_vec, y_train_sampled,
        cv=3, label="LinearSVM OVR"
    )

    print("\n" + "=" * 70)
    print(" KẾT QUẢ GRID SEARCH (4K CAPPING):")
    print(f"  - Best NB      : alpha={best_nb_params['alpha']} (CV F1={best_nb_score:.4f})")
    print(f"  - Best Softmax : lr={best_softmax_params['lr']}, epochs={best_softmax_params['epochs']} (CV F1={best_softmax_score:.4f})")
    print(f"  - Best SVM OVR : lambda={best_svm_params['lambda_param']}, epochs={best_svm_params['epochs']} (CV F1={best_svm_score:.4f})")
    print("=" * 70)

    # Lưu params tốt nhất ra JSON để train_optimal đọc lại tự động (bản 4k)
    import json
    os.makedirs("models", exist_ok=True)
    best_params_all = {
        'tfidf': tfidf_params,
        'nb':      best_nb_params,
        'softmax': best_softmax_params,
        'svm':     best_svm_params,
        'cv_scores': {
            'nb':      best_nb_score,
            'softmax': best_softmax_score,
            'svm':     best_svm_score,
        }
    }
    params_path = "models/best_params_4k.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2, ensure_ascii=False)
    print(f"\n   ✅ Params tốt nhất đã được lưu tại: {params_path}\n")

    # 6. Huấn luyện các mô hình đơn lẻ tối ưu trên toàn bộ tập Train (đã undersample)
    print("\n6. Huấn luyện và đánh giá từng mô hình đơn lẻ tối ưu...")
    
    # 6.1. Multinomial Naive Bayes
    print("   -> Đang huấn luyện MultinomialNB...")
    t0 = time.time()
    nb_final = CustomMultinomialNB(**best_nb_params)
    nb_final.fit(X_train_vec, y_train_sampled)
    y_pred_nb = nb_final.predict(X_test_vec)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='macro', zero_division=0)
    print(f"      Xong MultinomialNB ({time.time() - t0:.1f}s) | Test Acc: {acc_nb:.4%}, Test Macro F1: {f1_nb:.4%}")

    # 6.2. Softmax Regression (Đa lớp bản địa)
    print("   -> Đang huấn luyện CustomSoftmaxRegression...")
    t0 = time.time()
    softmax_final = CustomSoftmaxRegression(C=1.0, class_weight='balanced', verbose=True, **best_softmax_params)
    softmax_final.fit(X_train_vec, y_train_sampled)
    y_pred_softmax = softmax_final.predict(X_test_vec)
    acc_softmax = accuracy_score(y_test, y_pred_softmax)
    f1_softmax = f1_score(y_test, y_pred_softmax, average='macro', zero_division=0)
    print(f"      Xong SoftmaxRegression ({time.time() - t0:.1f}s) | Test Acc: {acc_softmax:.4%}, Test Macro F1: {f1_softmax:.4%}")



    # 6.4. Linear SVM (OVR)
    print("   -> Đang huấn luyện LinearSVM OVR (bao gồm Platt Scaling)...")
    t0 = time.time()
    svm_final = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, class_weight='balanced', **best_svm_params)
    )
    svm_final.fit(X_train_vec, y_train_sampled)
    y_pred_svm = svm_final.predict(X_test_vec)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    print(f"      Xong LinearSVM OVR ({time.time() - t0:.1f}s) | Test Acc: {acc_svm:.4%}, Test Macro F1: {f1_svm:.4%}")

    # 7. Huấn luyện mô hình Soft Voting Ensemble từ các mô hình tối ưu
    print("\n7. Huấn luyện và đánh giá mô hình Soft Voting Ensemble...")
    t0 = time.time()
    
    nb_est = CustomMultinomialNB(**best_nb_params)
    softmax_est = CustomSoftmaxRegression(C=1.0, class_weight='balanced', **best_softmax_params)
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, class_weight='balanced', **best_svm_params)
    )
    
    ensemble = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_est),
        ('SoftmaxRegression', softmax_est),
    ])
    
    print("   -> Đang huấn luyện Ensemble (NB + Softmax)...")
    ensemble.fit(X_train_vec, y_train_sampled)
    y_pred_ens = ensemble.predict(X_test_vec)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    f1_ens = f1_score(y_test, y_pred_ens, average='macro', zero_division=0)
    print(f"      Xong Ensemble ({time.time() - t0:.1f}s) | Test Acc: {acc_ens:.4%}, Test Macro F1: {f1_ens:.4%}")

    # 8. Bảng so sánh và phân tích kết quả chi tiết
    print("\n" + "=" * 70)
    print(" BẢNG SO SÁNH HIỆU NĂNG TRÊN TẬP KIỂM THỬ (TEST SET - 4K CAP)")
    print("=" * 70)
    print(f" {'Mô hình':32s} | {'Accuracy':12s} | {'Macro F1':12s}")
    print("-" * 70)
    print(f" {'Multinomial Naive Bayes':32s} | {acc_nb:11.4%} | {f1_nb:11.4%}")
    print(f" {'Softmax Regression':32s} | {acc_softmax:11.4%} | {f1_softmax:11.4%}")
    print(f" {'Linear SVM OVR (Platt)':32s} | {acc_svm:11.4%} | {f1_svm:11.4%}")
    print(f" {'Soft Voting Ensemble (2 Models)':32s} | {acc_ens:11.4%} | {f1_ens:11.4%}")
    print("=" * 70)

    # In chi tiết Báo cáo phân loại của từng mô hình
    print("\n>>> CHI TIẾT CLASSIFICATION REPORT TỪNG MÔ HÌNH <<<\n")
    
    print("--- Multinomial Naive Bayes ---")
    print(classification_report(y_test, y_pred_nb, zero_division=0))
    
    print("--- Softmax Regression ---")
    print(classification_report(y_test, y_pred_softmax, zero_division=0))
    

    
    print("--- Linear SVM OVR ---")
    print(classification_report(y_test, y_pred_svm, zero_division=0))
    
    print("--- Soft Voting Ensemble ---")
    print(classification_report(y_test, y_pred_ens, zero_division=0))

    # Lưu kết quả thực nghiệm và mô hình Ensemble vào thư mục models/
    print("\n9. Lưu mô hình Ensemble tốt nhất và các tham số tối ưu...")
    model_data = {
        'vectorizer': vectorizer,
        'model': ensemble,
        'best_params': {
            'tfidf': tfidf_params,
            'nb': best_nb_params,
            'softmax': best_softmax_params,
            'svm': best_svm_params,
        },
        'classes': list(np.unique(y)),
        'comparison_metrics': {
            'nb':       {'accuracy': acc_nb,      'macro_f1': f1_nb},
            'softmax':  {'accuracy': acc_softmax,  'macro_f1': f1_softmax},
            'svm':      {'accuracy': acc_svm,      'macro_f1': f1_svm},
            'ensemble': {'accuracy': acc_ens,      'macro_f1': f1_ens}
        }
    }
    output_model_path = "models/comparison_ensemble_model_4k.pkl"
    with open(output_model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"   => Lưu thành công mô hình vào: {output_model_path}")
    print(f"   => Tổng thời gian thực nghiệm: {time.time() - t_start:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
