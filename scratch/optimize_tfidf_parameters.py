import os
import re
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from custom_models.logistic_regression import CustomLogisticRegression
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

def evaluate_with_tfidf_config(tfidf_name, tfidf_params):
    print(f"\n--- ĐANG CHẠY THỬ NGHIỆM VỚI TF-IDF: {tfidf_name} ---")
    
    # 1. Đọc dữ liệu
    df_huff = pd.read_csv("data/dataset gốc/News_Category_Dataset_v3_ordered.csv")
    category_mapping = {
        "POLITICS": "Politics and society", "IMPACT": "Politics and society",
        "TRAVEL": "Lifestyle", "STYLE & BEAUTY": "Lifestyle", "HOME & LIVING": "Lifestyle",
        "STYLE": "Lifestyle", "FIFTY": "Lifestyle", "WELLNESS": "Health", "HEALTHY LIVING": "Health",
        "ENTERTAINMENT": "Entertainment", "COMEDY": "Entertainment", "MEDIA": "Entertainment",
        "FOOD & DRINK": "Food & drinks", "TASTE": "Food & drinks", "BUSINESS": "Business",
        "MONEY": "Business", "PARENTING": "Family", "PARENTS": "Family", "WEDDINGS": "Family",
        "DIVORCE": "Family", "QUEER VOICES": "Community", "BLACK VOICES": "Community",
        "LATINO VOICES": "Community", "WOMEN": "Community", "RELIGION": "Community",
        "THE WORLDPOST": "Politics and society", "WORLDPOST": "Politics and society",
        "WORLD NEWS": "Politics and society", "CRIME": "Politics and society",
        "WEIRD NEWS": "Entertainment", "GOOD NEWS": "Lifestyle", "SPORTS": "Sports",
        "TECH": "Tech & Science", "SCIENCE": "Tech & Science", "ENVIRONMENT": "Environment",
        "GREEN": "Environment", "ARTS": "Arts & Culture", "ARTS & CULTURE": "Arts & Culture",
        "CULTURE & ARTS": "Arts & Culture", "EDUCATION": "Education", "COLLEGE": "Education",
    }
    df_huff['category'] = df_huff['category'].map(category_mapping)
    df_huff = df_huff.dropna(subset=['category'])
    df_huff['text_raw'] = df_huff['headline'].fillna('') + " " + df_huff['short_description'].fillna('')
    df_huff_processed = pd.DataFrame({
        'text_raw': df_huff['text_raw'],
        'category': df_huff['category']
    })
    
    df_bbc = pd.read_csv("data/dataset gốc/BBC_dataset.csv", encoding='cp1252')
    bbc_mapping = {
        'sport': 'Sports', 'business': 'Business', 'politics': 'Politics and society',
        'tech': 'Tech & Science', 'entertainment': 'Entertainment'
    }
    df_bbc['category'] = df_bbc['type'].map(bbc_mapping)
    df_bbc_processed = pd.DataFrame({
        'text_raw': df_bbc['news'],
        'category': df_bbc['category']
    })

    df_edu = pd.read_csv("/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/education_data.csv")
    df_edu['text_raw'] = df_edu['headlines'].fillna('') + " " + df_edu['description'].fillna('')
    df_edu_processed = pd.DataFrame({
        'text_raw': df_edu['text_raw'],
        'category': 'Education'
    })

    df_tech = pd.read_csv("/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/technology_data.csv")
    df_tech['text_raw'] = df_tech['headlines'].fillna('') + " " + df_tech['description'].fillna('')
    df_tech_processed = pd.DataFrame({
        'text_raw': df_tech['text_raw'],
        'category': 'Tech & Science'
    })

    merged_raw = pd.concat([df_huff_processed, df_bbc_processed, df_edu_processed, df_tech_processed], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['text_raw', 'category'])
    merged_raw = merged_raw[merged_raw['text_raw'].str.strip() != '']
    
    # Capping pre-split ở mức 15,000 mẫu/nhãn
    sampled = []
    for cat, group in merged_raw.groupby('category'):
        n = min(len(group), 15000)
        sampled.append(group.sample(n, random_state=42))
    merged_capped = pd.concat(sampled).reset_index(drop=True)
    
    merged_capped['cleaned_text'] = merged_capped['text_raw'].apply(clean_text)
    
    # Split 80/20
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF
    print("   Trích xuất đặc trưng TF-IDF...")
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"   Số lượng đặc trưng trích xuất: {X_train_vec.shape[1]}")
    
    # Huấn luyện các mô hình thành phần
    print("   Huấn luyện MultinomialNB...")
    nb = CustomMultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='macro', zero_division=0)
    print(f"      => NB: Acc={acc_nb:.4%}, F1={f1_nb:.4%}")
    
    print("   Huấn luyện Logistic Regression...")
    lr = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced', beta1=0.9, beta2=0.999)
    )
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='macro', zero_division=0)
    print(f"      => LR: Acc={acc_lr:.4%}, F1={f1_lr:.4%}")
    
    print("   Huấn luyện Linear SVM...")
    svm = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    print(f"      => SVM: Acc={acc_svm:.4%}, F1={f1_svm:.4%}")
    
    print("   Huấn luyện Soft Voting Ensemble...")
    nb_est = CustomMultinomialNB(alpha=0.1)
    lr_est = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced', beta1=0.9, beta2=0.999)
    )
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    ensemble = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_est),
        ('LogisticRegression_OVR', lr_est),
        ('LinearSVM_OVR', svm_est),
    ])
    ensemble.fit(X_train_vec, y_train)
    y_pred_ens = ensemble.predict(X_test_vec)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    f1_ens = f1_score(y_test, y_pred_ens, average='macro', zero_division=0)
    print(f"      => Ensemble: Acc={acc_ens:.4%}, F1={f1_ens:.4%}")
    
    return {
        'nb': {'acc': acc_nb, 'f1': f1_nb},
        'lr': {'acc': acc_lr, 'f1': f1_lr},
        'svm': {'acc': acc_svm, 'f1': f1_svm},
        'ensemble': {'acc': acc_ens, 'f1': f1_ens}
    }

def main():
    # Cấu hình baseline
    baseline_params = {
        'max_features': 30000,
        'ngram_range': (1, 2),
        'sublinear_tf': False
    }
    
    # Cấu hình tối ưu đề xuất
    optimized_params = {
        'max_features': 50000,
        'ngram_range': (1, 3),
        'sublinear_tf': True,
        'min_df': 2
    }
    
    res_baseline = evaluate_with_tfidf_config("Baseline (30k, 1-2 ngrams, sublinear=False)", baseline_params)
    res_opt = evaluate_with_tfidf_config("Optimized (50k, 1-3 ngrams, sublinear=True, min_df=2)", optimized_params)
    
    print("\n" + "=" * 80)
    print(" BẢNG SO SÁNH KẾT QUẢ TỐI ƯU HÓA ĐẶC TRƯNG TF-IDF")
    print("=" * 80)
    print(f" {'Mô hình':30s} | {'Baseline Acc / F1':22s} | {'Optimized Acc / F1':22s} | {'Độ lệch Acc / F1':18s}")
    print("-" * 80)
    for model_key, model_name in [('nb', 'Multinomial NB'), ('lr', 'Logistic Regression'), ('svm', 'Linear SVM'), ('ensemble', 'Soft Voting Ensemble')]:
        acc_diff = res_opt[model_key]['acc'] - res_baseline[model_key]['acc']
        f1_diff = res_opt[model_key]['f1'] - res_baseline[model_key]['f1']
        
        baseline_str = f"{res_baseline[model_key]['acc']:6.2%} / {res_baseline[model_key]['f1']:6.2%}"
        opt_str = f"{res_opt[model_key]['acc']:6.2%} / {res_opt[model_key]['f1']:6.2%}"
        diff_str = f"{acc_diff:+.2%} / {f1_diff:+.2%}"
        
        print(f" {model_name:30s} | {baseline_str:22s} | {opt_str:22s} | {diff_str:18s}")
    print("=" * 80)

if __name__ == "__main__":
    main()
