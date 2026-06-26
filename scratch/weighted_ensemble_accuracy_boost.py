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

def main():
    print("=" * 80)
    print(" HUẤN LUYỆN & ĐÁNH GIÁ TRỌNG SỐ SOFT VOTING (OPTIMIZED TF-IDF + 15K CAP)")
    print("=" * 80)

    # 1. Đọc HuffPost
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
    
    # BBC
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

    # New Education
    df_edu = pd.read_csv("/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/education_data.csv")
    df_edu['text_raw'] = df_edu['headlines'].fillna('') + " " + df_edu['description'].fillna('')
    df_edu_processed = pd.DataFrame({
        'text_raw': df_edu['text_raw'],
        'category': 'Education'
    })

    # New Technology
    df_tech = pd.read_csv("/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/technology_data.csv")
    df_tech['text_raw'] = df_tech['headlines'].fillna('') + " " + df_tech['description'].fillna('')
    df_tech_processed = pd.DataFrame({
        'text_raw': df_tech['text_raw'],
        'category': 'Tech & Science'
    })

    # Gộp dữ liệu
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
    
    # Split
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF với Optimized params
    print("   Trích xuất đặc trưng với cấu hình TF-IDF tối ưu...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    # Fit các mô hình thành phần
    print("   Huấn luyện các mô hình thành phần...")
    nb_est = CustomMultinomialNB(alpha=0.1)
    lr_est = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced', beta1=0.9, beta2=0.999)
    )
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    
    nb_est.fit(X_train_vec, y_train)
    lr_est.fit(X_train_vec, y_train)
    svm_est.fit(X_train_vec, y_train)
    
    # Đánh giá các trọng số biểu quyết khác nhau
    weight_configs = [
        ("Equal Weights [1:1:1]", [1.0, 1.0, 1.0]),
        ("Weighted [5:2:3] (NB ưu tiên, LR thấp nhất)", [5.0, 2.0, 3.0]),
        ("Weighted [6:1:3] (NB cực mạnh, LR hỗ trợ nhẹ)", [6.0, 1.0, 3.0]),
        ("Weighted [7:0:3] (Bỏ hẳn LR, chỉ lấy NB + SVM)", [7.0, 0.0, 3.0]),
        ("Weighted [8:0:2] (NB lấn át, SVM hỗ trợ nhẹ)", [8.0, 0.0, 2.0]),
        ("NB Only (Trọng số 1:0:0)", [1.0, 0.0, 0.0])
    ]
    
    print("\n" + "=" * 60)
    print(" KẾT QUẢ ĐÁNH GIÁ CÁC TRỌNG SỐ BIỂU QUYẾT ENSEMBLE")
    print("=" * 60)
    print(f" {'Cấu hình trọng số (NB : LR : SVM)':45s} | {'Accuracy':10s} | {'Macro F1':10s}")
    print("-" * 75)
    
    # Lưu kết quả dự báo xác suất của từng mô hình
    probs_nb = nb_est.predict_proba(X_test_vec)
    probs_lr = lr_est.predict_proba(X_test_vec)
    probs_svm = svm_est.predict_proba(X_test_vec)
    probs_list = [probs_nb, probs_lr, probs_svm]
    
    classes = nb_est.classes
    
    for name, w in weight_configs:
        w_arr = np.array(w)
        w_norm = w_arr / np.sum(w_arr)
        
        # Tính weighted probability
        weighted_probs = np.zeros_like(probs_nb)
        for weight, probs in zip(w_norm, probs_list):
            weighted_probs += weight * probs
            
        y_pred = classes[np.argmax(weighted_probs, axis=1)]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f" {name:45s} | {acc:9.4%} | {f1:9.4%}")
    print("=" * 60)

if __name__ == "__main__":
    main()
