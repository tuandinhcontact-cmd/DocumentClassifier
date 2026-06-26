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

def run_flat_ensemble(cap_value=None):
    # 1. Đọc HuffPost
    huff_path = "data/dataset gốc/News_Category_Dataset_v3_ordered.csv"
    df_huff = pd.read_csv(huff_path)
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

    # Gộp
    merged_raw = pd.concat([df_huff_processed, df_bbc_processed, df_edu_processed, df_tech_processed], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['text_raw', 'category'])
    merged_raw = merged_raw[merged_raw['text_raw'].str.strip() != '']
    
    # Capping pre-split (nếu có cấu hình)
    if cap_value is not None:
        sampled = []
        for cat, group in merged_raw.groupby('category'):
            n = min(len(group), cap_value)
            sampled.append(group.sample(n, random_state=42))
        df_final = pd.concat(sampled).reset_index(drop=True)
    else:
        df_final = merged_raw
        
    df_final['cleaned_text'] = df_final['text_raw'].apply(clean_text)
    
    # Train/Test 80/20
    X = df_final['cleaned_text'].values
    y = df_final['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    # Train Ensemble (100 epochs cho tốc độ nhanh)
    nb_est = CustomMultinomialNB(alpha=0.1)
    lr_est = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced')
    )
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    ensemble = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_est),
        ('LogisticRegression_OVR', lr_est),
        ('LinearSVM_OVR', svm_est),
    ])
    
    t_start = time.time()
    ensemble.fit(X_train_vec, y_train)
    y_pred = ensemble.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    return acc, f1, len(df_final), time.time() - t_start

def main():
    print("=" * 80)
    print(" BẮT ĐẦU THỬ NGHIỆM: SO SÁNH CÁC MỨC CAPPING ĐỂ TỐI ƯU HÓA ACCURACY")
    print("=" * 80)
    
    configs = [
        ("Capping 11k (Hiện tại)", 11000),
        ("Capping 20k", 20000),
        ("Capping 30k", 30000),
        ("Uncapped (Không giới hạn)", None)
    ]
    
    results = []
    for name, cap in configs:
        print(f"\n🚀 Đang chạy cấu hình: {name}...")
        acc, f1, total_samples, t_run = run_flat_ensemble(cap)
        print(f"   => Kích thước dataset: {total_samples} dòng")
        print(f"   => Accuracy: {acc:.4%}")
        print(f"   => Macro F1: {f1:.4%}")
        print(f"   => Thời gian huấn luyện: {t_run:.1f}s")
        results.append((name, acc, f1, total_samples))
        
    print("\n" + "=" * 80)
    print(" KẾT QUẢ SO SÁNH TOÀN DIỆN CÁC CHIẾN LƯỢC CAPPING")
    print("=" * 80)
    print(f" {'Cấu hình':30s} | {'Mẫu':10s} | {'Accuracy':12s} | {'Macro F1':12s}")
    print("-" * 80)
    for name, acc, f1, total in results:
        print(f" {name:30s} | {total:<10d} | {acc:11.4%} | {f1:11.4%}")
    print("=" * 80)

if __name__ == "__main__":
    main()
