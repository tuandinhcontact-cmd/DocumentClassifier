import os
import re
import pandas as pd
import numpy as np
import time
from itertools import product
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.flat_ensemble import CustomOneVsRestClassifier

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
    print(" GRID SEARCH: TÌM THAM SỐ ADAM TỐI ƯU CHO LOGISTIC REGRESSION VÀ LINEAR SVM")
    print(" CẤU HÌNH: CAPPING 15,000 MẪU/NHÃN | EPOCHS = 100")
    print("=" * 80)

    # 1. Đọc và gộp dữ liệu
    huff_path = "data/dataset gốc/News_Category_Dataset_v3_ordered.csv"
    if not os.path.exists(huff_path):
        print(f"⚠️ Không tìm thấy file HuffPost!")
        return
        
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
    
    print("   Làm sạch văn bản...")
    merged_capped['cleaned_text'] = merged_capped['text_raw'].apply(clean_text)
    
    # Split
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Kích thước tập Train CV: {len(X_train_raw)}")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    
    # Định nghĩa không gian tìm kiếm tham số Adam (6 combinations)
    adam_param_grid = {
        'beta1': [0.80, 0.90, 0.95],
        'beta2': [0.99, 0.999]
    }
    
    keys = list(adam_param_grid.keys())
    combinations = list(product(*adam_param_grid.values()))
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # ─── 1. Grid Search Adam cho Logistic Regression ───
    print("\n" + "=" * 60)
    print(" 1. GRID SEARCH ADAM CHO LOGISTIC REGRESSION OVR")
    print("=" * 60)
    
    best_lr_score = -1
    best_lr_params = None
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        fold_scores = []
        
        t_start = time.time()
        for train_idx, val_idx in skf.split(X_train_vec, y_train):
            X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Huấn luyện LR với 100 epochs
            model = CustomOneVsRestClassifier(
                CustomLogisticRegression(
                    solver='adam', lr=0.01, epochs=100, class_weight='balanced',
                    beta1=params['beta1'], beta2=params['beta2'], eps=1e-8
                )
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            
        mean_score = np.mean(fold_scores)
        print(f"   [beta1={params['beta1']}, beta2={params['beta2']}] ➔ CV Macro F1 = {mean_score:.4%} ({time.time() - t_start:.1f}s)")
        
        if mean_score > best_lr_score:
            best_lr_score = mean_score
            best_lr_params = params
            
    print(f"\n   ✅ Best Adam params for LR: {best_lr_params} (CV F1 = {best_lr_score:.4%})")

    # ─── 2. Grid Search Adam cho Linear SVM ───
    print("\n" + "=" * 60)
    print(" 2. GRID SEARCH ADAM CHO LINEAR SVM OVR")
    print("=" * 60)
    
    best_svm_score = -1
    best_svm_params = None
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        fold_scores = []
        
        t_start = time.time()
        for train_idx, val_idx in skf.split(X_train_vec, y_train):
            X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Huấn luyện SVM với 100 epochs
            model = CustomOneVsRestClassifier(
                CustomLinearSVM(
                    lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced',
                    beta1=params['beta1'], beta2=params['beta2'], eps=1e-8
                )
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            
        mean_score = np.mean(fold_scores)
        print(f"   [beta1={params['beta1']}, beta2={params['beta2']}] ➔ CV Macro F1 = {mean_score:.4%} ({time.time() - t_start:.1f}s)")
        
        if mean_score > best_svm_score:
            best_svm_score = mean_score
            best_svm_params = params
            
    print(f"\n   ✅ Best Adam params for SVM: {best_svm_params} (CV F1 = {best_svm_score:.4%})")

if __name__ == "__main__":
    main()
