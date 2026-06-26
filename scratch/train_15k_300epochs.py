import os
import re
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
    print(" THỬ NGHIỆM: CAPPING 15K + NEW DATA + 300 EPOCHS + SOFT VOTING ENSEMBLE")
    print("=" * 80)

    # 1. Đọc HuffPost
    huff_path = "data/dataset gốc/News_Category_Dataset_v3_ordered.csv"
    if not os.path.exists(huff_path):
        print(f"⚠️ Không tìm thấy file HuffPost tại {huff_path}!")
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
    
    # 2. Đọc BBC
    bbc_path = "data/dataset gốc/BBC_dataset.csv"
    if not os.path.exists(bbc_path):
        print(f"⚠️ Không tìm thấy file BBC tại {bbc_path}!")
        return
        
    df_bbc = pd.read_csv(bbc_path, encoding='cp1252')
    bbc_mapping = {
        'sport': 'Sports', 'business': 'Business', 'politics': 'Politics and society',
        'tech': 'Tech & Science', 'entertainment': 'Entertainment'
    }
    df_bbc['category'] = df_bbc['type'].map(bbc_mapping)
    df_bbc_processed = pd.DataFrame({
        'text_raw': df_bbc['news'],
        'category': df_bbc['category']
    })

    # 3. Đọc New Education
    edu_new_path = "/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/education_data.csv"
    if not os.path.exists(edu_new_path):
        print(f"⚠️ Không tìm thấy file Education mới tại {edu_new_path}!")
        return
        
    df_edu = pd.read_csv(edu_new_path)
    df_edu['text_raw'] = df_edu['headlines'].fillna('') + " " + df_edu['description'].fillna('')
    df_edu_processed = pd.DataFrame({
        'text_raw': df_edu['text_raw'],
        'category': 'Education'
    })

    # 4. Đọc New Technology
    tech_new_path = "/Users/gtuan/.gemini/antigravity/scratch/DocumentClassifier/data/dataset gốc/technology_data.csv"
    if not os.path.exists(tech_new_path):
        print(f"⚠️ Không tìm thấy file Technology mới tại {tech_new_path}!")
        return
        
    df_tech = pd.read_csv(tech_new_path)
    df_tech['text_raw'] = df_tech['headlines'].fillna('') + " " + df_tech['description'].fillna('')
    df_tech_processed = pd.DataFrame({
        'text_raw': df_tech['text_raw'],
        'category': 'Tech & Science'
    })

    # Gộp dữ liệu
    merged_raw = pd.concat([
        df_huff_processed, 
        df_bbc_processed, 
        df_edu_processed, 
        df_tech_processed
    ], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['text_raw', 'category'])
    merged_raw = merged_raw[merged_raw['text_raw'].str.strip() != '']
    
    # Capping pre-split tại 15,000 mẫu/nhãn
    sampled = []
    for cat, group in merged_raw.groupby('category'):
        n = min(len(group), 15000)
        sampled.append(group.sample(n, random_state=42))
    merged_capped = pd.concat(sampled).reset_index(drop=True)
    
    print("   Đang làm sạch văn bản tập dữ liệu gộp...")
    t_start = time.time()
    merged_capped['cleaned_text'] = merged_capped['text_raw'].apply(clean_text)
    print(f"   Làm sạch xong trong {time.time() - t_start:.1f}s")
    
    # Split Train/Test 80/20
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Kích thước tập Train: {len(X_train_raw)} | Tập Test: {len(X_test_raw)}")
    print("   Phân bố các lớp ở tập Train:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"     - {u:20s}: {c} mẫu")
        
    # TF-IDF Vectorizer
    print("   Trích xuất đặc trưng TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=False)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    # Huấn luyện MultinomialNB
    print("\n1. Huấn luyện MultinomialNB (alpha=0.1)...")
    nb = CustomMultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='macro', zero_division=0)
    print(f"   => MultinomialNB: Acc={acc_nb:.4%}, Macro F1={f1_nb:.4%}")

    # Huấn luyện Logistic Regression OVR (300 Epochs)
    print("\n2. Huấn luyện Logistic Regression OVR (epochs=300)...")
    t0 = time.time()
    lr = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=300, class_weight='balanced')
    )
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='macro', zero_division=0)
    print(f"   => LR OVR: Acc={acc_lr:.4%}, Macro F1={f1_lr:.4%} ({time.time() - t0:.1f}s)")

    # Huấn luyện Linear SVM OVR (300 Epochs)
    print("\n3. Huấn luyện Linear SVM OVR (epochs=300)...")
    t0 = time.time()
    svm = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=300, class_weight='balanced')
    )
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)
    print(f"   => SVM OVR: Acc={acc_svm:.4%}, Macro F1={f1_svm:.4%} ({time.time() - t0:.1f}s)")

    # Huấn luyện Soft Voting Ensemble (300 Epochs)
    print("\n4. Huấn luyện Soft Voting Ensemble (epochs=300 cho LR và SVM)...")
    t0 = time.time()
    nb_est = CustomMultinomialNB(alpha=0.1)
    lr_est = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=300, class_weight='balanced')
    )
    svm_est = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=300, class_weight='balanced')
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
    print(f"   => Ensemble: Acc={acc_ens:.4%}, Macro F1={f1_ens:.4%} ({time.time() - t0:.1f}s)")
    
    print("\n" + "=" * 60)
    print(" BẢNG BÁO CÁO CHI TIẾT CHO SOFT VOTING ENSEMBLE (15K CAP, 300 EPOCHS)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_ens, zero_division=0))
    print("=" * 60)

if __name__ == "__main__":
    main()
