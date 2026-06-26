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
    print(" HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH PHẲNG KHÔNG GIỚI HẠN (UNCAPPED)")
    print(" DỮ LIỆU: HuffPost + BBC + Education + Tech (TỔNG THỂ TỰ NHIÊN)")
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

    # Gộp dữ liệu không giới hạn (Uncapped)
    merged_raw = pd.concat([df_huff_processed, df_bbc_processed, df_edu_processed, df_tech_processed], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['text_raw', 'category'])
    merged_raw = merged_raw[merged_raw['text_raw'].str.strip() != '']
    
    print(f"   Tổng số lượng mẫu ban đầu: {len(merged_raw)}")
    print(merged_raw['category'].value_counts())
    
    print("   Làm sạch văn bản...")
    merged_raw['cleaned_text'] = merged_raw['text_raw'].apply(clean_text)
    
    # Split Train/Test 80/20
    X = merged_raw['cleaned_text'].values
    y = merged_raw['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    # Huấn luyện
    print("   Huấn luyện MultinomialNB...")
    nb = CustomMultinomialNB(alpha=0.1)
    nb.fit(X_train_vec, y_train)
    y_pred_nb = nb.predict(X_test_vec)
    print(f"      => NB: Acc={accuracy_score(y_test, y_pred_nb):.4%}, F1={f1_score(y_test, y_pred_nb, average='macro'):.4%}")
    
    print("   Huấn luyện Logistic Regression...")
    lr = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced')
    )
    lr.fit(X_train_vec, y_train)
    y_pred_lr = lr.predict(X_test_vec)
    print(f"      => LR: Acc={accuracy_score(y_test, y_pred_lr):.4%}, F1={f1_score(y_test, y_pred_lr, average='macro'):.4%}")
    
    print("   Huấn luyện Linear SVM...")
    svm = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    svm.fit(X_train_vec, y_train)
    y_pred_svm = svm.predict(X_test_vec)
    print(f"      => SVM: Acc={accuracy_score(y_test, y_pred_svm):.4%}, F1={f1_score(y_test, y_pred_svm, average='macro'):.4%}")
    
    print("   Huấn luyện Soft Voting Ensemble...")
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
    ensemble.fit(X_train_vec, y_train)
    y_pred_ens = ensemble.predict(X_test_vec)
    acc_ens = accuracy_score(y_test, y_pred_ens)
    f1_ens = f1_score(y_test, y_pred_ens, average='macro')
    
    print("\n" + "=" * 60)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PHẲNG KHÔNG GIỚI HẠN (UNCAPPED)")
    print("=" * 60)
    print(f" Accuracy : {acc_ens:.4%}")
    print(f" Macro F1 : {f1_ens:.4%}")
    print("=" * 60)
    print(classification_report(y_test, y_pred_ens, zero_division=0))
    print("=" * 60)

if __name__ == "__main__":
    main()
