import os
import re
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

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
    # 1. Đọc và lọc các tập dữ liệu
    huff_path = "data/dataset gốc/News_Category_Dataset_v3_ordered.csv"
    bbc_path = "data/dataset gốc/BBC_dataset.csv"
    edu_new_path = "data/dataset gốc/education_data.csv"
    tech_new_path = "data/dataset gốc/technology_data.csv"
    sports_new_path = "data/dataset gốc/sports_data.csv"
    business_new_path = "data/dataset gốc/business_data.csv"
    
    if not all(os.path.exists(p) for p in [huff_path, bbc_path, edu_new_path, tech_new_path, sports_new_path, business_new_path]):
        print("⚠️ Thiếu file dataset gốc trong data/dataset gốc/!")
        return

    print("📖 Đang đọc dữ liệu...")
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
    df_huff_processed = pd.DataFrame({'text_raw': df_huff['text_raw'], 'category': df_huff['category']})
    
    df_bbc = pd.read_csv(bbc_path, encoding='cp1252')
    bbc_mapping = {'sport': 'Sports', 'business': 'Business', 'politics': 'Politics and society', 'tech': 'Tech & Science', 'entertainment': 'Entertainment'}
    df_bbc['category'] = df_bbc['type'].map(bbc_mapping)
    df_bbc_processed = pd.DataFrame({'text_raw': df_bbc['news'], 'category': df_bbc['category']})

    df_edu = pd.read_csv(edu_new_path)
    df_edu['text_raw'] = df_edu['headlines'].fillna('') + " " + df_edu['description'].fillna('')
    df_edu_processed = pd.DataFrame({'text_raw': df_edu['text_raw'], 'category': 'Education'})

    df_tech = pd.read_csv(tech_new_path)
    df_tech['text_raw'] = df_tech['headlines'].fillna('') + " " + df_tech['description'].fillna('')
    df_tech_processed = pd.DataFrame({'text_raw': df_tech['text_raw'], 'category': 'Tech & Science'})

    df_sports = pd.read_csv(sports_new_path)
    df_sports['text_raw'] = df_sports['headlines'].fillna('') + " " + df_sports['description'].fillna('')
    df_sports_processed = pd.DataFrame({'text_raw': df_sports['text_raw'], 'category': 'Sports'})

    df_business = pd.read_csv(business_new_path)
    df_business['text_raw'] = df_business['headlines'].fillna('') + " " + df_business['description'].fillna('')
    df_business_processed = pd.DataFrame({'text_raw': df_business['text_raw'], 'category': 'Business'})

    merged_raw = pd.concat([df_huff_processed, df_bbc_processed, df_edu_processed, df_tech_processed, df_sports_processed, df_business_processed], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['text_raw', 'category'])
    merged_raw = merged_raw[merged_raw['text_raw'].str.strip() != '']
    
    # Áp dụng capping 25k trước khi chia tập (để tập Train có tối đa 20,000 mẫu/nhãn)
    sampled = []
    for cat, group in merged_raw.groupby('category'):
        n = min(len(group), 25000)
        sampled.append(group.sample(n, random_state=42))
    merged_capped = pd.concat(sampled).reset_index(drop=True)
    
    print("🧹 Đang làm sạch văn bản...")
    merged_capped['cleaned_text'] = merged_capped['text_raw'].apply(clean_text)
    
    # Chia tách tập Train/Test 80/20 phân tầng
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Kích thước tập Train gốc: {len(X_train_raw)} | Tập Test: {len(X_test_raw)}")
    print("Phân bố nhãn trong tập Train gốc:")
    for cls in np.unique(y_train):
        print(f"  - {cls:20s}: {np.sum(y_train == cls)} mẫu")

    # TF-IDF Vectorizer (50k features, ngram 1-3)
    print("\nTrích xuất đặc trưng TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    # -------------------------------------------------------------
    # THỬ NGHIỆM 1: BASELINE (Không SMOTE, dùng Class Weight Balanced)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("🔥 THỬ NGHIỆM 1: BASELINE (Không SMOTE)")
    print("=" * 50)
    
    nb_baseline = CustomMultinomialNB(alpha=0.1)
    lr_baseline = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced', beta1=0.9, beta2=0.999)
    )
    svm_baseline = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight='balanced', beta1=0.8, beta2=0.999)
    )
    ensemble_baseline = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_baseline),
        ('LogisticRegression_OVR', lr_baseline),
        ('LinearSVM_OVR', svm_baseline),
    ])
    
    t_start = time.time()
    ensemble_baseline.fit(X_train_vec, y_train)
    t_baseline = time.time() - t_start
    
    y_pred_bl = ensemble_baseline.predict(X_test_vec)
    acc_bl = accuracy_score(y_test, y_pred_bl)
    f1_bl = f1_score(y_test, y_pred_bl, average='macro', zero_division=0)
    print(f"Baseline - Accuracy: {acc_bl:.4%} | Macro F1: {f1_bl:.4%} | Thời gian: {t_baseline:.1f}s")
    
    # -------------------------------------------------------------
    # THỬ NGHIỆM 2: ÁP DỤNG SMOTE LÊN TẬP TRAIN VEC
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("🔥 THỬ NGHIỆM 2: ÁP DỤNG SMOTE LÊN TẬP TRAIN VEC")
    print("=" * 50)
    
    print("Đang chạy SMOTE...")
    t_smote_start = time.time()
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)
    print(f"SMOTE hoàn thành trong {time.time() - t_smote_start:.1f}s")
    print(f"Kích thước tập Train sau SMOTE: {X_train_smote.shape[0]} mẫu")
    print("Phân bố nhãn trong tập Train sau SMOTE:")
    for cls in np.unique(y_train_smote):
        print(f"  - {cls:20s}: {np.sum(y_train_smote == cls)} mẫu")
        
    # Huấn luyện mô hình trên tập SMOTE (Với class_weight=None vì tập dữ liệu đã cân bằng)
    nb_smote = CustomMultinomialNB(alpha=0.1)
    lr_smote = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight=None, beta1=0.9, beta2=0.999)
    )
    svm_smote = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=100, class_weight=None, beta1=0.8, beta2=0.999)
    )
    ensemble_smote = FlatSoftVotingClassifier(estimators=[
        ('MultinomialNB', nb_smote),
        ('LogisticRegression_OVR', lr_smote),
        ('LinearSVM_OVR', svm_smote),
    ])
    
    t_start = time.time()
    ensemble_smote.fit(X_train_smote, y_train_smote)
    t_smote_train = time.time() - t_start
    
    y_pred_sm = ensemble_smote.predict(X_test_vec)
    acc_sm = accuracy_score(y_test, y_pred_sm)
    f1_sm = f1_score(y_test, y_pred_sm, average='macro', zero_division=0)
    print(f"SMOTE - Accuracy: {acc_sm:.4%} | Macro F1: {f1_sm:.4%} | Thời gian: {t_smote_train:.1f}s")
    
    # 3. Kết luận
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ SO SÁNH")
    print("=" * 50)
    print(f"1. Baseline: Accuracy = {acc_bl:.4%}, Macro F1 = {f1_bl:.4%}")
    print(f"2. SMOTE   : Accuracy = {acc_sm:.4%}, Macro F1 = {f1_sm:.4%}")
    diff_acc = acc_sm - acc_bl
    diff_f1 = f1_sm - f1_bl
    print(f"Chênh lệch Accuracy: {diff_acc:+.4%}")
    print(f"Chênh lệch Macro F1: {diff_f1:+.4%}")
    print("=" * 50)
    
    # In thêm báo cáo chi tiết của SMOTE để so sánh
    print("\nClassification Report của SMOTE:")
    print(classification_report(y_test, y_pred_sm, zero_division=0))

if __name__ == "__main__":
    main()
