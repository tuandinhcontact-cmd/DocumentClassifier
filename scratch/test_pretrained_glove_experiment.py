import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize
import gensim.downloader as api

from custom_models.logistic_regression import CustomLogisticRegression
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
    print(" THỬ NGHIỆM: PRE-TRAINED GLOVE (100D) + TRỌNG SỐ TF-IDF")
    print("=" * 80)
    
    # 1. Đọc dữ liệu
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
    
    # Capping tại 20,000 mẫu
    sampled = []
    for cat, group in merged_raw.groupby('category'):
        n = min(len(group), 20000)
        sampled.append(group.sample(n, random_state=42))
    merged_capped = pd.concat(sampled).reset_index(drop=True)
    
    print("🧹 Đang làm sạch văn bản...")
    merged_capped['cleaned_text'] = merged_capped['text_raw'].apply(clean_text)
    
    # Split
    X = merged_capped['cleaned_text'].values
    y = merged_capped['category'].values
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Kích thước tập Train: {len(X_train_raw)} | Tập Test: {len(X_test_raw)}")
    
    # 2. Tải mô hình GloVe 100D học sẵn
    print("\n⬇️ Đang tải mô hình GloVe 100D (glove-wiki-gigaword-100)...")
    t_dl = time.time()
    glove_model = api.load("glove-wiki-gigaword-100")
    print(f"Tải mô hình GloVe hoàn thành trong {time.time() - t_dl:.1f}s")
    
    # 3. Trích xuất đặc trưng TF-IDF làm trọng số
    print("\nTrích xuất đặc trưng TF-IDF (N-gram = 1)...")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 1), sublinear_tf=True, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    # Xây dựng ma trận trọng số GloVe cho bộ từ vựng
    print("Xây dựng ma trận biểu diễn GloVe cho bộ từ vựng...")
    vocab = vectorizer.vocabulary_
    num_words = len(vocab)
    vector_size = glove_model.vector_size
    
    # Ma trận V kích thước (num_words, vector_size)
    V = np.zeros((num_words, vector_size))
    words_found = 0
    
    # Sắp xếp từ vựng theo index của cột TF-IDF
    vocab_sorted = sorted(vocab.items(), key=lambda item: item[1])
    for word, idx in vocab_sorted:
        if word in glove_model:
            V[idx] = glove_model[word]
            words_found += 1
            
    print(f"Số lượng từ khóa tìm thấy trong từ điển GloVe: {words_found}/{num_words} ({words_found/num_words:.2%})")
    
    # Nhân ma trận để tạo vector văn bản có trọng số: X_doc = X_tfidf * V
    print("Tạo vector văn bản có trọng số TF-IDF...")
    X_train_glove = X_train_tfidf.dot(V)
    X_test_glove = X_test_tfidf.dot(V)
    
    # Chuẩn hóa L2 cho các hàng ma trận
    print("Chuẩn hóa L2 vector tài liệu...")
    X_train_glove = normalize(X_train_glove, norm='l2')
    X_test_glove = normalize(X_test_glove, norm='l2')
    
    # 4. Huấn luyện và Đánh giá Logistic Regression OVR
    print("\n" + "-" * 50)
    print(" Huấn luyện LOGISTIC REGRESSION (OVR) (Epochs = 50)")
    print("-" * 50)
    
    # A. Trên đặc trưng TF-IDF (đối chứng)
    print("A. Đang train LR trên TF-IDF...")
    lr_tfidf = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    lr_tfidf.fit(X_train_tfidf, y_train)
    t_train_tfidf = time.time() - t_start
    y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
    acc_lr_tfidf = accuracy_score(y_test, y_pred_lr_tfidf)
    f1_lr_tfidf = f1_score(y_test, y_pred_lr_tfidf, average='macro', zero_division=0)
    print(f"LR + TF-IDF (1-gram)   -> Accuracy: {acc_lr_tfidf:.4%} | Macro F1: {f1_lr_tfidf:.4%} | Time: {t_train_tfidf:.1f}s")
    
    # B. Trên đặc trưng GloVe + TF-IDF weighted
    print("B. Đang train LR trên GloVe + TF-IDF Weighted...")
    lr_glove = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    lr_glove.fit(X_train_glove, y_train)
    t_train_glove = time.time() - t_start
    y_pred_lr_glove = lr_glove.predict(X_test_glove)
    acc_lr_glove = accuracy_score(y_test, y_pred_lr_glove)
    f1_lr_glove = f1_score(y_test, y_pred_lr_glove, average='macro', zero_division=0)
    print(f"LR + GloVe Weighted   -> Accuracy: {acc_lr_glove:.4%} | Macro F1: {f1_lr_glove:.4%} | Time: {t_train_glove:.1f}s")
    
    # 5. Huấn luyện và Đánh giá SVM OVR
    print("\n" + "-" * 50)
    print(" Huấn luyện LINEAR SVM (OVR) (Epochs = 50)")
    print("-" * 50)
    
    # A. Trên đặc trưng TF-IDF (đối chứng)
    print("A. Đang train SVM trên TF-IDF...")
    svm_tfidf = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    svm_tfidf.fit(X_train_tfidf, y_train)
    t_train_svm_tfidf = time.time() - t_start
    y_pred_svm_tfidf = svm_tfidf.predict(X_test_tfidf)
    acc_svm_tfidf = accuracy_score(y_test, y_pred_svm_tfidf)
    f1_svm_tfidf = f1_score(y_test, y_pred_svm_tfidf, average='macro', zero_division=0)
    print(f"SVM + TF-IDF (1-gram)  -> Accuracy: {acc_svm_tfidf:.4%} | Macro F1: {f1_svm_tfidf:.4%} | Time: {t_train_svm_tfidf:.1f}s")
    
    # B. Trên đặc trưng GloVe + TF-IDF weighted
    print("B. Đang train SVM trên GloVe + TF-IDF Weighted...")
    svm_glove = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    svm_glove.fit(X_train_glove, y_train)
    t_train_svm_glove = time.time() - t_start
    y_pred_svm_glove = svm_glove.predict(X_test_glove)
    acc_svm_glove = accuracy_score(y_test, y_pred_svm_glove)
    f1_svm_glove = f1_score(y_test, y_pred_svm_glove, average='macro', zero_division=0)
    print(f"SVM + GloVe Weighted  -> Accuracy: {acc_svm_glove:.4%} | Macro F1: {f1_svm_glove:.4%} | Time: {t_train_svm_glove:.1f}s")

    # 6. Tổng kết
    print("\n" + "=" * 50)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CUỐI CÙNG")
    print("=" * 50)
    print(f"1. LR  + TF-IDF (1-gram)    : Accuracy = {acc_lr_tfidf:.4%}, Macro F1 = {f1_lr_tfidf:.4%}")
    print(f"2. LR  + GloVe (TF-IDF W)   : Accuracy = {acc_lr_glove:.4%}, Macro F1 = {f1_lr_glove:.4%}")
    print(f"3. SVM + TF-IDF (1-gram)    : Accuracy = {acc_svm_tfidf:.4%}, Macro F1 = {f1_svm_tfidf:.4%}")
    print(f"4. SVM + GloVe (TF-IDF W)   : Accuracy = {acc_svm_glove:.4%}, Macro F1 = {f1_svm_glove:.4%}")
    print("=" * 50)

if __name__ == "__main__":
    main()
