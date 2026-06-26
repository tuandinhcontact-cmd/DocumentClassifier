import os
import re
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec

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

def get_document_vectors(w2v_model, tokenized_texts):
    vectors = []
    vector_size = w2v_model.vector_size
    for words in tokenized_texts:
        valid_words = [w for w in words if w in w2v_model.wv]
        if len(valid_words) == 0:
            vectors.append(np.zeros(vector_size))
        else:
            vectors.append(np.mean(w2v_model.wv[valid_words], axis=0))
    return np.array(vectors)

def main():
    print("=" * 80)
    print(" THỬ NGHIỆM ĐẶC TRƯNG WORD2VEC VS TF-IDF CHO PHÂN LOẠI TIN TỨC")
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
    
    # 2. Huấn luyện Word2Vec từ đầu (vector_size = 100)
    print("\n🧠 Đang huấn luyện Word2Vec trực tiếp trên tập Train...")
    X_train_tokens = [text.split() for text in X_train_raw]
    X_test_tokens = [text.split() for text in X_test_raw]
    
    t0 = time.time()
    w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
    print(f"Huấn luyện Word2Vec xong trong {time.time() - t0:.1f}s")
    
    # Chuyển đổi văn bản sang vector trung bình cộng (Mean Word2Vec)
    print("Đang tạo vector tài liệu từ Word2Vec...")
    X_train_w2v = get_document_vectors(w2v_model, X_train_tokens)
    X_test_w2v = get_document_vectors(w2v_model, X_test_tokens)
    
    # 3. Trích xuất TF-IDF (để làm đối chứng)
    print("Đang trích xuất đặc trưng TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    # 4. Huấn luyện và Đánh giá Logistic Regression OVR
    print("\n" + "-" * 50)
    print(" Huấn luyện LOGISTIC REGRESSION (OVR) (Epochs = 50)")
    print("-" * 50)
    
    # A. Trên đặc trưng TF-IDF (Baseline)
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
    print(f"LR + TF-IDF   -> Accuracy: {acc_lr_tfidf:.4%} | Macro F1: {f1_lr_tfidf:.4%} | Time: {t_train_tfidf:.1f}s")
    
    # B. Trên đặc trưng Word2Vec
    print("B. Đang train LR trên Word2Vec...")
    lr_w2v = CustomOneVsRestClassifier(
        CustomLogisticRegression(solver='adam', lr=0.01, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    lr_w2v.fit(X_train_w2v, y_train)
    t_train_w2v = time.time() - t_start
    y_pred_lr_w2v = lr_w2v.predict(X_test_w2v)
    acc_lr_w2v = accuracy_score(y_test, y_pred_lr_w2v)
    f1_lr_w2v = f1_score(y_test, y_pred_lr_w2v, average='macro', zero_division=0)
    print(f"LR + Word2Vec -> Accuracy: {acc_lr_w2v:.4%} | Macro F1: {f1_lr_w2v:.4%} | Time: {t_train_w2v:.1f}s")
    
    # 5. Huấn luyện và Đánh giá SVM OVR
    print("\n" + "-" * 50)
    print(" Huấn luyện LINEAR SVM (OVR) (Epochs = 50)")
    print("-" * 50)
    
    # A. Trên đặc trưng TF-IDF (Baseline)
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
    print(f"SVM + TF-IDF   -> Accuracy: {acc_svm_tfidf:.4%} | Macro F1: {f1_svm_tfidf:.4%} | Time: {t_train_svm_tfidf:.1f}s")
    
    # B. Trên đặc trưng Word2Vec
    print("B. Đang train SVM trên Word2Vec...")
    svm_w2v = CustomOneVsRestClassifier(
        CustomLinearSVM(lr=0.01, lambda_param=0.001, epochs=50, class_weight='balanced')
    )
    t_start = time.time()
    svm_w2v.fit(X_train_w2v, y_train)
    t_train_svm_w2v = time.time() - t_start
    y_pred_svm_w2v = svm_w2v.predict(X_test_w2v)
    acc_svm_w2v = accuracy_score(y_test, y_pred_svm_w2v)
    f1_svm_w2v = f1_score(y_test, y_pred_svm_w2v, average='macro', zero_division=0)
    print(f"SVM + Word2Vec -> Accuracy: {acc_svm_w2v:.4%} | Macro F1: {f1_svm_w2v:.4%} | Time: {t_train_svm_w2v:.1f}s")

    # 6. Tổng kết
    print("\n" + "=" * 50)
    print("📊 BẢNG TỔNG HỢP SO SÁNH CUỐI CÙNG")
    print("=" * 50)
    print(f"1. LR  + TF-IDF  : Accuracy = {acc_lr_tfidf:.4%}, Macro F1 = {f1_lr_tfidf:.4%}")
    print(f"2. LR  + Word2Vec: Accuracy = {acc_lr_w2v:.4%}, Macro F1 = {f1_lr_w2v:.4%}")
    print(f"3. SVM + TF-IDF  : Accuracy = {acc_svm_tfidf:.4%}, Macro F1 = {f1_svm_tfidf:.4%}")
    print(f"4. SVM + Word2Vec: Accuracy = {acc_svm_w2v:.4%}, Macro F1 = {f1_svm_w2v:.4%}")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
