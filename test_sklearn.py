import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def main():
    print("=== Thử nghiệm Scikit-Learn Models (C++ Backend) ===")
    
    # 1. Đọc dữ liệu
    df = pd.read_csv("merged_cleaned_dataset.csv")
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    X = df['cleaned_text']
    y = df['category']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Vector hóa
    print("Đang Vector hóa TF-IDF (20,000 features)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # Danh sách các mô hình Scikit-learn mạnh nhất
    models = {
        "Sklearn MultinomialNB": MultinomialNB(alpha=1.0),
        "Sklearn Logistic Regression (OVR, Balanced)": LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1),
        "Sklearn LinearSVC (Balanced)": LinearSVC(class_weight='balanced', max_iter=2000, dual=False)
    }
    
    print("\nTiến hành huấn luyện phân loại phẳng (Flat Classification) trên 14 nhãn cùng lúc:")
    
    for name, model in models.items():
        start_time = time.time()
        print(f"\n-> Đang train {name}...")
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"   Thời gian train: {train_time:.2f} giây")
        print(f"   Accuracy: {acc:.4%}")
        print(f"   Macro F1: {f1:.4%}")

if __name__ == "__main__":
    main()
