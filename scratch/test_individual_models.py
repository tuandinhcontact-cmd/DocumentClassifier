import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

# Import optimized custom models
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.ensemble import CustomOneVsRestClassifier

def main():
    dataset_path = "data/merged_cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    X_raw = df['cleaned_text']
    y = df['category']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing with max_features=20000...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 1. MultinomialNB
    print("\nTraining MultinomialNB...")
    mnb = CustomMultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    print(f"MultinomialNB Accuracy: {accuracy_score(y_test, y_pred_mnb):.4%} | Macro F1: {f1_score(y_test, y_pred_mnb, average='macro'):.4%}")
    
    # 2. Logistic Regression
    print("\nTraining Logistic Regression OVR...")
    lr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced'))
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"LogisticRegression OVR Accuracy: {accuracy_score(y_test, y_pred_lr):.4%} | Macro F1: {f1_score(y_test, y_pred_lr, average='macro'):.4%}")
    
    # 3. Linear SVM
    print("\nTraining Linear SVM OVR...")
    svm = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight='balanced'))
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"LinearSVM OVR Accuracy: {accuracy_score(y_test, y_pred_svm):.4%} | Macro F1: {f1_score(y_test, y_pred_svm, average='macro'):.4%}")

if __name__ == "__main__":
    main()
