import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import optimized custom models
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.ensemble import CustomOneVsRestClassifier, CustomMultiClassVotingClassifier

def test_config(max_features, class_weight, name_tag):
    print(f"\n--- Testing Config: max_features={max_features}, class_weight={class_weight} ({name_tag}) ---")
    dataset_path = "data/merged_cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    X_raw = df['cleaned_text']
    y = df['category']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print("Initializing estimators...")
    lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=80, class_weight=class_weight))
    svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=80, class_weight=class_weight))
    mnb_mc = CustomMultinomialNB(alpha=1.0)
    
    flat_ensemble = CustomMultiClassVotingClassifier(estimators=[
        ('MultinomialNB', mnb_mc),
        ('LogisticRegression_OVR', lr_ovr),
        ('LinearSVM_OVR', svm_ovr)
    ])
    
    start_time = time.time()
    flat_ensemble.fit(X_train, y_train)
    y_pred = flat_ensemble.predict(X_test)
    elapsed = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"[{name_tag}] Time: {elapsed:.2f}s | Test Accuracy: {acc:.4%} | Macro F1: {macro_f1:.4%}")

def main():
    # Test 1: max_features=40000, class_weight='balanced'
    test_config(max_features=40000, class_weight='balanced', name_tag="40k features + Balanced")
    
    # Test 2: max_features=40000, class_weight=None
    test_config(max_features=40000, class_weight=None, name_tag="40k features + No Class Weight")

if __name__ == "__main__":
    main()
