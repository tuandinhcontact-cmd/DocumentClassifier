import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import optimized custom models
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM

class CustomOneVsRestClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            model = copy.deepcopy(self.base_estimator)
            model.fit(X, y_binary)
            self.models[cls] = model
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        for idx, cls in enumerate(self.classes):
            prob = self.models[cls].predict_proba(X)[:, 1]
            probs[:, idx] = prob
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

class CustomMultiClassVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators
        
    def fit(self, X, y):
        for name, est in self.estimators:
            print(f"Training {name} on all 14 classes...")
            est.fit(X, y)
        self.classes = self.estimators[0][1].classes
        return self
        
    def predict_proba(self, X):
        all_probs = []
        for name, est in self.estimators:
            all_probs.append(est.predict_proba(X))
        return np.mean(all_probs, axis=0)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

def main():
    dataset_path = "data/merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Missing dataset file {dataset_path}.")
        return

    print("Loading full dataset...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # 1. Stratified split into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    
    X_train_raw = train_df['cleaned_text']
    y_train = train_df['category']
    X_test_raw = test_df['cleaned_text']
    y_test = test_df['category']
    
    print(f"Train set: {len(X_train_raw)} samples, Test set: {len(X_test_raw)} samples")
    
    # 2. Vectorization
    print("TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 3. Flat models for all 14 classes
    lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced'))
    svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight='balanced'))
    mnb_mc = CustomMultinomialNB(alpha=1.0)
    
    # Flat Ensemble
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
    
    print(f"\nFlat Classifier Results:")
    print(f"Training & Evaluation Time: {elapsed:.2f} seconds")
    print(f"Test Accuracy: {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
