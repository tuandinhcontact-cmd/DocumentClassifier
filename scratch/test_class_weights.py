import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

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

class ThreeStepCascadeClassifier:
    def __init__(self, binary_estimators, multiclass_estimator):
        self.binary_estimators = binary_estimators
        self.multiclass_estimator = multiclass_estimator
        self.step1_model = None
        self.step2_model = None
        self.step3_model = None
        self.target1 = "Tech & Science"
        self.target2 = "Politics and society"

    def _select_best_binary(self, X, y, step_name):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        best_model = None
        best_score = -1
        for name, est_template in self.binary_estimators:
            model = copy.deepcopy(est_template)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds)
            if score > best_score:
                best_score = score
                best_model = model
        best_model.fit(X, y)
        return best_model

    def fit(self, X, y):
        current_X = X
        current_y = np.array(y)
        
        # Step 1
        y_bin1 = (current_y == self.target1).astype(int)
        self.step1_model = self._select_best_binary(current_X, y_bin1, self.target1)
        
        mask1 = (y_bin1 == 0)
        current_X = current_X[mask1]
        current_y = current_y[mask1]
        
        # Step 2
        y_bin2 = (current_y == self.target2).astype(int)
        self.step2_model = self._select_best_binary(current_X, y_bin2, self.target2)
        
        mask2 = (y_bin2 == 0)
        current_X = current_X[mask2]
        current_y = current_y[mask2]
        
        # Step 3
        self.step3_model = copy.deepcopy(self.multiclass_estimator)
        self.step3_model.fit(current_X, current_y)

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.full(num_samples, "UNKNOWN", dtype=object)
        classified = np.zeros(num_samples, dtype=bool)
        
        preds1 = self.step1_model.predict(X)
        is_target1 = (preds1 == 1)
        predictions[is_target1] = self.target1
        classified[is_target1] = True
        
        unclassified_idx = np.where(~classified)[0]
        if len(unclassified_idx) > 0:
            X_step2 = X[unclassified_idx]
            preds2 = self.step2_model.predict(X_step2)
            is_target2 = (preds2 == 1)
            idx_target2 = unclassified_idx[is_target2]
            predictions[idx_target2] = self.target2
            classified[idx_target2] = True
            
        unclassified_idx = np.where(~classified)[0]
        if len(unclassified_idx) > 0:
            X_step3 = X[unclassified_idx]
            preds3 = self.step3_model.predict(X_step3)
            predictions[unclassified_idx] = preds3
            
        return predictions

def main():
    dataset_path = "data/merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        return

    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    
    X_train_raw = train_df['cleaned_text']
    y_train = train_df['category']
    X_test_raw = test_df['cleaned_text']
    y_test = test_df['category']
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    print("--- 1. Flat Classifier (class_weight=None) ---")
    lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight=None))
    svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight=None))
    mnb_mc = CustomMultinomialNB(alpha=1.0)
    
    flat_ensemble = CustomMultiClassVotingClassifier(estimators=[
        ('MultinomialNB', mnb_mc),
        ('LogisticRegression_OVR', lr_ovr),
        ('LinearSVM_OVR', svm_ovr)
    ])
    flat_ensemble.fit(X_train, y_train)
    y_pred_flat = flat_ensemble.predict(X_test)
    print(f"Flat Acc: {accuracy_score(y_test, y_pred_flat):.4%}")
    
    print("\n--- 2. Cascade Classifier (class_weight=None) ---")
    lr = CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight=None)
    mnb = CustomMultinomialNB(alpha=1.0)
    svm = CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight=None)
    binary_estimators = [('LogisticRegression', lr), ('MultinomialNB', mnb), ('LinearSVM', svm)]
    
    lr_ovr_c = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight=None))
    svm_ovr_c = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight=None))
    mnb_mc_c = CustomMultinomialNB(alpha=1.0)
    multiclass_estimator = CustomMultiClassVotingClassifier(estimators=[
        ('MultinomialNB', mnb_mc_c),
        ('LogisticRegression_OVR', lr_ovr_c),
        ('LinearSVM_OVR', svm_ovr_c)
    ])
    cascade = ThreeStepCascadeClassifier(binary_estimators, multiclass_estimator)
    cascade.fit(X_train, y_train)
    y_pred_cascade = cascade.predict(X_test)
    print(f"Cascade Acc: {accuracy_score(y_test, y_pred_cascade):.4%}")

if __name__ == "__main__":
    main()
