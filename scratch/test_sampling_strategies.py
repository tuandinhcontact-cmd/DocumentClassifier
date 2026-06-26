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
        
        # Step 1: Tech & Science vs Rest
        y_bin1 = (current_y == self.target1).astype(int)
        self.step1_model = self._select_best_binary(current_X, y_bin1, self.target1)
        
        mask1 = (y_bin1 == 0)
        current_X = current_X[mask1]
        current_y = current_y[mask1]
        
        # Step 2: Politics and society vs Rest
        y_bin2 = (current_y == self.target2).astype(int)
        self.step2_model = self._select_best_binary(current_X, y_bin2, self.target2)
        
        mask2 = (y_bin2 == 0)
        current_X = current_X[mask2]
        current_y = current_y[mask2]
        
        # Step 3: Multiclass for remaining 12 classes
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

def apply_undersampling(df, cap_limit):
    if cap_limit is None:
        return df
    sampled_groups = []
    for cat_name, group in df.groupby("category"):
        sampled_group = group.sample(
            n=min(len(group), cap_limit),
            random_state=42
        )
        sampled_groups.append(sampled_group)
    return pd.concat(sampled_groups).sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    dataset_path = "data/merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Missing dataset file {dataset_path}.")
        return

    print("Loading full dataset...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    # 1. Stratified split into Train (80%) and Test (20%)
    # This keeps a realistic imbalanced Test set
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['category']
    )
    
    X_test_raw = test_df['cleaned_text']
    y_test = test_df['category']
    
    # Experiment caps
    caps = [2000, 4000, 8000, 15000, 30000, None]
    
    results = []
    
    for cap in caps:
        print(f"\n--- Testing Undersampling Cap: {cap} ---")
        
        # Apply cap to Train set only
        train_sampled = apply_undersampling(train_df, cap)
        X_train_raw = train_sampled['cleaned_text']
        y_train = train_sampled['category']
        
        print(f"Train size: {len(train_sampled)} rows")
        
        # Fit TF-IDF on Train set only
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)
        
        # Initialize models
        lr = CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced')
        mnb = CustomMultinomialNB(alpha=1.0)
        svm = CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight='balanced')
        
        binary_estimators = [
            ('LogisticRegression', lr),
            ('MultinomialNB', mnb),
            ('LinearSVM', svm)
        ]
        
        lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced'))
        svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.01, lambda_param=0.01, epochs=100, class_weight='balanced'))
        mnb_mc = CustomMultinomialNB(alpha=1.0)
        
        multiclass_estimator = CustomMultiClassVotingClassifier(estimators=[
            ('MultinomialNB', mnb_mc),
            ('LogisticRegression_OVR', lr_ovr),
            ('LinearSVM_OVR', svm_ovr)
        ])
        
        cascade = ThreeStepCascadeClassifier(
            binary_estimators=binary_estimators, 
            multiclass_estimator=multiclass_estimator
        )
        
        start_time = time.time()
        cascade.fit(X_train, y_train)
        y_pred = cascade.predict(X_test)
        elapsed = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Time: {elapsed:.2f}s | Test Accuracy: {acc:.4%} | Macro F1-Score: {macro_f1:.4%}")
        results.append({
            'cap': cap,
            'train_size': len(train_sampled),
            'test_acc': acc,
            'macro_f1': macro_f1,
            'time': elapsed
        })
        
    print("\n=== SUMMARY OF RESULTS ===")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
