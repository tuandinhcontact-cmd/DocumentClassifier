import os
import pickle
import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
            
        # L1 Normalization
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

class CustomMultiClassVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators # List of tuples: (name, estimator_object)
        
    def fit(self, X, y):
        for name, est in self.estimators:
            print(f"    -> Training {name} for MultiClassVoting...")
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
        self.classes_order = []

    def _select_best_binary(self, X, y, step_name):
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        best_model_name = None
        best_model = None
        best_score = -1
        
        print(f"  Selecting best model for {step_name}...")
        for name, est_template in self.binary_estimators:
            model = copy.deepcopy(est_template)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds)
            print(f"    - {name}: F1={score:.4f}")
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                
        print(f"  -> Best model for {step_name} is {best_model_name} (Val F1: {best_score:.4f}). Retraining...")
        best_model.fit(X, y)
        return best_model

    def fit(self, X, y):
        current_X = X
        current_y = np.array(y)
        
        # Step 1: Tech & Science
        print(f"\n--- Step 1: Binary Classification for '{self.target1}' vs Rest ---")
        y_bin1 = (current_y == self.target1).astype(int)
        
        start_step1 = time.time()
        self.step1_model = self._select_best_binary(current_X, y_bin1, self.target1)
        
        mask1 = (y_bin1 == 0)
        current_X = current_X[mask1]
        current_y = current_y[mask1]
        print(f"Step 1 Trained in {time.time() - start_step1:.2f}s. Remaining samples: {len(current_y)}")
        
        # Step 2: Politics and society
        print(f"\n--- Step 2: Binary Classification for '{self.target2}' vs Rest ---")
        y_bin2 = (current_y == self.target2).astype(int)
        
        start_step2 = time.time()
        self.step2_model = self._select_best_binary(current_X, y_bin2, self.target2)
        
        mask2 = (y_bin2 == 0)
        current_X = current_X[mask2]
        current_y = current_y[mask2]
        print(f"Step 2 Trained in {time.time() - start_step2:.2f}s. Remaining samples: {len(current_y)}")
        
        # Step 3: Multiclass for remaining 12 classes
        print(f"\n--- Step 3: Multi-class Classification for remaining 12 classes ---")
        start_step3 = time.time()
        self.step3_model = copy.deepcopy(self.multiclass_estimator)
        self.step3_model.fit(current_X, current_y)
        print(f"Step 3 Trained in {time.time() - start_step3:.2f}s.")
        
        print("\nSuccessfully trained 3-Step Cascade Classifier with Global TF-IDF!")
        
    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.full(num_samples, "UNKNOWN", dtype=object)
        classified = np.zeros(num_samples, dtype=bool)
        
        # Step 1
        preds1 = self.step1_model.predict(X)
        is_target1 = (preds1 == 1)
        predictions[is_target1] = self.target1
        classified[is_target1] = True
        
        # Step 2
        unclassified_idx = np.where(~classified)[0]
        if len(unclassified_idx) > 0:
            X_step2 = X[unclassified_idx]
            preds2 = self.step2_model.predict(X_step2)
            is_target2 = (preds2 == 1)
            
            idx_target2 = unclassified_idx[is_target2]
            predictions[idx_target2] = self.target2
            classified[idx_target2] = True
            
        # Step 3
        unclassified_idx = np.where(~classified)[0]
        if len(unclassified_idx) > 0:
            X_step3 = X[unclassified_idx]
            preds3 = self.step3_model.predict(X_step3)
            predictions[unclassified_idx] = preds3
            classified[unclassified_idx] = True
            
        return predictions

def main():
    print("=== Training 3-Step Hybrid Cascade Classifier ===")
    
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 20000
    
    dataset_path = "data/merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Missing dataset file {dataset_path}.")
        return

    # 1. Read data
    print("Reading dataset...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Sampling {SUBSET_SIZE} rows...")
        df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text']
    y = df['category']
    
    # 2. Train/Test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. TF-IDF vectorization
    print("Vector hóa văn bản bằng TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 4. Initialize optimized custom estimators
    print("Initializing estimators...")
    lr = CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced')
    mnb = CustomMultinomialNB(alpha=1.0)
    svm = CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20)
    
    binary_estimators = [
        ('LogisticRegression', lr),
        ('MultinomialNB', mnb),
        ('LinearSVM', svm)
    ]
    
    # Custom OVR Multi-class models for Step 3
    lr_ovr = CustomOneVsRestClassifier(base_estimator=CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced'))
    svm_ovr = CustomOneVsRestClassifier(base_estimator=CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20))
    mnb_mc = CustomMultinomialNB(alpha=1.0)
    
    # Soft Voting Multi-class ensemble
    multiclass_estimator = CustomMultiClassVotingClassifier(estimators=[
        ('MultinomialNB', mnb_mc),
        ('LogisticRegression_OVR', lr_ovr),
        ('LinearSVM_OVR', svm_ovr)
    ])
    
    # 5. Fit 3-Step Cascade Classifier
    start_time = time.time()
    cascade_custom = ThreeStepCascadeClassifier(
        binary_estimators=binary_estimators, 
        multiclass_estimator=multiclass_estimator
    )
    cascade_custom.fit(X_train, y_train)
    
    # 6. Predict and evaluate
    print("\nPredicting on test set...")
    y_pred = cascade_custom.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTotal Training & Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"Test Accuracy: {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Export custom model to pickle file
    print("\nSaving 3-step cascade model data to cascade_model.pkl...")
    model_data = {
        'model': cascade_custom,
        'vectorizer': vectorizer,
        'classes_order': cascade_custom.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Successfully exported 3-step cascade model!")

if __name__ == "__main__":
    main()
