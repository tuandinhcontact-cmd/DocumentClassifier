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
from custom_models.random_forest import CustomRandomForest

# Custom Voting Classifier from scratch
class CustomVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators # List of tuples: (name, estimator_object)
        
    def fit(self, X, y):
        for name, est in self.estimators:
            est.fit(X, y)
        return self
        
    def predict_proba(self, X):
        probs = [est.predict_proba(X) for name, est in self.estimators]
        return np.mean(probs, axis=0)
        
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Custom Cascade Classifier using TF-IDF only and optimized models
class CustomHierarchicalCascadeClassifier:
    def __init__(self, estimators):
        self.estimators = estimators # List of tuples: (name, model)
        self.models = {}
        self.classes_order = []
        self.node_metadata = {}

    def fit(self, X, y):
        # Sort categories by size (descending)
        class_counts = pd.Series(y).value_counts()
        self.classes_order = class_counts.index.tolist()
        
        current_X = X
        current_y = np.array(y)
        
        print("\n--- Training Custom Optimized TF-IDF Cascade Nodes (One-vs-Rest) ---")
        for i, class_name in enumerate(self.classes_order[:-1]):
            # Binary labels
            y_binary = (current_y == class_name).astype(int)
            
            # Defensive check: if class has no samples in active set
            if len(np.unique(y_binary)) < 2:
                print(f"Skipping class '{class_name}' because it does not have enough active samples.")
                continue
                
            start_step = time.time()
            
            # Split for validation to pick the best model
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import f1_score
            
            X_tr, X_val, y_tr, y_val = train_test_split(current_X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
            
            best_model_name = None
            best_model = None
            best_score = -1
            
            print(f"  Selecting best model for node {i+1} [{class_name}]...")
            for name, est_template in self.estimators:
                model = copy.deepcopy(est_template)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                score = f1_score(y_val, preds)
                print(f"    - {name}: F1={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            
            # Retrain best model on full current data
            print(f"  -> Best model for [{class_name}] is {best_model_name} (Val F1: {best_score:.4f}). Retraining on all node data...")
            best_model.fit(current_X, y_binary)
            
            step_time = time.time() - start_step
            
            self.models[class_name] = best_model
            self.node_metadata[class_name] = {
                'class_size': int(class_counts[class_name]),
                'training_samples': int(len(current_y)),
                'best_model': best_model_name
            }
            
            # Filter remaining samples
            remaining_mask = (y_binary == 0)
            if not remaining_mask.any():
                break
                
            current_X = current_X[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Node {i+1} [{class_name}]: Trained in {step_time:.2f}s. Remaining: {len(current_y)} samples.\n")
            
        print("Successfully trained all Custom Cascade Nodes!")
        
    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.full(num_samples, self.classes_order[-1], dtype=object)
        classified = np.zeros(num_samples, dtype=bool)
        
        for class_name in self.classes_order[:-1]:
            if class_name not in self.models:
                continue
                
            unclassified_indices = np.where(~classified)[0]
            if len(unclassified_indices) == 0:
                break
                
            X_unclass = X[unclassified_indices]
            model = self.models[class_name]
            
            preds_binary = model.predict(X_unclass)
            
            is_class_indices = unclassified_indices[preds_binary == 1]
            predictions[is_class_indices] = class_name
            classified[is_class_indices] = True
            
        return predictions

def main():
    print("=== Training Custom Optimized TF-IDF Cascade Classifier ===")
    
    # Configure parameters
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 20000
    
    dataset_path = "merged_cleaned_dataset.csv"
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
    
    # 4. Initialize optimized custom models
    print("Initializing optimized custom estimators...")
    lr = CustomLogisticRegression(solver='adam', lr=0.01, epochs=100, class_weight='balanced')
    mnb = CustomMultinomialNB(alpha=1.0)
    svm = CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20)
    rf = CustomRandomForest(n_estimators=10, max_depth=12, min_samples_split=2, n_jobs=-1)
    
    estimators = [
        ('LogisticRegression', lr),
        ('MultinomialNB', mnb),
        ('LinearSVM', svm),
        ('RandomForest', rf)
    ]
    
    # 5. Fit Custom Cascade Classifier
    start_time = time.time()
    cascade_custom = CustomHierarchicalCascadeClassifier(estimators=estimators)
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
    print("\nSaving custom TF-IDF model data to cascade_model.pkl...")
    model_data = {
        'model': cascade_custom,
        'vectorizer': vectorizer,
        'classes_order': cascade_custom.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Successfully exported custom TF-IDF cascade model!")

if __name__ == "__main__":
    main()
