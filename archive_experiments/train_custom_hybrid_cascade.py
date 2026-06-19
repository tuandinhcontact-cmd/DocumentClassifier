import os
import pickle
import numpy as np
import pandas as pd
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from gensim.models import Word2Vec

# Import 4 custom models from scratch
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.gaussian_nb import CustomGaussianNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.random_forest import CustomRandomForest

def csr_to_list_of_dicts(csr_matrix):
    """
    Converts a scipy CSR sparse matrix to a list of dicts for fast O(1) index lookup.
    """
    list_of_dicts = []
    n_samples = csr_matrix.shape[0]
    indptr = csr_matrix.indptr
    indices = csr_matrix.indices
    data = csr_matrix.data
    
    for i in range(n_samples):
        row_dict = {}
        for idx in range(indptr[i], indptr[i+1]):
            row_dict[indices[idx]] = data[idx]
        list_of_dicts.append(row_dict)
    return list_of_dicts

def tfidf_weighted_document_vector(tokenized_doc, w2v_model, vector_size, tfidf_row_sparse, vocab_dict):
    """
    Computes the TF-IDF weighted average word vector for a single document.
    """
    words = [word for word in tokenized_doc if word in w2v_model.wv]
    if len(words) < 1:
        return np.zeros(vector_size)
        
    vectors = []
    weights = []
    
    for word in words:
        v = w2v_model.wv[word]
        col_idx = vocab_dict.get(word, -1)
        w = tfidf_row_sparse.get(col_idx, 1.0) if col_idx != -1 else 1.0
        
        vectors.append(v)
        weights.append(w)
        
    weights = np.array(weights)
    sum_weights = np.sum(weights)
    if sum_weights > 0:
        return np.average(vectors, axis=0, weights=weights)
    else:
        return np.mean(vectors, axis=0)

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

# Custom Hybrid Cascade Classifier from scratch with optimizations
class CustomHybridCascadeClassifier:
    def __init__(self, threshold_large=1000, vector_size=100, max_features=10000):
        self.threshold_large = threshold_large
        self.vector_size = vector_size
        self.max_features = max_features
        
        self.models = {}
        self.classes_order = []
        self.class_counts = {}
        self.node_metadata = {}
        
        self.global_tfidf = None
        self.global_w2v = None
        self.global_scaler = None

    def fit(self, X_raw, y):
        y_series = pd.Series(y)
        class_counts = y_series.value_counts()
        self.classes_order = class_counts.index.tolist()
        self.class_counts = class_counts.to_dict()
        
        # 1. Fit Global TF-IDF
        print("Fitting custom TF-IDF Vectorizer...")
        self.global_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features)
        self.global_tfidf.fit(X_raw)
        
        # 2. Fit Global Word2Vec
        print("Tokenizing texts for Word2Vec...")
        tokenized_docs = [doc.split() for doc in X_raw]
        print(f"Training Word2Vec model (vector_size={self.vector_size})...")
        self.global_w2v = Word2Vec(
            sentences=tokenized_docs,
            vector_size=self.vector_size,
            window=5,
            min_count=2,
            workers=4,
            seed=42
        )
        
        # 3. Fit Global Scaler using TF-IDF weighted Word2Vec
        print("Generating TF-IDF weighted Word2Vec vectors for global scaling...")
        X_tfidf_all = self.global_tfidf.transform(X_raw)
        tfidf_dicts_all = csr_to_list_of_dicts(X_tfidf_all)
        
        X_w2v_all = np.array([
            tfidf_weighted_document_vector(
                tokenized_docs[i], self.global_w2v, self.vector_size, tfidf_dicts_all[i], self.global_tfidf.vocabulary_
            ) for i in range(len(tokenized_docs))
        ])
        
        self.global_scaler = MinMaxScaler()
        self.global_scaler.fit(X_w2v_all)
        
        # Sequentially train custom cascade nodes
        current_X_raw = np.array(X_raw)
        current_y = np.array(y)
        
        print("\n--- Training Custom Hybrid Cascade Nodes (One-vs-Rest) ---")
        for i, class_name in enumerate(self.classes_order[:-1]):
            class_size = self.class_counts[class_name]
            y_binary = (current_y == class_name).astype(int)
            
            # Defensive check: if class has no samples in the active set
            if len(np.unique(y_binary)) < 2:
                print(f"Skipping class '{class_name}' because it does not have enough active samples.")
                continue
                
            # Determine feature representation based on class size
            if class_size >= self.threshold_large:
                feature_type = 'word2vec'
                # Extract TF-IDF weighted Word2Vec features for current active samples
                current_tfidf = self.global_tfidf.transform(current_X_raw)
                current_tfidf_dicts = csr_to_list_of_dicts(current_tfidf)
                current_tokens = [doc.split() for doc in current_X_raw]
                
                X_features = np.array([
                    tfidf_weighted_document_vector(
                        current_tokens[idx], self.global_w2v, self.vector_size, current_tfidf_dicts[idx], self.global_tfidf.vocabulary_
                    ) for idx in range(len(current_tokens))
                ])
                X_features = self.global_scaler.transform(X_features)
                
                # Optimized custom models for Word2Vec (continuous dense features)
                lr = CustomLogisticRegression(lr=0.5, epochs=100, C=1.0, class_weight='balanced')
                gnb = CustomGaussianNB()
                svm = CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20)
                rf = CustomRandomForest(n_estimators=10, max_depth=6, min_samples_split=2, n_jobs=-1) # Increased trees and depth since parallelized!
                
                ensemble = CustomVotingClassifier(
                    estimators=[
                        ('lr', lr),
                        ('gnb', gnb),
                        ('svm', svm),
                        ('rf', rf)
                    ]
                )
            else:
                feature_type = 'tfidf'
                # Extract TF-IDF features
                X_features = self.global_tfidf.transform(current_X_raw)
                
                # Optimized custom models for TF-IDF (sparse counts/frequencies)
                lr = CustomLogisticRegression(lr=0.5, epochs=100, C=1.0, class_weight='balanced')
                gnb = CustomGaussianNB()
                svm = CustomLinearSVM(lr=0.1, lambda_param=0.01, epochs=20)
                rf = CustomRandomForest(n_estimators=10, max_depth=6, min_samples_split=2, n_jobs=-1) # Increased trees and depth since parallelized!
                
                ensemble = CustomVotingClassifier(
                    estimators=[
                        ('lr', lr),
                        ('gnb', gnb),
                        ('svm', svm),
                        ('rf', rf)
                    ]
                )
            
            start_step = time.time()
            ensemble.fit(X_features, y_binary)
            step_time = time.time() - start_step
            
            self.models[class_name] = ensemble
            self.node_metadata[class_name] = {
                'feature_type': feature_type,
                'class_size': int(class_size),
                'training_samples': int(len(current_y))
            }
            
            # Filter remaining samples
            remaining_mask = (y_binary == 0)
            if not remaining_mask.any():
                break
                
            current_X_raw = current_X_raw[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Node {i+1} [{class_name}] ({feature_type.upper()}): Trained in {step_time:.2f}s. Remaining: {len(current_y)} samples.")
            
        print("Successfully trained all Custom Hybrid Cascade Nodes!")
        
    def predict_proba_node(self, X_raw, class_name):
        metadata = self.node_metadata.get(class_name)
        if not metadata:
            raise ValueError(f"No metadata found for class: {class_name}")
            
        feature_type = metadata['feature_type']
        model = self.models[class_name]
        
        if feature_type == 'word2vec':
            X_tfidf = self.global_tfidf.transform(X_raw)
            tfidf_dicts = csr_to_list_of_dicts(X_tfidf)
            tokenized = [doc.split() for doc in X_raw]
            X_features = np.array([
                tfidf_weighted_document_vector(
                    tokenized[idx], self.global_w2v, self.vector_size, tfidf_dicts[idx], self.global_tfidf.vocabulary_
                ) for idx in range(len(tokenized))
            ])
            X_features = self.global_scaler.transform(X_features)
        else: # tfidf
            X_features = self.global_tfidf.transform(X_raw)
            
        return model.predict_proba(X_features)
        
    def predict(self, X_raw):
        num_samples = len(X_raw)
        predictions = np.full(num_samples, self.classes_order[-1], dtype=object)
        classified = np.zeros(num_samples, dtype=bool)
        
        X_raw_arr = np.array(X_raw)
        
        for class_name in self.classes_order[:-1]:
            if class_name not in self.models:
                continue
                
            unclassified_indices = np.where(~classified)[0]
            if len(unclassified_indices) == 0:
                break
                
            X_unclass = X_raw_arr[unclassified_indices]
            
            probs = self.predict_proba_node(X_unclass.tolist(), class_name)
            preds_binary = (probs[:, 1] >= 0.5).astype(int)
            
            is_class_indices = unclassified_indices[preds_binary == 1]
            predictions[is_class_indices] = class_name
            classified[is_class_indices] = True
            
        return predictions

def main():
    print("=== Training Custom Optimized Hybrid Cascade Classifier ===")
    
    # Configure performance params
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    THRESHOLD_LARGE = 5000
    VECTOR_SIZE = 100
    MAX_FEATURES = 10000
    
    dataset_path = "merged_cleaned_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Missing dataset file {dataset_path}.")
        return

    # 1. Read data
    print("Reading dataset...")
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    if USE_SUBSET and len(df) > SUBSET_SIZE:
        print(f"Sampling {SUBSET_SIZE} rows to speed up from-scratch training...")
        df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df['category'], random_state=42)
        df = df.reset_index(drop=True)
        
    X = df['cleaned_text'].tolist()
    y = df['category'].tolist()
    
    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Fit Custom Hybrid Model
    start_time = time.time()
    custom_hybrid_cascade = CustomHybridCascadeClassifier(
        threshold_large=THRESHOLD_LARGE,
        vector_size=VECTOR_SIZE,
        max_features=MAX_FEATURES
    )
    custom_hybrid_cascade.fit(X_train, y_train)
    
    # 4. Predict and evaluate
    print("\nPredicting on test set...")
    y_pred = custom_hybrid_cascade.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTotal Training & Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"Test Accuracy: {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Export custom model to pickle file
    print("\nSaving custom hybrid model data to cascade_model.pkl...")
    model_data = {
        'model': custom_hybrid_cascade,
        'vectorizer': custom_hybrid_cascade.global_tfidf,
        'classes_order': custom_hybrid_cascade.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Successfully exported custom hybrid cascade model!")

if __name__ == "__main__":
    main()
