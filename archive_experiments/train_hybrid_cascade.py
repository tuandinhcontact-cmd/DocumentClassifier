import os
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from gensim.models import Word2Vec

def document_vector(tokenized_doc, w2v_model, vector_size):
    """
    Computes the average word vector for a document.
    """
    words = [word for word in tokenized_doc if word in w2v_model.wv]
    if len(words) >= 1:
        return np.mean(w2v_model.wv[words], axis=0)
    else:
        return np.zeros(vector_size)

class HybridCascadeClassifier:
    def __init__(self, threshold_large=5000, vector_size=100, max_features=25000):
        self.threshold_large = threshold_large
        self.vector_size = vector_size
        self.max_features = max_features
        
        self.models = {}
        self.classes_order = []
        self.class_counts = {}
        self.node_metadata = {} # Store feature_type, sample_count, etc. per class
        
        self.global_tfidf = None
        self.global_w2v = None
        self.global_scaler = None

    def fit(self, X_raw, y):
        # 1. Calculate class statistics
        y_series = pd.Series(y)
        class_counts = y_series.value_counts()
        self.classes_order = class_counts.index.tolist()
        self.class_counts = class_counts.to_dict()
        
        # 2. Fit Global TF-IDF Vectorizer
        print("Fitting global TF-IDF Vectorizer...")
        self.global_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features)
        self.global_tfidf.fit(X_raw)
        
        # 3. Train Global Word2Vec Model
        print("Tokenizing texts for Word2Vec training...")
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
        
        # 4. Generate Global Word2Vec embeddings and fit global scaler
        print("Generating global Word2Vec vectors...")
        X_w2v_all = np.array([document_vector(doc, self.global_w2v, self.vector_size) for doc in tokenized_docs])
        print("Fitting global MinMaxScaler for Word2Vec...")
        self.global_scaler = MinMaxScaler()
        self.global_scaler.fit(X_w2v_all)
        
        # Sequentially train cascade nodes
        current_X_raw = np.array(X_raw)
        current_y = np.array(y)
        
        print("\n--- Training Hybrid Cascade Nodes (One-vs-Rest) ---")
        for i, class_name in enumerate(self.classes_order[:-1]):
            class_size = self.class_counts[class_name]
            y_binary = (current_y == class_name).astype(int)
            
            # Determine feature representation based on training samples of this class
            if class_size >= self.threshold_large:
                feature_type = 'word2vec'
                # Extract Word2Vec features for current active samples
                current_tokens = [doc.split() for doc in current_X_raw]
                X_features = np.array([document_vector(doc, self.global_w2v, self.vector_size) for doc in current_tokens])
                X_features = self.global_scaler.transform(X_features)
                
                # Define classifiers suitable for Word2Vec (continuous dense features)
                lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual=False))
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                
                ensemble = VotingClassifier(
                    estimators=[
                        ('lr', lr),
                        ('svm', svm),
                        ('rf', rf)
                    ],
                    voting='soft'
                )
            else:
                feature_type = 'tfidf'
                # Extract TF-IDF features
                X_features = self.global_tfidf.transform(current_X_raw)
                
                # Define classifiers suitable for TF-IDF (sparse counts/frequencies)
                lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
                mnb = MultinomialNB()
                svm = CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, dual=False))
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                
                ensemble = VotingClassifier(
                    estimators=[
                        ('lr', lr),
                        ('mnb', mnb),
                        ('svm', svm),
                        ('rf', rf)
                    ],
                    voting='soft'
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
            
            # Filter remaining samples for the next node
            remaining_mask = (y_binary == 0)
            if not remaining_mask.any():
                break
                
            current_X_raw = current_X_raw[remaining_mask]
            current_y = current_y[remaining_mask]
            
            print(f"Node {i+1} [{class_name}] ({feature_type.upper()}): Trained in {step_time:.2f}s. Remaining: {len(current_y)} samples.")
            
        print("Successfully trained all Hybrid Cascade Nodes!")
        
    def predict_proba_node(self, X_raw, class_name):
        """
        Predict binary probability of belonging to class_name for a list of raw documents.
        """
        metadata = self.node_metadata.get(class_name)
        if not metadata:
            raise ValueError(f"No metadata found for class: {class_name}")
            
        feature_type = metadata['feature_type']
        model = self.models[class_name]
        
        if feature_type == 'word2vec':
            tokenized = [doc.split() for doc in X_raw]
            X_features = np.array([document_vector(doc, self.global_w2v, self.vector_size) for doc in tokenized])
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
            
            # Predict probabilities
            probs = self.predict_proba_node(X_unclass.tolist(), class_name)
            # Binary prediction threshold = 0.5
            preds_binary = (probs[:, 1] >= 0.5).astype(int)
            
            is_class_indices = unclassified_indices[preds_binary == 1]
            predictions[is_class_indices] = class_name
            classified[is_class_indices] = True
            
        return predictions

def main():
    print("=== Training Hybrid Word2Vec & TF-IDF Cascade Classifier ===")
    
    # Configure parameters
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 25000
    THRESHOLD_LARGE = 5000
    VECTOR_SIZE = 100
    
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
        
    X = df['cleaned_text'].tolist()
    y = df['category'].tolist()
    
    # 2. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Fit Hybrid Model
    start_time = time.time()
    hybrid_cascade = HybridCascadeClassifier(
        threshold_large=THRESHOLD_LARGE,
        vector_size=VECTOR_SIZE,
        max_features=MAX_FEATURES
    )
    hybrid_cascade.fit(X_train, y_train)
    
    # 4. Predict and evaluate
    print("\nPredicting on test set...")
    y_pred = hybrid_cascade.predict(X_test)
    elapsed_time = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTotal Training & Evaluation Time: {elapsed_time:.2f} seconds")
    print(f"Test Accuracy: {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Export hybrid model to pickle file
    print("\nSaving hybrid model data to cascade_model.pkl...")
    model_data = {
        'model': hybrid_cascade,
        'vectorizer': hybrid_cascade.global_tfidf,
        'classes_order': hybrid_cascade.classes_order
    }
    with open("cascade_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("Successfully exported hybrid cascade model!")

if __name__ == "__main__":
    main()
