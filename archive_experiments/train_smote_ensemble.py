import os
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE

def main():
    print("=== Training SMOTE + TF-IDF + Soft Voting Ensemble ===")
    
    # Configure parameters
    USE_SUBSET = False
    SUBSET_SIZE = 30000
    MAX_FEATURES = 5000
    
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
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. TF-IDF vectorization
    print("Vectorizing using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    # 4. Apply SMOTE to balance the training set
    print("Applying SMOTE to balance classes...")
    start_smote = time.time()
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)
    print(f"SMOTE finished in {time.time() - start_smote:.2f}s.")
    print(f"Original training samples: {X_train_tfidf.shape[0]}")
    print(f"Resampled training samples: {X_train_resampled.shape[0]}")
    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
    
    # 5. Initialize Ensemble (Voting Classifier)
    print("Initializing Soft Voting Ensemble components...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    mnb = MultinomialNB()
    svm = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False))
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
    
    # 6. Fit Ensemble
    print("Training Ensemble on SMOTE-balanced training set...")
    start_train = time.time()
    ensemble.fit(X_train_resampled, y_train_resampled)
    train_time = time.time() - start_train
    print(f"Ensemble training completed in {train_time:.2f} seconds.")
    
    # 7. Predict and evaluate
    print("Evaluating on test set...")
    y_pred = ensemble.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Accuracy: {acc:.4%}")
    print(f"Macro F1-Score: {macro_f1:.4%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
