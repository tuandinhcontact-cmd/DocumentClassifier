import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def main():
    print("=== Ensemble Learning with Soft Voting ===")
    
    # 1. Load a sample dataset (Breast Cancer Wisconson dataset)
    print("\nLoading dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {target_names}")
    
    # 2. Split dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Standardize features (important for Logistic Regression, Naive Bayes, and SVC)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize individual classifiers
    # Note: For SVC to support soft voting (which averages predicted probabilities), 
    # we MUST set probability=True.
    log_reg = LogisticRegression(random_state=42, max_iter=10000)
    gnb = GaussianNB()
    svc = SVC(probability=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    
    # 5. Create the Ensemble Soft Voting Classifier
    estimators = [
        ('lr', log_reg),
        ('gnb', gnb),
        ('svc', svc),
        ('rf', rf)
    ]
    
    ensemble_clf = VotingClassifier(
        estimators=estimators,
        voting='soft'
    )
    
    # List of all models to evaluate
    models = {
        'Logistic Regression': log_reg,
        'Gaussian Naive Bayes': gnb,
        'Support Vector Machine (SVC)': svc,
        'Random Forest': rf,
        'Ensemble (Soft Voting)': ensemble_clf
    }
    
    results = {}
    
    # 6. Train and evaluate each model
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {'Accuracy': acc, 'ROC AUC': auc}
        
        print(f"\n[{name}]")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
    # 7. Print Summary
    print("\n" + "="*40)
    print("SUMMARY OF RESULTS")
    print("="*40)
    print(f"{'Classifier':<30} | {'Accuracy':<10} | {'ROC AUC':<10}")
    print("-"*56)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['Accuracy']:.4f}     | {metrics['ROC AUC']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
