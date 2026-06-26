import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from train_3step_cascade import ThreeStepCascadeClassifier, CustomOneVsRestClassifier, CustomMultiClassVotingClassifier
from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM

def main():
    dataset_path = "data/merged_cleaned_dataset.csv"
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['cleaned_text'].fillna('')
    
    X_raw = df['cleaned_text']
    y = df['category']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Initializing estimators...")
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
    
    print("Training 3-Step Cascade on new dataset...")
    cascade = ThreeStepCascadeClassifier(
        binary_estimators=binary_estimators, 
        multiclass_estimator=multiclass_estimator,
        max_features_step1=40000,
        max_features_step2=40000,
        max_features_step3=20000
    )
    
    start_time = time.time()
    cascade.fit(X_train_raw, y_train)
    y_pred = cascade.predict(X_test_raw)
    elapsed = time.time() - start_time
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n[3-Step Cascade on Clean Data] Time: {elapsed:.2f}s | Test Accuracy: {acc:.4%} | Macro F1: {macro_f1:.4%}")

if __name__ == "__main__":
    main()
