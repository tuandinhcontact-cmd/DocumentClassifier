import os
import pickle
import numpy as np
import sys

# Thêm project root vào PYTHONPATH để import custom_models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_models.logistic_regression import CustomLogisticRegression
from custom_models.multinomial_nb import CustomMultinomialNB
from custom_models.linear_svm import CustomLinearSVM
from custom_models.flat_ensemble import CustomOneVsRestClassifier, FlatSoftVotingClassifier

MODEL_PATH = "models/flat_gridsearch_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model not found!")
    exit(1)

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

print("Loaded classes:", model_data['classes'])
print("Total number of classes:", len(model_data['classes']))
print("Best params:", model_data['best_params'])

vectorizer = model_data['vectorizer']
model = model_data['model']

test_texts = [
    "Jannik Sinner defeated Novak Djokovic in the Australian Open semifinal in four sets",
    "Nirmala Sitharaman will present the interim budget for the upcoming fiscal year in Parliament",
    "Students are adjusting to online courses and classroom learning strategies at universities",
    "New advances in artificial intelligence and machine learning models are being developed by tech companies"
]

X_vec = vectorizer.transform(test_texts)
preds = model.predict(X_vec)

for text, pred in zip(test_texts, preds):
    print(f"Text: {text}")
    print(f"Predicted Class: {pred}")
    print("-" * 50)
