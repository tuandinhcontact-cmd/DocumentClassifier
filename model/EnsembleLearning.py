import sys
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import GridSearchCV


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1" / "train_test_data"

sys.path.append(str(BASE_DIR))


# =========================
# IMPORT CUSTOM MODELS
# =========================

from Algorithms.LogisticRegression import LogisticRegression
from Algorithms.GaussianNB import GaussianNB
from Algorithms.SupportVectorMachine import SupportVectorMachine
from Algorithms.RandomForest import RandomForest


# =========================
# LOAD DATA
# =========================

def load_data(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    return joblib.load(file_path)


X_train = load_data("X_train_postSMOTE.pkl")
y_train = load_data("y_train_postSMOTE.pkl")

X_test = load_data("X_test_preSMOTE.pkl")
y_test = load_data("y_test_preSMOTE.pkl")


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# =========================
# WRAPPER CHO MODEL TỰ VIẾT
# =========================

class CustomClassifierWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, model_class):
        self.model_class = model_class

    def fit(self, X, y):
        self.model = self.model_class()
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array(self.model.predict(X))

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        y_pred = self.predict(X)

        proba = np.zeros((len(y_pred), len(self.classes_)))

        for i, label in enumerate(y_pred):
            class_index = np.where(self.classes_ == label)[0][0]
            proba[i][class_index] = 1.0

        return proba


# =========================
# BUILD ENSEMBLE
# =========================

ensemble = VotingClassifier(
    estimators=[
        ("lr", CustomClassifierWrapper(LogisticRegression)),
        ("nb", CustomClassifierWrapper(GaussianNB)),
        ("svm", CustomClassifierWrapper(SupportVectorMachine)),
        ("rf", CustomClassifierWrapper(RandomForest))
    ],
    voting="soft"
)


# =========================
# TRAIN KHÔNG GRIDSEARCH
# =========================

start_time = time.time()
ensemble.fit(X_train, y_train)
end_time = time.time()

print(f"\nThời gian huấn luyện không GridSearchCV: {end_time - start_time:.2f} giây")


# Train evaluation
y_train_pred = ensemble.predict(X_train)
y_train_proba = ensemble.predict_proba(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_loss = log_loss(y_train, y_train_proba, labels=ensemble.classes_)

print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Train Log Loss : {train_loss:.4f}")


# Test evaluation
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_loss = log_loss(y_test, y_proba, labels=ensemble.classes_)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Log Loss : {test_loss:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# =========================
# GRIDSEARCH
# =========================

param_grid = {
    "weights": [
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [1, 2, 1, 1],
        [1, 1, 2, 1],
        [1, 1, 1, 2]
    ]
}

grid_search = GridSearchCV(
    estimator=ensemble,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1
)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nThời gian huấn luyện GridSearchCV: {end_time - start_time:.2f} giây")
print(f"Best Params: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")


# =========================
# EVALUATE BEST MODEL
# =========================

best_ensemble = grid_search.best_estimator_

MODEL_DIR = Path(__file__).resolve().parent

joblib.dump(best_ensemble, MODEL_DIR / "ensemble_classifier_pipeline.pkl")
print("Đã lưu ensemble model vào model/ensemble_classifier_pipeline.pkl")

y_pred_gs = best_ensemble.predict(X_test)
y_proba_gs = best_ensemble.predict_proba(X_test)

test_accuracy_gs = accuracy_score(y_test, y_pred_gs)
test_loss_gs = log_loss(y_test, y_proba_gs, labels=best_ensemble.classes_)

print(f"\nTest Accuracy GridSearchCV: {test_accuracy_gs:.4f}")
print(f"Test Log Loss GridSearchCV : {test_loss_gs:.4f}")

print("\nClassification Report GridSearchCV:")
print(classification_report(y_test, y_pred_gs, zero_division=0))