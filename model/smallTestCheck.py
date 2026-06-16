import sys
import time
from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import GridSearchCV


# =========================
# CONFIG TEST NHANH
# =========================

TRAIN_PER_CLASS = 100      # mỗi class lấy tối đa 200 mẫu train
TEST_PER_CLASS = 15        # mỗi class lấy tối đa 30 mẫu test
RANDOM_STATE = 42

RUN_GRIDSEARCH = True     # test nhanh thì để False trước


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


def select_rows(X, indices):
    """
    Hỗ trợ cả numpy array, scipy sparse matrix, pandas DataFrame.
    """
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def balanced_subset(X, y, max_per_class, random_state=42):
    """
    Lấy tối đa max_per_class mẫu cho mỗi class.
    Cách này tốt hơn X[:3000] vì tránh bị thiếu class nếu dữ liệu đang sắp xếp theo nhãn.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y)

    selected_indices = []

    for label in np.unique(y_arr):
        label_indices = np.where(y_arr == label)[0]

        if len(label_indices) > max_per_class:
            chosen = rng.choice(label_indices, size=max_per_class, replace=False)
        else:
            chosen = label_indices

        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    X_sub = select_rows(X, selected_indices)
    y_sub = y_arr[selected_indices]

    return X_sub, y_sub


X_train = load_data("X_train_postSMOTE.pkl")
y_train = load_data("y_train_postSMOTE.pkl")

X_test = load_data("X_test_preSMOTE.pkl")
y_test = load_data("y_test_preSMOTE.pkl")


print("===== DATA GỐC =====")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)


# =========================
# LẤY DATA NHỎ ĐỂ TEST
# =========================

X_train_small, y_train_small = balanced_subset(
    X_train,
    y_train,
    max_per_class=TRAIN_PER_CLASS,
    random_state=RANDOM_STATE
)

X_test_small, y_test_small = balanced_subset(
    X_test,
    y_test,
    max_per_class=TEST_PER_CLASS,
    random_state=RANDOM_STATE
)

print("\n===== DATA TEST NHANH =====")
print("X_train_small shape:", X_train_small.shape)
print("y_train_small shape:", y_train_small.shape)
print("X_test_small shape :", X_test_small.shape)
print("y_test_small shape :", y_test_small.shape)

print("\nTrain label distribution:")
print(Counter(y_train_small))

print("\nTest label distribution:")
print(Counter(y_test_small))


# =========================
# KIỂM TRA NaN / INF
# =========================

def check_nan_inf(X, name):
    if hasattr(X, "toarray"):
        X_check = X.toarray()
    else:
        X_check = np.asarray(X)

    print(f"\n{name}:")
    print("NaN:", np.isnan(X_check).any())
    print("Inf:", np.isinf(X_check).any())


check_nan_inf(X_train_small, "X_train_small")
check_nan_inf(X_test_small, "X_test_small")


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
            proba = np.asarray(self.model.predict_proba(X))

            # Nếu predict_proba trả sai shape thì báo lỗi sớm
            expected_shape = (X.shape[0], len(self.classes_))
            if proba.shape != expected_shape:
                raise ValueError(
                    f"predict_proba trả sai shape. "
                    f"Expected {expected_shape}, got {proba.shape}"
                )

            # Chuẩn hóa nhẹ nếu tổng xác suất không đúng 1
            row_sum = proba.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            proba = proba / row_sum

            return proba

        # Fallback cho model không có predict_proba
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
# TRAIN TEST NHANH
# =========================

print("\n===== TRAIN ENSEMBLE TEST NHANH =====")

start_time = time.time()
ensemble.fit(X_train_small, y_train_small)
end_time = time.time()

print(f"\nThời gian huấn luyện test nhanh: {end_time - start_time:.2f} giây")


# =========================
# KIỂM TRA PREDICT_PROBA TỪNG MODEL
# =========================

print("\n===== KIỂM TRA PREDICT_PROBA TỪNG MODEL =====")

for name, clf in ensemble.named_estimators_.items():
    proba = clf.predict_proba(X_test_small[:10])
    print(f"{name}: proba shape = {proba.shape}")
    print(f"{name}: row sum =", proba.sum(axis=1)[:5])


# =========================
# TRAIN EVALUATION
# =========================

y_train_pred = ensemble.predict(X_train_small)
y_train_proba = ensemble.predict_proba(X_train_small)

train_accuracy = accuracy_score(y_train_small, y_train_pred)
train_loss = log_loss(y_train_small, y_train_proba, labels=ensemble.classes_)

print(f"\nTrain Accuracy test nhanh: {train_accuracy:.4f}")
print(f"Train Log Loss test nhanh : {train_loss:.4f}")


# =========================
# TEST EVALUATION
# =========================

y_pred = ensemble.predict(X_test_small)
y_proba = ensemble.predict_proba(X_test_small)

test_accuracy = accuracy_score(y_test_small, y_pred)
test_loss = log_loss(y_test_small, y_proba, labels=ensemble.classes_)

print(f"\nTest Accuracy test nhanh: {test_accuracy:.4f}")
print(f"Test Log Loss test nhanh : {test_loss:.4f}")

print("\nClassification Report test nhanh:")
print(classification_report(y_test_small, y_pred, zero_division=0))


# =========================
# GRIDSEARCH TEST NHANH - TÙY CHỌN
# =========================

if RUN_GRIDSEARCH:
    print("\n===== GRIDSEARCH TEST NHANH =====")

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
        cv=2,
        scoring="accuracy",
        verbose=1,
        n_jobs=1
    )

    start_time = time.time()
    grid_search.fit(X_train_small, y_train_small)
    end_time = time.time()

    print(f"\nThời gian GridSearchCV test nhanh: {end_time - start_time:.2f} giây")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    best_ensemble = grid_search.best_estimator_

    y_pred_gs = best_ensemble.predict(X_test_small)
    y_proba_gs = best_ensemble.predict_proba(X_test_small)

    test_accuracy_gs = accuracy_score(y_test_small, y_pred_gs)
    test_loss_gs = log_loss(y_test_small, y_proba_gs, labels=best_ensemble.classes_)

    print(f"\nTest Accuracy GridSearchCV test nhanh: {test_accuracy_gs:.4f}")
    print(f"Test Log Loss GridSearchCV test nhanh : {test_loss_gs:.4f}")

    print("\nClassification Report GridSearchCV test nhanh:")
    print(classification_report(y_test_small, y_pred_gs, zero_division=0))


# =========================
# LƯU MODEL TEST NHANH
# =========================

MODEL_DIR = Path(__file__).resolve().parent
joblib.dump(ensemble, MODEL_DIR / "ensemble_classifier_quick_test.pkl")

print("\nĐã lưu model test nhanh vào model/ensemble_classifier_quick_test.pkl")