import sys
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1" / "train_test_data"
MODEL_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR))


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


# =========================
# ENSURE NUMPY FLOAT64
# =========================

X_train = np.ascontiguousarray(X_train, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


print("===== DATA =====")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

print("\nNaN in X_train:", np.isnan(X_train).any())
print("Inf in X_train:", np.isinf(X_train).any())
print("NaN in X_test :", np.isnan(X_test).any())
print("Inf in X_test :", np.isinf(X_test).any())


# =========================
# BUILD SKLEARN MULTICLASS SVM
# =========================

svm_model = make_pipeline(
    StandardScaler(),
    LinearSVC(
        C=1.0,
        loss="squared_hinge",
        dual=False,
        multi_class="ovr",
        max_iter=10000,
        random_state=42
    )
)


# =========================
# TRAIN
# =========================

print("\n===== TRAIN SKLEARN LINEAR SVM =====")

start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()

print(f"\nThời gian huấn luyện LinearSVC: {end_time - start_time:.2f} giây")


# =========================
# TRAIN EVALUATION
# =========================

print("\n===== TRAIN EVALUATION =====")

y_train_pred = svm_model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")


# =========================
# TEST EVALUATION
# =========================

print("\n===== TEST EVALUATION =====")

y_pred = svm_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# SAVE MODEL
# =========================

model_path = MODEL_DIR / "sklearn_linear_svm_multiclass.pkl"
joblib.dump(svm_model, model_path)

print(f"\nĐã lưu sklearn LinearSVC model vào: {model_path}")