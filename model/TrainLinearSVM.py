import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# =========================
# PATH SETUP
# =========================

# File này nên đặt trong thư mục: model/TrainSVM.py
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1" / "train_test_data"
MODEL_DIR = BASE_DIR / "model"

sys.path.append(str(BASE_DIR))


# =========================
# IMPORT CUSTOM MODEL
# =========================

from Algorithms.SupportVectorMachine import SupportVectorMachine


# =========================
# CONFIG
# =========================

X_TRAIN_FILE = "X_train_postSMOTE.pkl"
Y_TRAIN_FILE = "y_train_postSMOTE.pkl"
X_TEST_FILE = "X_test_preSMOTE.pkl"
Y_TEST_FILE = "y_test_preSMOTE.pkl"

MODEL_SAVE_NAME = "custom_multiclass_svm.pkl"

# Hyperparameters cho custom multiclass SVM mini-batch
LR = 0.01
N_ITERS = 1000
C = 1.0
REG_STRENGTH = 0.0001
BATCH_SIZE = 512

# Bật nếu muốn test nhanh trên một phần nhỏ data trước
USE_SMALL_SAMPLE = False
TRAIN_SAMPLE_SIZE = 10000


# =========================
# LOAD DATA
# =========================

def load_data(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    return joblib.load(file_path)


def main():
    print("========== LOAD DATA ==========")

    X_train = load_data(X_TRAIN_FILE)
    y_train = load_data(Y_TRAIN_FILE)
    X_test = load_data(X_TEST_FILE)
    y_test = load_data(Y_TEST_FILE)

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Before sampling:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    if USE_SMALL_SAMPLE:
        print(f"\n========== USE SMALL SAMPLE: {TRAIN_SAMPLE_SIZE} ==========")
        X_train = X_train[:TRAIN_SAMPLE_SIZE]
        y_train = y_train[:TRAIN_SAMPLE_SIZE]

    print("\nAfter sampling:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\n========== SCALE DATA ==========")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data scaled successfully.")

    print("\n========== TRAIN CUSTOM MULTICLASS SVM ==========")

    model = SupportVectorMachine(
        lr=LR,
        n_iters=N_ITERS,
        C=C,
        reg_strength=REG_STRENGTH,
        batch_size=BATCH_SIZE,
        verbose=True
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"\nTraining time: {train_time:.2f} seconds")

    print("\n========== EVALUATE ==========")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n========== SAVE MODEL ==========")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_SAVE_NAME

    saved_object = {
        "model": model,
        "scaler": scaler,
        "config": {
            "lr": LR,
            "n_iters": N_ITERS,
            "C": C,
            "reg_strength": REG_STRENGTH,
            "batch_size": BATCH_SIZE,
            "x_train_file": X_TRAIN_FILE,
            "y_train_file": Y_TRAIN_FILE,
            "x_test_file": X_TEST_FILE,
            "y_test_file": Y_TEST_FILE,
        }
    }

    joblib.dump(saved_object, model_path)

    print(f"Saved model bundle to: {model_path}")


if __name__ == "__main__":
    main()