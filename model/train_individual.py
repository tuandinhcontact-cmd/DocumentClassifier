import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1" / "train_test_data"
SAVE_DIR = BASE_DIR / "model" / "individual_models"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR))


# =========================
# IMPORT CUSTOM MODELS
# =========================

from Algorithms.GaussianNB import GaussianNB
from Algorithms.LogisticRegression import LogisticRegression
from Algorithms.RandomForest import RandomForest
from Algorithms.SupportVectorMachine import SupportVectorMachine


# =========================
# LOAD DATA
# =========================

def load_pkl(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    return joblib.load(path)


print("Loading data...")

X_train = load_pkl("X_train_postSMOTE.pkl")
y_train = load_pkl("y_train_postSMOTE.pkl")

X_test = load_pkl("X_test_preSMOTE.pkl")
y_test = load_pkl("y_test_preSMOTE.pkl")

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_test shape : {y_test.shape}")


# =========================
# DEFINE MODELS
# =========================

models = {
    "gaussian_nb": GaussianNB(),

    "logistic_regression": LogisticRegression(),

    "support_vector_machine": SupportVectorMachine(),

    "random_forest": RandomForest(
        n_trees=10,
        max_depth=10,
        min_samples_split=2,
        sample_size=3000
    )
}


# =========================
# TRAIN - EVALUATE - SAVE
# =========================

results = {}

for model_name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training model: {model_name}")
    print("=" * 60)

    start_time = time.time()

    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    print(f"Training done in {train_time:.2f} seconds")

    print(f"Predicting with {model_name}...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy of {model_name}: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_path = SAVE_DIR / f"{model_name}.pkl"

    joblib.dump(model, save_path)

    print(f"Saved model to: {save_path}")

    results[model_name] = {
        "accuracy": acc,
        "train_time": train_time,
        "save_path": str(save_path)
    }


# =========================
# SAVE RESULTS
# =========================

results_path = SAVE_DIR / "training_results.pkl"
joblib.dump(results, results_path)

print("\n" + "=" * 60)
print("ALL MODELS TRAINED AND SAVED")
print("=" * 60)

for model_name, info in results.items():
    print(
        f"{model_name}: "
        f"accuracy={info['accuracy']:.4f}, "
        f"time={info['train_time']:.2f}s, "
        f"file={info['save_path']}"
    )

print(f"\nSaved training results to: {results_path}")