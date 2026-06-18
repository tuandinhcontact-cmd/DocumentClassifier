import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# PATH SETUP
# =========================

# File này nên đặt trong thư mục: model/TrainTFIDFLinearSVC.py
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data1"
MODEL_DIR = BASE_DIR / "model"

sys.path.append(str(BASE_DIR))


# =========================
# CONFIG
# =========================

# Ưu tiên dùng file 14 labels đã xử lý
CSV_FILE = DATA_DIR / "data_preparation" / "final_data_14_custom_labels.csv"

MODEL_SAVE_NAME = "tfidf_linear_svc.pkl"
REPORT_SAVE_NAME = "tfidf_linear_svc_report.txt"

TEXT_COLUMNS = ["headline", "short_description"]
LABEL_COLUMN = "category"

TEST_SIZE = 0.15
RANDOM_STATE = 42

# TF-IDF config
MAX_FEATURES = 100000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
MAX_DF = 0.90
SUBLINEAR_TF = True

# LinearSVC config
C = 1.0
MAX_ITER = 5000
CLASS_WEIGHT = "balanced"  # đổi thành None nếu không muốn cân bằng class


# =========================
# LOAD DATA
# =========================

def load_dataframe():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    missing_text_cols = [col for col in TEXT_COLUMNS if col not in df.columns]
    if missing_text_cols:
        raise ValueError(
            f"Thiếu cột text: {missing_text_cols}. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Thiếu cột label '{LABEL_COLUMN}'. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    return df


def build_text_column(df):
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str)

    df["text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1).str.strip()

    # Bỏ các dòng text rỗng hoặc label rỗng
    df = df[df["text"] != ""]
    df = df[df[LABEL_COLUMN].notna()]

    return df


def main():
    print("========== LOAD CSV DATA ==========")

    df = load_dataframe()
    df = build_text_column(df)

    X = df["text"].values
    y = df[LABEL_COLUMN].values

    print("Dataset size:", len(df))
    print("Number of classes:", df[LABEL_COLUMN].nunique())
    print("\nClass distribution:")
    print(df[LABEL_COLUMN].value_counts())

    print("\n========== TRAIN TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("X_train size:", len(X_train))
    print("X_test size:", len(X_test))

    print("\n========== BUILD PIPELINE ==========")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            max_df=MAX_DF,
            sublinear_tf=SUBLINEAR_TF,
            lowercase=True,
            stop_words="english"
        )),
        ("svc", LinearSVC(
            C=C,
            max_iter=MAX_ITER,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE
        ))
    ])

    print("\n========== TRAIN TF-IDF + LINEAR SVC ==========")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training time: {train_time:.2f} seconds")

    print("\n========== EVALUATE ==========")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    print("\n========== SAVE MODEL ==========")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_SAVE_NAME
    report_path = MODEL_DIR / REPORT_SAVE_NAME

    saved_object = {
        "model": model,
        "config": {
            "csv_file": str(CSV_FILE),
            "text_columns": TEXT_COLUMNS,
            "label_column": LABEL_COLUMN,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "max_features": MAX_FEATURES,
            "ngram_range": NGRAM_RANGE,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "sublinear_tf": SUBLINEAR_TF,
            "C": C,
            "max_iter": MAX_ITER,
            "class_weight": CLASS_WEIGHT,
            "accuracy": acc,
            "train_time": train_time
        }
    }

    joblib.dump(saved_object, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TF-IDF + LinearSVC Report\n")
        f.write("=========================\n\n")
        f.write(f"CSV file: {CSV_FILE}\n")
        f.write(f"Dataset size: {len(df)}\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"Saved model bundle to: {model_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
