import joblib
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

PROCESSED_DATA_PATH = CURRENT_DIR / "processed_data_w2v.csv"

MODEL_DIR = PROJECT_ROOT / "model"
W2V_MODEL_PATH = MODEL_DIR / "word2vec_embedding.model"

OUTPUT_DIR = PROJECT_ROOT / "data1" / "train_test_data"
OUTPUT_DIR.mkdir(exist_ok=True)


# Load CSV, không dùng joblib.load
df = pd.read_csv(PROCESSED_DATA_PATH)

df["tokens"] = df["tokens"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Load Word2Vec model
w2v_model = Word2Vec.load(str(W2V_MODEL_PATH))

print("Loaded processed data and Word2Vec model!")
print("Data shape:", df.shape)
print("Word2Vec vector size:", w2v_model.vector_size)


def get_average_vector(tokens, model):
    vector_size = model.vector_size

    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)


# Tạo X bằng average Word2Vec embedding
X = np.array([
    get_average_vector(tokens, w2v_model)
    for tokens in df["tokens"]
])

# Mã hóa nhãn
y = df["category"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# Lưu dữ liệu preSMOTE
joblib.dump(X_train, OUTPUT_DIR / "X_train_preSMOTE.pkl")
joblib.dump(X_test, OUTPUT_DIR / "X_test_preSMOTE.pkl")
joblib.dump(y_train, OUTPUT_DIR / "y_train_preSMOTE.pkl")
joblib.dump(y_test, OUTPUT_DIR / "y_test_preSMOTE.pkl")
joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")

print("Saved train/test data successfully!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("Output folder:", OUTPUT_DIR)