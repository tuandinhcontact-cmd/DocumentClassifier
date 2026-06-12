import re
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec

# DocumentClassifier_git/data1/data_preparation/DataPreparation.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
DATA_PATH = CURRENT_DIR / "final_data_14_custom_labels.csv"

# Đường dẫn file stopwords
STOPWORDS_PATH = PROJECT_ROOT / "data1" / "preprocess" / "stop_words.txt"

# Thư mục lưu model
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Đọc dữ liệu thành công!")
print(df.head())
print(df.columns)

with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    stopwords = set([line.strip() for line in f.readlines()])

print(f"Số lượng stopwords: {len(stopwords)}")

def clean_text(text):
    if pd.isna(text):
        text = ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]

    return tokens

df["headline"] = df["headline"].fillna("")
df["short_description"] = df["short_description"].fillna("")

df["combined"] = df["headline"] + " " + df["short_description"]
df["tokens"] = df["combined"].apply(clean_text)

print("Xử lý text xong!")
print(df[["headline", "short_description", "combined", "tokens"]].head())
df = df[df["tokens"].apply(len) > 0]

print(f"Số dòng còn lại sau khi xử lý: {len(df)}")

w2v_model = Word2Vec(
    sentences=df["tokens"],
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    sg=1
)


w2v_model.save(str(MODEL_DIR / "word2vec_embedding.model"))

print("Đã lưu Word2Vec model tại:")
print(MODEL_DIR / "word2vec_embedding.model")
OUTPUT_PATH = CURRENT_DIR / "processed_data_w2v.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("Đã lưu dữ liệu đã xử lý tại:")
print(OUTPUT_PATH)