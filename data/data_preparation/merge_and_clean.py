"""
merge_and_clean.py
==================
Bước 2 trong pipeline chuẩn bị dữ liệu:
  - Đọc 6 file CSV gốc từ thư mục `dataset gốc/`
  - Chuẩn hóa tên nhãn (category mapping) về 12 lớp thống nhất
  - Áp dụng capping tối đa 25,000 mẫu/nhãn (để sau khi chia 80/20 Train/Test
    mỗi lớp trong tập Train có tối đa ~20,000 mẫu)
  - Lưu kết quả ra `data/merged_dataset.csv` với 2 cột: text_raw, category

KHÔNG thực hiện lowercase / remove stopwords ở bước này.
Các bước tiền xử lý văn bản (lowercase, stopwords, TF-IDF) được thực hiện
bên trong từng script huấn luyện mô hình.

Cách chạy (từ thư mục gốc DocumentClassifier/):
    python data/data_preparation/merge_and_clean.py
"""

import os
import pandas as pd

# ─── Đường dẫn ───────────────────────────────────────────────────────────────
CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.abspath(os.path.join(CURRENT_DIR, "..", "dataset gốc"))
OUTPUT_PATH   = os.path.abspath(os.path.join(CURRENT_DIR, "..", "merged_dataset.csv"))

CAP_PER_CLASS = 20000   # Capping trước khi chia Train/Test → tập Train có tối đa ~16,000 mẫu/nhãn

# ─── Bảng ánh xạ nhãn HuffPost → 12 lớp thống nhất ─────────────────────────
HUFFPOST_MAPPING = {
    "POLITICS":        "Politics and society",
    "IMPACT":          "Politics and society",
    "THE WORLDPOST":   "Politics and society",
    "WORLDPOST":       "Politics and society",
    "WORLD NEWS":      "Politics and society",
    "CRIME":           "Politics and society",
    "TRAVEL":          "Lifestyle",
    "STYLE & BEAUTY":  "Lifestyle",
    "HOME & LIVING":   "Lifestyle",
    "STYLE":           "Lifestyle",
    "FIFTY":           "Lifestyle",
    "GOOD NEWS":       "Lifestyle",
    "WELLNESS":        "Health",
    "HEALTHY LIVING":  "Health",
    "ENTERTAINMENT":   "Entertainment",
    "COMEDY":          "Entertainment",
    "MEDIA":           "Entertainment",
    "WEIRD NEWS":      "Entertainment",
    "FOOD & DRINK":    "Food & drinks",
    "TASTE":           "Food & drinks",
    "BUSINESS":        "Business",
    "MONEY":           "Business",
    "PARENTING":       "Family",
    "PARENTS":         "Family",
    "WEDDINGS":        "Family",
    "DIVORCE":         "Family",
    "QUEER VOICES":    "Community",
    "BLACK VOICES":    "Community",
    "LATINO VOICES":   "Community",
    "WOMEN":           "Community",
    "RELIGION":        "Community",
    "SPORTS":          "Sports",
    "TECH":            "Tech & Science",
    "SCIENCE":         "Tech & Science",
    "ENVIRONMENT":     "Environment",
    "GREEN":           "Environment",
    "ARTS":            "Arts & Culture",
    "ARTS & CULTURE":  "Arts & Culture",
    "CULTURE & ARTS":  "Arts & Culture",
    "EDUCATION":       "Education",
    "COLLEGE":         "Education",
}

BBC_MAPPING = {
    "sport":         "Sports",
    "business":      "Business",
    "politics":      "Politics and society",
    "tech":          "Tech & Science",
    "entertainment": "Entertainment",
}


def read_huffpost(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "News_Category_Dataset_v3_ordered.csv")
    print(f"  [1/6] Đọc HuffPost: {path}")
    df = pd.read_csv(path)
    df["category"] = df["category"].map(HUFFPOST_MAPPING)
    df = df.dropna(subset=["category"])
    df["text_raw"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")
    result = df[["text_raw", "category"]].copy()
    print(f"        → {len(result):,} dòng | {result['category'].nunique()} nhãn")
    return result


def read_bbc(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "BBC_dataset.csv")
    print(f"  [2/6] Đọc BBC: {path}")
    df = pd.read_csv(path, encoding="cp1252")
    df["category"] = df["type"].map(BBC_MAPPING)
    df = df.dropna(subset=["category"])
    df["text_raw"] = df["news"].fillna("")
    result = df[["text_raw", "category"]].copy()
    print(f"        → {len(result):,} dòng | {result['category'].nunique()} nhãn")
    return result


def read_supplementary(raw_dir: str, filename: str, label: str, idx: int) -> pd.DataFrame:
    """Đọc các file bổ sung có cột headlines + description."""
    path = os.path.join(raw_dir, filename)
    print(f"  [{idx}/6] Đọc {label}: {path}")
    df = pd.read_csv(path)
    df["text_raw"] = df["headlines"].fillna("") + " " + df["description"].fillna("")
    df["category"] = label
    result = df[["text_raw", "category"]].copy()
    print(f"        → {len(result):,} dòng")
    return result


def main():
    print("=" * 65)
    print(" MERGE & CLEAN — Gộp 6 nguồn dữ liệu → merged_dataset.csv")
    print("=" * 65)

    # 1. Đọc từng nguồn
    dfs = [
        read_huffpost(RAW_DIR),
        read_bbc(RAW_DIR),
        read_supplementary(RAW_DIR, "education_data.csv",  "Education",      3),
        read_supplementary(RAW_DIR, "technology_data.csv", "Tech & Science",  4),
        read_supplementary(RAW_DIR, "sports_data.csv",     "Sports",          5),
        read_supplementary(RAW_DIR, "business_data.csv",   "Business",        6),
    ]

    # 2. Gộp lại
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["text_raw", "category"])
    merged = merged[merged["text_raw"].str.strip() != ""]
    print(f"\n  Tổng sau khi gộp: {len(merged):,} dòng | {merged['category'].nunique()} nhãn")

    # 3. Capping tối đa CAP_PER_CLASS mẫu/nhãn
    print(f"\n  Áp dụng capping tối đa {CAP_PER_CLASS:,} mẫu/nhãn...")
    sampled = []
    print(f"\n  {'Nhãn':<25} {'Gốc':>8} {'Sau cap':>10}")
    print("  " + "-" * 45)
    for cat, group in sorted(merged.groupby("category")):
        n = min(len(group), CAP_PER_CLASS)
        sampled.append(group.sample(n, random_state=42))
        print(f"  {cat:<25} {len(group):>8,} {n:>10,}")

    merged_capped = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n  Tổng sau khi capping: {len(merged_capped):,} dòng")

    # 4. Lưu file
    merged_capped.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\n  ✅ Đã lưu tại: {OUTPUT_PATH}")
    print("=" * 65)
    print("  Phân phối nhãn cuối cùng:")
    print(merged_capped["category"].value_counts().to_string())
    print("=" * 65)


if __name__ == "__main__":
    main()


