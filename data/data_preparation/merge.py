import pandas as pd
from pathlib import Path


# =========================
# PATH SETUP
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "data_preparation" / "final_data1.csv"
OUTPUT_PATH = BASE_DIR / "data" / "data_preparation" / "final_data_14_custom_labels.csv"


# =========================
# LOAD DATA
# =========================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu gốc: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)
print(f"Tổng số bài báo ban đầu: {len(df)}")


# =========================
# CATEGORY MAPPING
# =========================

category_mapping = {
    # 1. Politics and society
    "POLITICS": "Politics and society",
    "IMPACT": "Politics and society",

    # 2. Lifestyle
    "TRAVEL": "Lifestyle",
    "STYLE & BEAUTY": "Lifestyle",
    "HOME & LIVING": "Lifestyle",
    "STYLE": "Lifestyle",
    "FIFTY": "Lifestyle",

    # 3. Health
    "WELLNESS": "Health",
    "HEALTHY LIVING": "Health",

    # 4. Entertainment
    "ENTERTAINMENT": "Entertainment",
    "COMEDY": "Entertainment",
    "MEDIA": "Entertainment",

    # 5. Food & drinks
    "FOOD & DRINK": "Food & drinks",
    "TASTE": "Food & drinks",

    # 6. Business
    "BUSINESS": "Business",
    "MONEY": "Business",

    # 7. Family
    "PARENTING": "Family",
    "PARENTS": "Family",
    "WEDDINGS": "Family",
    "DIVORCE": "Family",

    # 8. Community
    "QUEER VOICES": "Community",
    "BLACK VOICES": "Community",
    "LATINO VOICES": "Community",
    "WOMEN": "Community",
    "RELIGION": "Community",

    # 9. News
    "THE WORLDPOST": "News",
    "WORLDPOST": "News",
    "WORLD NEWS": "News",
    "CRIME": "News",
    "WEIRD NEWS": "News",
    "GOOD NEWS": "News",

    # 10. Sports
    "SPORTS": "Sports",

    # 11. Tech & Science
    "TECH": "Tech & Science",
    "SCIENCE": "Tech & Science",

    # 12. Environment
    "ENVIRONMENT": "Environment",
    "GREEN": "Environment",

    # 13. Arts & Culture
    "ARTS": "Arts & Culture",
    "ARTS & CULTURE": "Arts & Culture",
    "CULTURE & ARTS": "Arts & Culture",

    # 14. Education
    "EDUCATION": "Education",
    "COLLEGE": "Education",
}


# =========================
# MAP LABELS
# =========================

df["category"] = df["category"].map(category_mapping)
df = df.dropna(subset=["category"])

print(f"\nSố lượng nhãn hiện tại: {df['category'].nunique()} nhãn")


# =========================
# UNDERSAMPLING
# =========================

MAX_SAMPLES = None  # Đặt None để không giới hạn số lượng mẫu mỗi nhãn (không cắt tỉa)

if MAX_SAMPLES is not None:
    balanced_groups = []
    for category_name, group in df.groupby("category"):
        sampled_group = group.sample(
            n=min(len(group), MAX_SAMPLES),
            random_state=42
        )
        balanced_groups.append(sampled_group)
    df_balanced = pd.concat(balanced_groups)
else:
    df_balanced = df.copy()

df_balanced = df_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================
# CHECK DISTRIBUTION
# =========================

if MAX_SAMPLES is not None:
    print(f"\nPhân bố dữ liệu 14 nhãn sau khi cắt tỉa tối đa {MAX_SAMPLES} mẫu mỗi nhãn:")
else:
    print("\nPhân bố dữ liệu 14 nhãn (không giới hạn số lượng mẫu):")
print(df_balanced["category"].value_counts())

print(f"\nTổng số bài báo sau khi xử lý: {len(df_balanced)}")


# =========================
# SAVE DATA
# =========================

df_balanced.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Đã lưu thành công vào file: {OUTPUT_PATH}")