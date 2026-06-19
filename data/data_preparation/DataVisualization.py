import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "final_data_14_custom_labels.csv"

df = pd.read_csv(file_path)

df.info()


# Category distribution
category_counts = df['category'].value_counts()

# Plot category distribution
plt.figure(figsize=(15, 10))
ax = sns.barplot(
    x=category_counts.index,
    y=category_counts.values,
    palette='viridis'
)

plt.title('Distribution of News Categories', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=90)

# Add count labels on top of each bar
for i, count in enumerate(category_counts.values):
    ax.text(i, count + 100, f"{count:,}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Calculate category percentages
category_percentages = (category_counts / len(df) * 100).reset_index()
category_percentages.columns = ['Category', 'Percentage']

print(category_percentages.sort_values('Percentage', ascending=False).head(10))

# ==============================
# Category distribution AFTER SMOTE
# ==============================

PROJECT_ROOT = BASE_DIR.parent.parent

TRAIN_TEST_DIR = PROJECT_ROOT / "data1" / "train_test_data"
MODEL_DIR = PROJECT_ROOT / "model"

y_train_smote_path = TRAIN_TEST_DIR / "y_train_postSMOTE.pkl"
label_encoder_path = MODEL_DIR / "label_encoder.pkl"

# Load y_train sau SMOTE và label encoder
y_train_smote = joblib.load(y_train_smote_path)
label_encoder = joblib.load(label_encoder_path)

# Chuyển nhãn số về tên category
y_train_smote_labels = label_encoder.inverse_transform(y_train_smote)

# Đếm số lượng category sau SMOTE
smote_category_counts = pd.Series(y_train_smote_labels).value_counts()

# Plot category distribution after SMOTE
plt.figure(figsize=(15, 10))
ax = sns.barplot(
    x=smote_category_counts.index,
    y=smote_category_counts.values,
    palette="viridis"
)

plt.title("Distribution of News Categories After SMOTE", fontsize=16)
plt.xlabel("Category", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=90)

# Add count labels on top of each bar
for i, count in enumerate(smote_category_counts.values):
    ax.text(i, count + 100, f"{count:,}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()

# Calculate category percentages after SMOTE
smote_category_percentages = (
    smote_category_counts / len(y_train_smote) * 100
).reset_index()

smote_category_percentages.columns = ["Category", "Percentage"]

print("Top categories after SMOTE:")
print(smote_category_percentages.sort_values("Percentage", ascending=False).head(10))
