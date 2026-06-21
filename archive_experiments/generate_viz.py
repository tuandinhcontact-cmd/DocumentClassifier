import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Tổng quan Dữ liệu Huấn luyện (Data Visualization)
File Notebook này dùng để trực quan hóa tập dữ liệu cuối cùng (`data/merged_cleaned_dataset.csv`) được sử dụng để huấn luyện mô hình 3-Step Cascade.
"""

code_imports = """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập phong cách cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
"""

code_load = """\
# 1. Tải dữ liệu
dataset_path = 'data/merged_cleaned_dataset.csv'
df = pd.read_csv(dataset_path)

# Lấp đầy giá trị NaN (nếu có) bằng chuỗi rỗng
df['cleaned_text'] = df['cleaned_text'].fillna('')

print(f"Tổng số bài báo: {len(df):,}")
print("Thông tin dữ liệu:")
df.info()

display(df.head())
"""

code_dist = """\
# 2. Phân phối các Nhãn (Categories)
category_counts = df['category'].value_counts()

plt.figure(figsize=(14, 8))
ax = sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, palette="viridis", legend=False)

# Thêm nhãn số lượng vào cuối mỗi thanh
for i, v in enumerate(category_counts.values):
    ax.text(v + 300, i + 0.1, f"{v:,}", color='black', fontweight='bold')

plt.title('Phân phối Bài báo theo Thể loại (Sự mất cân bằng dữ liệu)', fontsize=16, fontweight='bold')
plt.xlabel('Số lượng Bài báo', fontsize=12)
plt.ylabel('Thể loại', fontsize=12)
plt.tight_layout()
plt.show()

# Tính tỷ lệ phần trăm của 2 nhãn lớn nhất
top_2 = category_counts.iloc[:2].sum()
total = category_counts.sum()
print(f"Hai nhãn lớn nhất (Tech & Politics) chiếm: {top_2 / total:.2%} tổng số dữ liệu.")
"""

code_length = """\
# 3. Phân phối độ dài văn bản (Số từ)
df['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(12, 6))
sns.histplot(df['word_count'], bins=50, color='coral', kde=True)
plt.title('Phân phối Độ dài Văn bản (Số lượng từ/Bài báo)', fontsize=16, fontweight='bold')
plt.xlabel('Số từ', fontsize=12)
plt.ylabel('Tần suất', fontsize=12)
plt.xlim(0, 100) # Cắt trục X ở 100 từ để dễ nhìn vì đa số tin tức rất ngắn
plt.tight_layout()
plt.show()

print(f"Độ dài trung bình: {df['word_count'].mean():.1f} từ")
print(f"Độ dài tối đa: {df['word_count'].max():.1f} từ")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_dist),
    nbf.v4.new_code_cell(code_length)
]

with open('visualization.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Đã tạo thành công visualization.ipynb")
