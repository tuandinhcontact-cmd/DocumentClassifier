import pandas as pd

# 1. Đọc dữ liệu gốc (File chứa 191k dòng ban đầu)
df = pd.read_csv('data1/preprocess/final_data1.csv')
print(f"Tổng số bài báo ban đầu: {len(df)}")

# 2. Từ điển ánh xạ toàn bộ nhãn gốc về 14 nhãn của bạn
category_mapping = {
    # 1. Politics and society
    'POLITICS': 'Politics and society',
    'IMPACT': 'Politics and society',
    
    # 2. Lifestyle
    'TRAVEL': 'Lifestyle',
    'STYLE & BEAUTY': 'Lifestyle',
    'HOME & LIVING': 'Lifestyle',
    'STYLE': 'Lifestyle',
    'FIFTY': 'Lifestyle',
    
    # 3. Health
    'WELLNESS': 'Health',
    'HEALTHY LIVING': 'Health',
    
    # 4. Entertainment
    'ENTERTAINMENT': 'Entertainment',
    'COMEDY': 'Entertainment',
    'MEDIA': 'Entertainment',
    
    # 5. Food & drinks
    'FOOD & DRINK': 'Food & drinks',
    'TASTE': 'Food & drinks',
    
    # 6. Business
    'BUSINESS': 'Business',
    'MONEY': 'Business',
    
    # 7. Family
    'PARENTING': 'Family',
    'PARENTS': 'Family',
    'WEDDINGS': 'Family',
    'DIVORCE': 'Family',
    
    # 8. Community
    'QUEER VOICES': 'Community',
    'BLACK VOICES': 'Community',
    'LATINO VOICES': 'Community',
    'WOMEN': 'Community',
    'RELIGION': 'Community',
    
    # 9. News
    'THE WORLDPOST': 'News',
    'WORLDPOST': 'News',
    'WORLD NEWS': 'News',
    'CRIME': 'News',
    'WEIRD NEWS': 'News',
    'GOOD NEWS': 'News',
    
    # 10. Sports
    'SPORTS': 'Sports',
    
    # 11. Tech & Science
    'TECH': 'Tech & Science',
    'SCIENCE': 'Tech & Science',
    
    # 12. Environment
    'ENVIRONMENT': 'Environment',
    'GREEN': 'Environment',
    
    # 13. Arts & Culture
    'ARTS': 'Arts & Culture',
    'ARTS & CULTURE': 'Arts & Culture',
    'CULTURE & ARTS': 'Arts & Culture',
    
    # 14. Education
    'EDUCATION': 'Education',
    'COLLEGE': 'Education'
}

# 3. Ghi đè nhãn mới bằng hàm map()
# Những nhãn không có trong từ điển (nếu có) sẽ bị biến thành NaN và xóa đi
df['category'] = df['category'].map(category_mapping)
df = df.dropna(subset=['category'])

print(f"\nSố lượng nhãn hiện tại: {df['category'].nunique()} nhãn")

# 4. Cân bằng dữ liệu (Cắt tỉa Undersampling)
# ==========================================
MAX_SAMPLES = 11000 # Cắt tỉa ở mức 11.000 như bạn đang cấu hình

# Tạo một danh sách rỗng để chứa dữ liệu của từng nhãn sau khi cắt
balanced_groups = []

# Đi qua từng nhóm nhãn, nhãn nào > 11000 thì cắt, < 11000 thì lấy hết
for category_name, group in df.groupby('category'):
    sampled_group = group.sample(n=min(len(group), MAX_SAMPLES), random_state=42)
    balanced_groups.append(sampled_group)

# Gộp tất cả các mảnh nhỏ lại thành một bảng lớn hoàn chỉnh
df_balanced = pd.concat(balanced_groups)

# Xáo trộn lại dữ liệu (Rất quan trọng)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nPhân bố dữ liệu 14 nhãn sau khi cắt tỉa (Tối đa 11000):")
print(df_balanced['category'].value_counts())

# 5. Lưu ra file CSV cuối cùng để đem đi Train
df_balanced.to_csv('final_data_14_custom_labels.csv', index=False)
print("\n✅ Đã lưu thành công vào file: final_data_14_custom_labels.csv")