import os
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def main():
    print("=== Đang gộp và làm sạch các tập dữ liệu ===")
    
    stop_words = ENGLISH_STOP_WORDS
    
    # Định nghĩa thư mục hiện tại để thiết lập đường dẫn tuyệt đối
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Từ điển ánh xạ nhãn 14 lớp chuẩn
    category_mapping = {
        "POLITICS": "Politics and society",
        "IMPACT": "Politics and society",
        "TRAVEL": "Lifestyle",
        "STYLE & BEAUTY": "Lifestyle",
        "HOME & LIVING": "Lifestyle",
        "STYLE": "Lifestyle",
        "FIFTY": "Lifestyle",
        "WELLNESS": "Health",
        "HEALTHY LIVING": "Health",
        "ENTERTAINMENT": "Entertainment",
        "COMEDY": "Entertainment",
        "MEDIA": "Entertainment",
        "FOOD & DRINK": "Food & drinks",
        "TASTE": "Food & drinks",
        "BUSINESS": "Business",
        "MONEY": "Business",
        "PARENTING": "Family",
        "PARENTS": "Family",
        "WEDDINGS": "Family",
        "DIVORCE": "Family",
        "QUEER VOICES": "Community",
        "BLACK VOICES": "Community",
        "LATINO VOICES": "Community",
        "WOMEN": "Community",
        "RELIGION": "Community",
        "THE WORLDPOST": "Politics and society",
        "WORLDPOST": "Politics and society",
        "WORLD NEWS": "Politics and society",
        "CRIME": "Politics and society",
        "WEIRD NEWS": "Entertainment",
        "GOOD NEWS": "Lifestyle",
        "SPORTS": "Sports",
        "TECH": "Tech & Science",
        "SCIENCE": "Tech & Science",
        "ENVIRONMENT": "Environment",
        "GREEN": "Environment",
        "ARTS": "Arts & Culture",
        "ARTS & CULTURE": "Arts & Culture",
        "CULTURE & ARTS": "Arts & Culture",
        "EDUCATION": "Education",
        "COLLEGE": "Education",
    }
    
    # 1. Đọc và tiền xử lý News_Category_Dataset_v3_ordered.csv (gốc JSON converted)
    file1_path = os.path.abspath(os.path.join(current_dir, "News_Category_Dataset_v3_ordered.csv"))
    print(f"Đọc {file1_path}...")
    df1 = pd.read_csv(file1_path)
    
    # Ánh xạ nhãn và loại bỏ các lớp không nằm trong 14 nhãn mục tiêu
    df1['category'] = df1['category'].map(category_mapping)
    df1 = df1.dropna(subset=['category'])
    
    df1['headline'] = df1['headline'].fillna('')
    df1['short_description'] = df1['short_description'].fillna('')
    df1['text_raw'] = df1['headline'] + " " + df1['short_description']
    
    df1_processed = pd.DataFrame({
        'cleaned_text': df1['text_raw'].apply(lambda x: clean_text(x, stop_words)),
        'category': df1['category']
    })
    
    # Áp dụng giới hạn max_samples = 3500 cho 12 nhãn thiểu số của tập dữ liệu JSON, giữ nguyên Tech và Politics
    JSON_MAX_SAMPLES = 3500
    if JSON_MAX_SAMPLES is not None:
        print(f"⚠️ Áp dụng giới hạn tối đa {JSON_MAX_SAMPLES} mẫu cho 12 nhãn thiểu số (giữ nguyên Tech và Politics) cho News_Category_Dataset...")
        sampled_groups = []
        for category_name, group in df1_processed.groupby("category"):
            if category_name in ["Tech & Science", "Politics and society"]:
                sampled_groups.append(group)
            else:
                sampled_group = group.sample(
                    n=min(len(group), JSON_MAX_SAMPLES),
                    random_state=42
                )
                sampled_groups.append(sampled_group)
        df1_processed = pd.concat(sampled_groups).reset_index(drop=True)
    
    # 2. Đọc và tiền xử lý BBC_dataset.csv (dùng encoding CP1252)
    file2_path = os.path.abspath(os.path.join(current_dir, "..", "BBC_dataset.csv"))
    print(f"Đọc {file2_path}...")
    df2 = pd.read_csv(file2_path, encoding='cp1252')
    # Ánh xạ nhãn BBC
    bbc_mapping = {
        'sport': 'Sports',
        'business': 'Business',
        'politics': 'Politics and society',
        'tech': 'Tech & Science',
        'entertainment': 'Entertainment'
    }
    df2['category'] = df2['type'].map(bbc_mapping)
    df2_processed = pd.DataFrame({
        'cleaned_text': df2['news'].apply(lambda x: clean_text(x, stop_words)),
        'category': df2['category']
    })
    
    # 3. Đọc và tiền xử lý news.csv (dùng encoding_errors='replace')
    file3_path = os.path.abspath(os.path.join(current_dir, "..", "news.csv"))
    print(f"Đọc {file3_path}...")
    df3 = pd.read_csv(file3_path, encoding_errors='replace')
    # Ánh xạ nhãn news.csv
    news_mapping = {
        'politic': 'Politics and society',
        'science': 'Tech & Science',
        'technology': 'Tech & Science'
    }
    df3['category'] = df3['label'].map(news_mapping)
    df3_processed = pd.DataFrame({
        'cleaned_text': df3['text'].apply(lambda x: clean_text(x, stop_words)),
        'category': df3['category']
    })
    
    # 4. Gộp tất cả lại
    print("Đang gộp các tập dữ liệu...")
    merged_df = pd.concat([df1_processed, df2_processed, df3_processed], ignore_index=True)
    
    # Loại bỏ các dòng có text hoặc category bị trống
    merged_df = merged_df.dropna(subset=['cleaned_text', 'category'])
    merged_df = merged_df[merged_df['cleaned_text'].str.strip() != '']
    
    # Áp dụng cắt tỉa sau gộp (Post-merge Capping) cho Tech & Politics
    TECH_CAP = 40000
    POL_CAP = 40000
    print(f"⚠️ Áp dụng cắt tỉa sau gộp: Tech & Science tối đa {TECH_CAP}, Politics tối đa {POL_CAP}...")
    sampled_groups = []
    for category_name, group in merged_df.groupby("category"):
        if category_name == "Tech & Science":
            sampled_group = group.sample(n=min(len(group), TECH_CAP), random_state=42)
            sampled_groups.append(sampled_group)
        elif category_name == "Politics and society":
            sampled_group = group.sample(n=min(len(group), POL_CAP), random_state=42)
            sampled_groups.append(sampled_group)
        else:
            sampled_groups.append(group)
    merged_df = pd.concat(sampled_groups).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nSố lượng dòng dữ liệu sau khi gộp:")
    print(f"- News_Category_Dataset_v3_ordered.csv (Mapped): {len(df1_processed)} dòng")
    print(f"- BBC_dataset.csv: {len(df2_processed)} dòng")
    print(f"- news.csv (Đã làm sạch): {len(df3_processed)} dòng")
    print(f"- Tổng cộng sau khi cắt tỉa: {len(merged_df)} dòng")
    
    print("\nPhân phối các danh mục (Category distribution):")
    print(merged_df['category'].value_counts())
    
    # 5. Lưu ra tệp tin mới
    output_path = os.path.abspath(os.path.join(current_dir, "..", "merged_cleaned_dataset.csv"))
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nĐã lưu tập dữ liệu gộp sạch tại: {output_path}")

if __name__ == "__main__":
    main()
