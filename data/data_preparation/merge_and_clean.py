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
    
    # 1. Đọc và tiền xử lý final_data_14_custom_labels.csv
    file1 = "final_data_14_custom_labels.csv"
    print(f"Đọc {file1}...")
    df1 = pd.read_csv(file1)
    df1['headline'] = df1['headline'].fillna('')
    df1['short_description'] = df1['short_description'].fillna('')
    df1['text_raw'] = df1['headline'] + " " + df1['short_description']
    df1_processed = pd.DataFrame({
        'cleaned_text': df1['text_raw'].apply(lambda x: clean_text(x, stop_words)),
        'category': df1['category']
    })
    
    # 2. Đọc và tiền xử lý BBC_dataset.csv (dùng encoding CP1252)
    file2 = "BBC_dataset.csv"
    print(f"Đọc {file2}...")
    df2 = pd.read_csv(file2, encoding='cp1252')
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
    file3 = "news.csv"
    print(f"Đọc {file3}...")
    df3 = pd.read_csv(file3, encoding_errors='replace')
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
    
    print(f"\nSố lượng dòng dữ liệu sau khi gộp:")
    print(f"- {file1}: {len(df1_processed)} dòng")
    print(f"- {file2}: {len(df2_processed)} dòng")
    print(f"- {file3}: {len(df3_processed)} dòng")
    print(f"- Tổng cộng: {len(merged_df)} dòng")
    
    print("\nPhân phối các danh mục (Category distribution):")
    print(merged_df['category'].value_counts())
    
    # 5. Lưu ra tệp tin mới
    output_path = "merged_cleaned_dataset.csv"
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nĐã lưu tập dữ liệu gộp sạch tại: {output_path}")

if __name__ == "__main__":
    main()
