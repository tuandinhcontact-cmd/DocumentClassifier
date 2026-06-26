import pandas as pd
import re

def main():
    file_path = "data/news.csv"
    df = pd.read_csv(file_path, encoding_errors='replace')
    df['text'] = df['text'].fillna('')
    
    print("=== Phân tích dòng trống & mẫu phi ASCII ===")
    
    # 1. Dòng trống hoặc chỉ chứa khoảng trắng
    empty_mask = df['text'].str.strip() == ''
    empty_count = empty_mask.sum()
    print(f"- Số dòng rỗng hoặc chỉ chứa khoảng trắng: {empty_count}")
    
    # 2. Các dòng trùng lặp có nội dung (không trống)
    non_empty_df = df[~empty_mask]
    non_empty_df['text_lower'] = non_empty_df['text'].str.lower().str.strip()
    duplicates = non_empty_df[non_empty_df.duplicated(subset=['text_lower'], keep=False)].sort_values(by='text_lower')
    print(f"- Số dòng trùng lặp có nội dung: {len(duplicates)}")
    
    print("\nVí dụ các dòng trùng lặp có nội dung (top 6):")
    printed = 0
    for text_val, group in duplicates.groupby('text_lower'):
        if len(text_val) > 10: # Chỉ in dòng có độ dài tương đối
            print(f"Nội dung trùng ({len(group)} lần): {group['text'].iloc[0][:100]}...")
            print(f"Nhãn của nhóm này: {group['label'].tolist()}")
            printed += 1
            if printed >= 4:
                break

    # 3. Phân tích ký tự non-ascii
    print("\nMột số ví dụ chứa ký tự phi ASCII:")
    non_ascii_mask = df['text'].apply(lambda x: bool(re.search(r'[^\x00-\x7F]', str(x))))
    non_ascii_samples = df[non_ascii_mask].head(5)
    for idx, row in non_ascii_samples.iterrows():
        print(f"[{row['label']}]: {row['text'][:120]}...")
        # Tìm các ký tự non-ascii cụ thể
        non_ascii_chars = re.findall(r'[^\x00-\x7F]', row['text'])
        print(f"  -> Các ký tự phi ASCII: {list(set(non_ascii_chars))}")

if __name__ == "__main__":
    main()
