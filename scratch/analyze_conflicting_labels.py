import pandas as pd

def main():
    file_path = "data/news.csv"
    df = pd.read_csv(file_path, encoding_errors='replace')
    df['text'] = df['text'].fillna('').str.strip()
    
    # Lọc bỏ dòng trống
    df = df[df['text'] != '']
    
    # Tìm trùng lặp nội dung
    # Gom nhóm theo text viết thường và lấy tập hợp nhãn duy nhất cho mỗi nhóm
    df['text_lower'] = df['text'].str.lower()
    
    grouped = df.groupby('text_lower')['label'].nunique()
    conflicting_groups = grouped[grouped > 1]
    
    print(f"=== Phân tích Mâu thuẫn Nhãn ===")
    print(f"Tổng số nhóm nội dung bị mâu thuẫn nhãn (cùng 1 text nhưng nhãn khác nhau): {len(conflicting_groups)}")
    
    # Tính số dòng bị ảnh hưởng
    conflicting_texts = conflicting_groups.index
    conflicting_rows = df[df['text_lower'].isin(conflicting_texts)]
    print(f"Tổng số dòng bị ảnh hưởng: {len(conflicting_rows)}")
    
    # In một số ví dụ chi tiết
    print("\nChi tiết một số dòng bị mâu thuẫn nhãn:")
    count = 0
    for text_val in conflicting_texts[:10]:
        rows = df[df['text_lower'] == text_val]
        print(f"\nNội dung (Độ dài: {len(text_val)} ký tự): {rows['text'].iloc[0]}")
        print(f"Các dòng tương ứng trong file gốc:")
        for idx, row in rows.iterrows():
            print(f"  - Dòng {idx}: Nhãn = '{row['label']}'")
        count += 1
        
if __name__ == "__main__":
    main()
