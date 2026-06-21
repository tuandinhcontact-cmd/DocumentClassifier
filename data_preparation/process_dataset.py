import os
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Thêm danh sách stopword tiếng Việt cơ bản (nếu dữ liệu của bạn là tiếng Việt)
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "trong", "được", "cho", "với", "những", "một",
    "các", "đã", "để", "ra", "này", "về", "thì", "từ", "lên", "đến", "nhiều",
    "như", "nhưng", "khi", "lại", "vào", "ở", "nếu", "cũng", "đều", "hơn",
    "sau", "trước", "nơi", "sự", "việc", "cái", "con", "người", "ngày", "năm"
}

def clean_text(text, stop_words):
    """
    Hàm làm sạch văn bản:
    1. Chuyển về chữ thường.
    2. Loại bỏ ký tự đặc biệt và chữ số (chỉ giữ lại chữ cái và khoảng trắng).
    3. Loại bỏ stopword.
    4. Xóa khoảng trắng thừa.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Loại bỏ ký tự đặc biệt và số
    text = re.sub(r'[^a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]', ' ', text)
    
    # Tách từ và loại bỏ stopword
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    # Ghép lại thành chuỗi hoàn chỉnh
    return " ".join(cleaned_words)

def process_dataset(input_file_path, language='english'):
    print(f"Bắt đầu xử lý tệp: {input_file_path}")
    
    # 1. Kiểm tra định dạng và đọc tệp
    _, ext = os.path.splitext(input_file_path)
    ext = ext.lower()
    
    if ext == '.csv':
        df = pd.read_csv(input_file_path)
    elif ext == '.json':
        # Thử đọc dạng chuẩn hoặc json lines (orient='records')
        try:
            df = pd.read_json(input_file_path, lines=True)
        except Exception:
            df = pd.read_json(input_file_path)
    else:
        raise ValueError("Định dạng tệp không được hỗ trợ! Vui lòng dùng tệp .csv hoặc .json")
    
    print(f"Số lượng mẫu ban đầu: {len(df)}")
    
    # Kiểm tra sự tồn tại của các cột cần thiết
    required_cols = ['headline', 'short_description']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Không tìm thấy cột '{col}' trong dữ liệu! Các cột hiện có: {list(df.columns)}")
            
    # 2. Điền dữ liệu trống (NaN) bằng chuỗi rỗng
    df['headline'] = df['headline'].fillna('')
    df['short_description'] = df['short_description'].fillna('')
    
    # 3. Gộp trường headline và short_description thành một cột mới
    print("Đang gộp trường headline và short_description...")
    df['combined_text'] = df['headline'] + " " + df['short_description']
    
    # 4. Xác định danh sách stopwords
    if language.lower() == 'vietnamese':
        stop_words = VIETNAMESE_STOPWORDS
        print("Sử dụng danh sách stopword Tiếng Việt cơ bản.")
    else:
        stop_words = ENGLISH_STOP_WORDS
        print("Sử dụng danh sách stopword Tiếng Anh từ scikit-learn.")
        
    # 5. Tiến hành làm sạch văn bản
    print("Đang làm sạch stopword và các ký tự đặc biệt...")
    df['cleaned_text'] = df['combined_text'].apply(lambda x: clean_text(x, stop_words))
    
    # 6. Lưu kết quả ra file mới
    output_file_path = os.path.join(os.path.dirname(input_file_path), "cleaned_dataset.csv")
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"Đã xử lý xong! Dữ liệu sạch đã được lưu tại: {output_file_path}")
    
    return output_file_path

if __name__ == "__main__":
    # Cấu hình đường dẫn tệp đầu vào ở đây
    # Ví dụ: INPUT_FILE = "News_Category_Dataset_v3.json"
    INPUT_FILE = "final_data1.csv" 
    LANGUAGE = "english" # Hoặc "vietnamese"
    
    if os.path.exists(INPUT_FILE):
        process_dataset(INPUT_FILE, language=LANGUAGE)
    else:
        print(f"Không tìm thấy tệp {INPUT_FILE}. Vui lòng cập nhật đường dẫn chính xác trong mã nguồn.")
