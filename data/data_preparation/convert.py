import json
import csv
import os

def is_valid_article(item):
    """
    Hàm kiểm duyệt chất lượng từng bài báo trước khi đưa vào CSV.
    """
    # 1. Lấy dữ liệu và dùng .strip() để dọn sạch khoảng trắng dư thừa
    headline = item.get("headline", "").strip()
    short_desc = item.get("short_description", "").strip()
    category = item.get("category", "").strip()

    # 2. Loại bỏ các bài báo bị khuyết hoặc chỉ chứa toàn dấu cách
    if not headline or not short_desc or not category:
        return False
    
    return True

def convert_json_to_csv(json_file_path, csv_file_path, max_samples=None):
    """
    Chuyển đổi từ file JSONL gốc sang file CSV sạch.
    Chỉ giữ lại 3 cột thiết yếu: headline, short_description, category.
    """
    try:
        data = []
        total_original_lines = 0
        
        # Mở và đọc file JSONL từng dòng một
        with open(json_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    total_original_lines += 1
                    item = json.loads(line.strip())
                    
                    # Nếu bài báo vượt qua vòng kiểm duyệt, thì thêm vào mảng data
                    if is_valid_article(item):
                        # Gói gọn đúng 3 cột cần thiết
                        clean_item = {
                            "headline": item.get("headline", "").strip(),
                            "short_description": item.get("short_description", "").strip(),
                            "category": item.get("category", "").strip()
                        }
                        data.append(clean_item)
        
        if not data:
            print("❌ Lỗi: Không tìm thấy dữ liệu hợp lệ sau khi lọc.")
            return False
            
        # Giới hạn số lượng mẫu nếu max_samples không phải là None
        if max_samples is not None:
            print(f"⚠️ Giới hạn số lượng mẫu tối đa: {max_samples:,}")
            data = data[:max_samples]
            
        # In báo cáo chi tiết ra Terminal
        print("-" * 50)
        print("📊 BÁO CÁO KẾT QUẢ TIỀN XỬ LÝ (PREPROCESSING):")
        print(f"Tổng số bài báo gốc đọc được: {total_original_lines:,}")
        print(f"Số bài báo rác đã bị xóa: {total_original_lines - len(data):,}")
        print(f"Số lượng bài báo SẠCH giữ lại: {len(data):,}")
        print("-" * 50)
        
        # Chỉ định chính xác 3 trường cần xuất ra CSV
        field_names = ["headline", "short_description", "category"]
        
        # Ghi file CSV
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as file:
            # Tham số extrasaction='ignore' rất quan trọng để bỏ qua các key thừa trong dictionary
            writer = csv.DictWriter(file, fieldnames=field_names, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        
        print(f"✅ Hoàn thành! File dữ liệu sạch đã được lưu tại:\n👉 {csv_file_path}")
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi giải mã JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
        return False

def main():
    # Sử dụng đường dẫn tuyệt đối để chống lỗi "File not found" khi chạy ở Terminal khác thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_file_path = os.path.abspath(os.path.join(current_dir, "..", "dataset gốc", "News_Category_Dataset_v3_ordered.json"))
    csv_file_path = os.path.join(current_dir, "News_Category_Dataset_v3_ordered.csv") 
    
    print(f"Đang đọc dữ liệu từ: {json_file_path} ...")
    
    if not os.path.exists(json_file_path):
        print(f"❌ Lỗi: Không tìm thấy file gốc tại đường dẫn:\n{json_file_path}")
        return
    
    # Cấu hình giới hạn số lượng mẫu (None nghĩa là không giới hạn)
    MAX_SAMPLES = None
    convert_json_to_csv(json_file_path, csv_file_path, max_samples=MAX_SAMPLES)

if __name__ == "__main__":
    main()