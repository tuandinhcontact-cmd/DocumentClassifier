import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Đang vẽ biểu đồ phân bố danh mục ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "merged_cleaned_dataset.csv"))
    output_image_path = os.path.abspath(os.path.join(current_dir, "..", "merged_cleaned_dataset_distribution.png"))
    
    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Không tìm thấy tập dữ liệu tại: {dataset_path}")
        return
        
    # 1. Đọc dữ liệu
    df = pd.read_csv(dataset_path)
    
    # 2. Thiết lập giao diện biểu đồ
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 3. Lấy dữ liệu phân bố và sắp xếp
    category_counts = df['category'].value_counts()
    
    # 4. Vẽ biểu đồ cột ngang (Horizontal Bar Plot)
    colors = sns.color_palette("viridis", len(category_counts))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette=colors)
    
    # 5. Thêm nhãn số liệu cụ thể trên từng cột
    for index, value in enumerate(category_counts.values):
        plt.text(value + (max(category_counts.values) * 0.01), index, f"{value:,}", va='center', fontweight='bold', fontsize=10)
        
    plt.title(f"Phân bố các danh mục dữ liệu (Tổng cộng: {len(df):,} dòng)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Số lượng bản ghi (Items)", fontsize=12, labelpad=10)
    plt.ylabel("Danh mục (Category)", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    # 6. Lưu biểu đồ dưới dạng hình ảnh
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    
    print(f"✅ Đã lưu thành công biểu đồ tại:\n👉 {output_image_path}")

if __name__ == "__main__":
    main()
