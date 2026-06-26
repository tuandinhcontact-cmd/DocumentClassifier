import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Đang tính toán phân bố dữ liệu tập data cuối ===")
    
    # 1. Đọc dữ liệu
    df_huff = pd.read_csv("data/dataset gốc/News_Category_Dataset_v3_ordered.csv")
    category_mapping = {
        "POLITICS": "Politics and society", "IMPACT": "Politics and society",
        "TRAVEL": "Lifestyle", "STYLE & BEAUTY": "Lifestyle", "HOME & LIVING": "Lifestyle",
        "STYLE": "Lifestyle", "FIFTY": "Lifestyle", "WELLNESS": "Health", "HEALTHY LIVING": "Health",
        "ENTERTAINMENT": "Entertainment", "COMEDY": "Entertainment", "MEDIA": "Entertainment",
        "FOOD & DRINK": "Food & drinks", "TASTE": "Food & drinks", "BUSINESS": "Business",
        "MONEY": "Business", "PARENTING": "Family", "PARENTS": "Family", "WEDDINGS": "Family",
        "DIVORCE": "Family", "QUEER VOICES": "Community", "BLACK VOICES": "Community",
        "LATINO VOICES": "Community", "WOMEN": "Community", "RELIGION": "Community",
        "THE WORLDPOST": "Politics and society", "WORLDPOST": "Politics and society",
        "WORLD NEWS": "Politics and society", "CRIME": "Politics and society",
        "WEIRD NEWS": "Entertainment", "GOOD NEWS": "Lifestyle", "SPORTS": "Sports",
        "TECH": "Tech & Science", "SCIENCE": "Tech & Science", "ENVIRONMENT": "Environment",
        "GREEN": "Environment", "ARTS": "Arts & Culture", "ARTS & CULTURE": "Arts & Culture",
        "CULTURE & ARTS": "Arts & Culture", "EDUCATION": "Education", "COLLEGE": "Education",
    }
    df_huff['category'] = df_huff['category'].map(category_mapping)
    df_huff = df_huff.dropna(subset=['category'])
    df_huff['text_raw'] = df_huff['headline'].fillna('') + " " + df_huff['short_description'].fillna('')
    df_huff_processed = pd.DataFrame({
        'category': df_huff['category']
    })
    
    df_bbc = pd.read_csv("data/dataset gốc/BBC_dataset.csv", encoding='cp1252')
    bbc_mapping = {
        'sport': 'Sports', 'business': 'Business', 'politics': 'Politics and society',
        'tech': 'Tech & Science', 'entertainment': 'Entertainment'
    }
    df_bbc['category'] = df_bbc['type'].map(bbc_mapping)
    df_bbc_processed = pd.DataFrame({
        'category': df_bbc['category']
    })

    df_edu = pd.read_csv("data/dataset gốc/education_data.csv")
    df_edu_processed = pd.DataFrame({
        'category': ['Education'] * len(df_edu)
    })

    df_tech = pd.read_csv("data/dataset gốc/technology_data.csv")
    df_tech_processed = pd.DataFrame({
        'category': ['Tech & Science'] * len(df_tech)
    })

    df_sports = pd.read_csv("data/dataset gốc/sports_data.csv")
    df_sports_processed = pd.DataFrame({
        'category': ['Sports'] * len(df_sports)
    })

    df_business = pd.read_csv("data/dataset gốc/business_data.csv")
    df_business_processed = pd.DataFrame({
        'category': ['Business'] * len(df_business)
    })

    # Gộp tự nhiên
    merged_raw = pd.concat([
        df_huff_processed, 
        df_bbc_processed, 
        df_edu_processed, 
        df_tech_processed,
        df_sports_processed,
        df_business_processed
    ], ignore_index=True)
    merged_raw = merged_raw.dropna(subset=['category'])
    
    # Tính phân bố tự nhiên (Uncapped)
    dist_uncapped = merged_raw['category'].value_counts()
    
    # Tính phân bố có Capping 20k
    sampled = []
    for cat, group in merged_raw.groupby('category'):
        n = min(len(group), 20000)
        sampled.append(group.sample(n, random_state=42))
    merged_capped = pd.concat(sampled).reset_index(drop=True)
    dist_capped = merged_capped['category'].value_counts()
    
    # Tạo DataFrame để vẽ biểu đồ
    df_plot = pd.DataFrame({
        'Tự nhiên (Uncapped)': dist_uncapped,
        'Capping 20k': dist_capped
    }).fillna(0)
    
    # Sắp xếp theo số lượng Capping 20k tăng dần
    df_plot = df_plot.sort_values(by='Capping 20k', ascending=True)
    
    # Thiết lập giao diện biểu đồ
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Vẽ cột ngang (chỉ vẽ phân phối Capping 20k)
    categories = df_plot.index
    y_indices = np.arange(len(categories))
    height = 0.6
    
    rects = ax.barh(y_indices, df_plot['Capping 20k'], height, label='Giới hạn Capping 20,000', color='#E68A8A')
    
    # Thiết lập nhãn trục và tiêu đề
    ax.set_title('Phân bố các danh mục dữ liệu sau bước chuẩn bị (Capping tối đa 20,000 mẫu)\n(HuffPost + BBC + Education + Tech + Sports + Business)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Số lượng mẫu (Documents)', fontsize=12, labelpad=10)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(categories, fontsize=11, fontweight='medium')
    ax.legend(fontsize=11, loc='lower right')
    
    # Thêm số lượng trực tiếp lên cột
    for rect in rects:
        w = rect.get_width()
        if w > 0:
            ax.text(w + 300, rect.get_y() + rect.get_height()/2, f'{int(w):,}', 
                    ha='left', va='center', fontsize=10, color='#B22222', fontweight='semibold')
            
    plt.tight_layout()
    
    # Lưu biểu đồ
    artifacts_dir = "/Users/gtuan/.gemini/antigravity/brain/e5d3d9d4-10f5-4a38-b541-628e0af1cf37"
    out_path = os.path.join(artifacts_dir, "final_dataset_distribution.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ Đã vẽ và lưu biểu đồ tại: {out_path}")
    
    # Lưu thêm 1 bản trong thư mục static của flask để UI web có thể dùng nếu muốn
    static_out_path = "static/final_dataset_distribution.png"
    plt.savefig(static_out_path, dpi=300)
    print(f"✅ Đã lưu bản phụ tại: {static_out_path}")

if __name__ == "__main__":
    main()
