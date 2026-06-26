import os
import pandas as pd

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(current_dir, "..", "data", "dataset gốc", "News_Category_Dataset_v3_ordered.csv"))
    
    if not os.path.exists(csv_path):
        print(f"Error: file not found at {csv_path}")
        return

    print("=== Loading HuffPost dataset ===")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    print("\n=== Category distribution in original HuffPost data (top 42 categories) ===")
    orig_dist = df['category'].value_counts()
    print(orig_dist)
    print(f"Total categories: {len(orig_dist)}")
    
    # Label mapping (copied from merge_and_clean.py)
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
    
    # 1. Distribution after mapping (without dropping unmapped yet)
    df['mapped_category'] = df['category'].map(category_mapping)
    
    print("\n=== Unmapped categories (will be dropped) ===")
    unmapped = df[df['mapped_category'].isna()]['category'].value_counts()
    print(unmapped)
    
    df_mapped = df.dropna(subset=['mapped_category'])
    
    print(f"\n=== Mapped Category distribution (13 target categories) BEFORE capping ===")
    mapped_dist = df_mapped['mapped_category'].value_counts()
    print(mapped_dist)
    print(f"Total rows after mapping (before capping): {len(df_mapped)}")

    print("\n=== Mapped Category distribution (13 target categories) AFTER capping (JSON_MAX_SAMPLES = 3500) ===")
    # Apply selective capping: 3500 for minority classes, keep Tech & Science and Politics and society
    sampled_groups = []
    for cat_name, group in df_mapped.groupby("mapped_category"):
        if cat_name in ["Tech & Science", "Politics and society"]:
            sampled_groups.append(group)
        else:
            sampled_group = group.sample(n=min(len(group), 3500), random_state=42)
            sampled_groups.append(sampled_group)
    df_capped = pd.concat(sampled_groups)
    capped_dist = df_capped['mapped_category'].value_counts()
    print(capped_dist)
    print(f"Total rows after capping: {len(df_capped)}")

if __name__ == "__main__":
    main()
