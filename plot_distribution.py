import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_cleaned_dataset.csv')
counts = df['category'].value_counts().sort_values(ascending=True)

plt.figure(figsize=(12, 8))
# Create horizontal bar chart
bars = plt.barh(counts.index, counts.values, color='skyblue')
plt.title('Distribution of Categories in the Dataset', fontsize=16)
plt.xlabel('Number of Articles', fontsize=14)
plt.ylabel('Category', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 100, bar.get_y() + bar.get_height()/2, 
             f'{int(width):,}', 
             va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('category_distribution.png', dpi=300)
print("Plot saved as category_distribution.png")
