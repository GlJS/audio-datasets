import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import matplotlib.cm as cm
import json
import numpy as np

# Load and parse JSON to get category groupings
def get_category_groups():
    with open('visualization/1.json') as f:
        data = json.load(f)
    
    groups = {
        "Natural sounds": [],
        "Human sounds": [],
        "Sounds of things": [],
        "Source-ambiguous sounds": [],
        "Animal": [], 
        "Channel, environment and background": [],
        "Music": [],
        
    }
    
    def get_subcategories(item):
        subcats = [item['name']]
        for child_id in item['child_ids']:
            child = next(x for x in data if x['id'] == child_id)
            subcats.extend(get_subcategories(child))
        return subcats
        
    for item in data:
        if item['name'] in groups.keys():
            groups[item['name']] = get_subcategories(item)
            
    return groups

# Load all category CSV files
dfs = []
for f in os.listdir("output_cats/"):
    if f.endswith(".csv"):
        df = pd.read_csv(f"output_cats/{f}")
        dfs.append(df)
df = pd.concat(dfs)

# Drop rows with NA labels
df = df.dropna(subset=['label'])

# Fix "Animal " label 
df['label'] = df['label'].replace('Animal ', 'Animal')

df["dataset"] = df["dataset"].replace("RichDetailAudioTextSimulation", "RDATS")

# Recompute counts after cleaning
dataset_category_counts = df.groupby(['label', 'dataset']).size().unstack(fill_value=0)

# Get unique categories and datasets
categories = df['label'].unique()
# Get datasets without sorting by size
datasets = dataset_category_counts.columns.tolist()
# Get sorted datasets for legend only
datasets_sorted = dataset_category_counts.sum().sort_values(ascending=False).index

# Get category groupings
groups = get_category_groups()

# Create mapping of categories to groups
cat_to_group = {}
for group, cats in groups.items():
    for cat in cats:
        cat_to_group[cat] = group

# Calculate total counts per category
total_counts = dataset_category_counts.sum(axis=1)

# Sort categories by groups and counts within groups
sorted_cats = []
for group in groups:
    group_cats = [c for c in categories if cat_to_group.get(c) == group]
    # Sort group categories by their total counts in descending order
    group_cats_sorted = sorted(group_cats, key=lambda x: total_counts[x], reverse=True)
    sorted_cats.extend(group_cats_sorted)

# Create extended color palette for datasets (at least 58 colors)
colors = []
colors.extend(plt.cm.Pastel1(np.linspace(0, 1, 9)))  # 9 pastel colors
colors.extend(plt.cm.Pastel2(np.linspace(0, 1, 8)))  # 8 more pastel colors
colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))   # 20 vibrant colors
colors.extend(plt.cm.tab20b(np.linspace(0, 1, 20)))  # 20 more vibrant colors
colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))    # 12 medium intensity colors
colors.extend(plt.cm.Set2(np.linspace(0, 1, 8)))     # 8 medium intensity colors
dataset_colors = {dataset: colors[i] for i, dataset in enumerate(datasets)}

group_cmap = cm.get_cmap('Dark2', len(groups))
group_colors = {group: group_cmap(i) for i, group in enumerate(groups.keys())}

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(20, 10))

# Remove plot border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Plot stacked bars
bottom = pd.Series(0, index=sorted_cats)
for dataset in datasets:
    if dataset is not None:  # Skip None/missing datasets
        values = dataset_category_counts[dataset]
        values = values[sorted_cats]  # Reorder values to match sorted categories
        ax.bar(sorted_cats, values, bottom=bottom, label=dataset, color=dataset_colors[dataset])
        bottom += values

# Set y-axis to log scale
ax.set_yscale('log')

# Customize plot with colored x-labels
plt.xticks(rotation=90, ha='center', fontsize=8)
# Color the x-axis labels according to their group color
for tick, cat in zip(ax.get_xticklabels(), sorted_cats):
    tick.set_color(group_colors[cat_to_group[cat]])

# Add group labels
prev_group = None
start_idx = 0
for i, cat in enumerate(sorted_cats):
    curr_group = cat_to_group.get(cat)
    if curr_group != prev_group:
        if prev_group is not None:
            # Add group label
            mid = (start_idx + i - 1) / 2
            # Split long group names
            if len(prev_group) > 10:
                words = prev_group.split()
                split_idx = len(words) // 2
                label = f'{" ".join(words[:split_idx])}\n{" ".join(words[split_idx:])}'
            else:
                label = prev_group
                
            # Calculate group count
            group_cats = [c for c in sorted_cats[start_idx:i] if cat_to_group.get(c) == prev_group]
            group_count = dataset_category_counts[dataset_category_counts.index.isin(group_cats)].sum().sum()
            
            # Add group label and count
            ax.text(mid, -0.40, label, ha='center', va='bottom', transform=ax.get_xaxis_transform(), 
                   color=group_colors[prev_group], fontsize=11)
            ax.text(mid, -0.41, f'n={group_count:,}', ha='center', va='top', transform=ax.get_xaxis_transform(),
                   color=group_colors[prev_group], fontsize=9)
            
        start_idx = i
        prev_group = curr_group

# Add last group label
mid = (start_idx + len(sorted_cats) - 1) / 2
if len(prev_group) > 10:
    words = prev_group.split()
    split_idx = len(words) // 2
    label = f'{" ".join(words[:split_idx])}\n{" ".join(words[split_idx:])}'
else:
    label = prev_group

# Calculate last group count    
group_cats = [c for c in sorted_cats[start_idx:] if cat_to_group.get(c) == prev_group]
group_count = dataset_category_counts[dataset_category_counts.index.isin(group_cats)].sum().sum()

ax.text(mid, -0.40, f'{label}', ha='center', va='bottom', transform=ax.get_xaxis_transform(),
        color=group_colors[prev_group], fontsize=11)
ax.text(mid, -0.41, f'n={group_count:,}', ha='center', va='top', transform=ax.get_xaxis_transform(),
        color=group_colors[prev_group], fontsize=9)

plt.xlabel('Category', labelpad=30, fontsize=12)
plt.ylabel('Count (log scale)', fontsize=12)
# plt.title('Category Distribution Across Datasets', fontsize=14)

# Reorder legend handles and labels based on dataset sizes
handles, labels = ax.get_legend_handles_labels()
handles_dict = dict(zip(labels, handles))
ordered_handles = [handles_dict[dataset] for dataset in datasets_sorted]
ordered_labels = list(datasets_sorted)

# Add legend with smaller font inside the plot
plt.legend(ordered_handles, ordered_labels, title='Datasets (sorted by size)', 
          loc='upper right', ncol=3, frameon=True, bbox_to_anchor=(1, 1.05),
          fontsize=7.5, title_fontsize=12)

# Adjust layout to make room for group labels, remove margins
plt.margins(x=0)
plt.subplots_adjust(bottom=0.3, right=0.95)  # Added right padding

# Save plot without border
plt.savefig("bar/category_distribution.png", dpi=300, bbox_inches='tight', pad_inches=0.1)  # Added pad_inches
plt.close()
