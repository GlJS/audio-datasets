import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

# Read the segment files
balanced = pd.read_csv('data/Audioset/balanced_train_segments.csv', skiprows=3, header=None)
unbalanced = pd.read_csv('data/Audioset/unbalanced_train_segments.csv', skiprows=3, header=None)
eval_segments = pd.read_csv('data/Audioset/eval_segments.csv', skiprows=3, header=None)

# Read class labels and ontology
class_labels = pd.read_csv('data/Audioset/class_labels_indices.csv')
with open('visualization/1.json', 'r') as f:
    ontology = json.load(f)

# Create mapping of id to top-level category
top_level_map = {}
for item in ontology:
    if not item.get('child_ids'):
        continue
    for child in item['child_ids']:
        top_level_map[child] = item['name']
    top_level_map[item['id']] = item['name']

# Function to process labels string into list
def process_labels(labels_str):
    if pd.isna(labels_str):
        return []
    return str(labels_str).strip('"').split(',')

# Combine all segments
all_segments = pd.concat([balanced, unbalanced, eval_segments])

# Get all unique labels from AudioSet
all_labels = []
for labels in all_segments.iloc[:, 3]:
    all_labels.extend(process_labels(labels))

# Clean empty strings and whitespace
all_labels = [label.strip() for label in all_labels if label.strip()]

# Convert to Series for value_counts
labels_series = pd.Series(all_labels)
audioset_counts = labels_series.value_counts()
total_audioset = len(all_labels)

# Load model predictions
def load_model_predictions():
    dfs = []
    for f in os.listdir("output_cats/"):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(f"output_cats/{f}", usecols=['dataset', 'label'])
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {str(e)}")
    
    df = pd.concat(dfs)
    df = df.dropna(subset=['label'])
    return df

# Get model prediction counts
model_predictions = load_model_predictions()
model_counts = model_predictions['label'].value_counts()
total_model = len(model_predictions)

# Get top 200 labels with their counts from AudioSet
top_200_audioset = audioset_counts.head(200)

# Merge AudioSet counts with class labels for readable names
audioset_with_names = pd.merge(
    pd.DataFrame({'mid': top_200_audioset.index, 'audioset_count': top_200_audioset.values}),
    class_labels,
    left_on='mid',
    right_on='mid'
)[['display_name', 'mid', 'audioset_count']]

# Create comparison DataFrame with normalized counts
comparison_df = pd.DataFrame({
    'display_name': audioset_with_names['display_name'],
    'mid': audioset_with_names['mid'],
    'audioset_norm': audioset_with_names['audioset_count'] / total_audioset,
    'model_norm': [model_counts.get(name, 0) / total_model for name in audioset_with_names['display_name']],
    'audioset_count': audioset_with_names['audioset_count'],
    'model_count': [model_counts.get(name, 0) for name in audioset_with_names['display_name']]
})

# Add top-level category information
comparison_df['top_level'] = comparison_df['mid'].map(top_level_map)

# Calculate normalized relative difference
comparison_df['norm_relative_diff'] = (comparison_df['model_norm'] - comparison_df['audioset_norm']) / comparison_df['audioset_norm']

# Sort by absolute normalized relative difference
comparison_df['abs_norm_relative_diff'] = abs(comparison_df['norm_relative_diff'])
comparison_df_sorted = comparison_df.sort_values('abs_norm_relative_diff', ascending=False)

# Find missing categories in model predictions
missing_in_model = comparison_df[comparison_df['model_count'] == 0].sort_values('audioset_count', ascending=False)

print("\nTop Categories from AudioSet Missing in Model Predictions:")
print("Format: Category Name (Count in AudioSet) - Top-level Category")
print("-" * 70)
for idx, row in missing_in_model.head(20).iterrows():
    print(f"{row['display_name']} ({int(row['audioset_count'])}) - {row['top_level']}")

print("\nTop 20 Normalized Discrepancies (accounting for dataset size differences):")
print(comparison_df_sorted[['display_name', 'top_level', 'audioset_norm', 'model_norm', 'norm_relative_diff']].head(20).to_string(float_format=lambda x: '{:.6f}'.format(x)))
