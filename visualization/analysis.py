import pandas as pd
import numpy as np
from scipy import stats
import os
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def load_data():
    # Load all category CSV files
    dfs = []
    total_rows = 0
    print("Loading datasets:")
    for f in os.listdir("output_cats/"):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(f"output_cats/{f}", usecols=['dataset', 'label'])
                rows = len(df)
                total_rows += rows
                print(f"Loaded {f}: {rows:,} rows")
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {str(e)}")
    
    print(f"\nTotal rows loaded: {total_rows:,}")
    df = pd.concat(dfs)
    
    # Clean data
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].replace('Animal ', 'Animal')
    df["dataset"] = df["dataset"].replace("RichDetailAudioTextSimulation", "RDATS")
    
    return df

def analyze_category_distribution(df):
    print("\nStarting category distribution analysis...")
    
    # Calculate overall category statistics
    total_categories = df['label'].nunique()
    total_datasets = df['dataset'].nunique()
    print(f"Total unique categories: {total_categories}")
    print(f"Total datasets: {total_datasets}")
    
    # Calculate category counts across all datasets
    category_counts = df['label'].value_counts()
    print("\nTop 10 Most Common Categories Overall:")
    for cat, count in category_counts.head(10).items():
        print(f"{cat}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    print("\nBottom 10 Least Common Categories Overall:")
    for cat, count in category_counts.tail(10).items():
        print(f"{cat}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    # Calculate dataset-specific statistics
    dataset_sizes = df.groupby('dataset').size()
    print("\nDataset Size Distribution:")
    size_stats = dataset_sizes.describe()
    print(f"Mean samples per dataset: {size_stats['mean']:,.1f}")
    print(f"Median samples per dataset: {size_stats['50%']:,.1f}")
    print(f"Std samples per dataset: {size_stats['std']:,.1f}")
    
    # Find datasets with unusual sizes
    size_zscore = (dataset_sizes - size_stats['mean']) / size_stats['std']
    print("\nDatasets with Unusual Sizes (|z-score| > 2):")
    unusual_sizes = size_zscore[abs(size_zscore) > 2].sort_values()
    for dataset, zscore in unusual_sizes.items():
        size = dataset_sizes[dataset]
        print(f"{dataset}: {size:,} samples (z-score: {zscore:.2f})")
    
    # Analyze category concentration per dataset
    print("\nDatasets with High Category Concentration:")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        cat_props = dataset_df['label'].value_counts() / len(dataset_df)
        
        # Find categories that make up more than 25% of the dataset
        high_conc_cats = cat_props[cat_props > 0.25]
        if not high_conc_cats.empty:
            print(f"\n{dataset} ({len(dataset_df):,} total samples):")
            for cat, prop in high_conc_cats.items():
                count = int(prop * len(dataset_df))
                print(f"  {cat}: {prop:.1%} ({count:,} samples)")
    
    # Analyze category diversity
    dataset_category_counts = df.groupby(['dataset', 'label']).size()
    category_diversity = df.groupby('dataset')['label'].nunique()
    print("\nCategory Diversity Distribution:")
    diversity_stats = category_diversity.describe()
    print(f"Mean categories per dataset: {diversity_stats['mean']:.1f}")
    print(f"Median categories per dataset: {diversity_stats['50%']:.1f}")
    print(f"Std categories per dataset: {diversity_stats['std']:.1f}")
    
    # Find datasets with unusual diversity
    diversity_zscore = (category_diversity - diversity_stats['mean']) / diversity_stats['std']
    print("\nDatasets with Unusual Category Diversity (|z-score| > 2):")
    unusual_diversity = diversity_zscore[abs(diversity_zscore) > 2].sort_values()
    for dataset, zscore in unusual_diversity.items():
        num_cats = category_diversity[dataset]
        print(f"{dataset}: {num_cats} categories (z-score: {zscore:.2f})")

if __name__ == "__main__":
    print("Starting analysis...")
    df = load_data()
    print("\nAnalyzing category distributions...")
    analyze_category_distribution(df) 