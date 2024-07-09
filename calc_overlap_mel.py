import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import sys
import polars as pl
from datasets import *
known_datasets = ["AnimalSpeak", "AudioCaps", "AudioCaption", "AudioDiffCaps", "Audioset", 
                    "CAPTDURE", "Clotho", "ClothoAQA", "ClothoDetail", "DAQA", "FAVDBench", 
                    "FSD50k", "MACS", "MULTIS", "SoundDescs", "SoundingEarth", 
                    "TextToAudioGrounding", "WavText5K", "mAQA"]

if not os.path.exists("cross_tab.csv"):
    def load_dataset(dataset_name):
        if dataset_name in globals():
            dataset = globals()[dataset_name]()
            dataset = dataset.dropna(subset=["caption"])
            dataset["dataset"] = dataset_name.lower()
            assert "file_name" in dataset.columns, f"Dataset {dataset_name} does not have a 'file_name' column"
            assert "split" in dataset.columns, f"Dataset {dataset_name} does not have a 'split' column"
            dataset = dataset[["file_name", "split", "dataset"]]
            print(f"Loaded dataset {dataset_name} with {len(dataset)} rows")
        else:
            raise ValueError(f"Dataset {dataset_name} not found!")

        return pl.from_pandas(dataset)

    all_datasets = pl.concat(process_map(load_dataset, known_datasets))
    # all_datasets = pl.concat([load_dataset(dataset) for dataset in known_datasets])
    print("Combined datasets")
        
    with open("/scratch-shared/gwijngaard/embeddings/similar_pairs_5000_cpu.pkl", "rb") as f:
        similar_pairs = pickle.load(f)
    print(f"Number of similar pairs: {len(similar_pairs)}")

    # df_data = pl.read_pickle("/storage/embeddings/similar_pairs_5000.pkl")
    df_data = pl.DataFrame(similar_pairs)
    df_data.columns = ["file1", "file2", "value"]
    df_data = df_data.filter(df_data["file1"] != "/")
    df_data = df_data.filter(df_data["file2"] != "/")


    # Extract dataset and filename from the given paths
    def extract_dataset(path):
        parts = path.split('/')
        return parts[4]
    def extract_filename(path):
        parts = path.split('/')
        return parts[-1]

    # Apply the extraction function to the DataFrame
    # df_data = df_data.with_columns([
    #     pl.col("file1").map_elements(lambda x: extract_dataset(x), return_dtype=pl.String).alias("dataset1"),
    #     pl.col("file1").map_elements(lambda x: extract_filename(x), return_dtype=pl.String).alias("file_name1"),
    #     pl.col("file2").map_elements(lambda x: extract_dataset(x), return_dtype=pl.String).alias("dataset2"),
    #     pl.col("file2").map_elements(lambda x: extract_filename(x), return_dtype=pl.String).alias("file_name2")
    # ])
    df_data = df_data.with_columns([
        pl.col("file1").str.split("/").list[4].alias("dataset1"),
        pl.col("file1").str.split("/").list[-1].alias("file_name1"),
        pl.col("file2").str.split("/").list[4].alias("dataset2"),
        pl.col("file2").str.split("/").list[-1].alias("file_name2")
    ])


    # Extra case for audiocaption
    df_data = df_data.with_columns([
        pl.when(pl.col("dataset1") == "audiocaption")
        .then(pl.col("file1").str.split("/").list[-2] + "/" + pl.col("file1").str.split("/").list[-1])
        .otherwise(pl.col("file_name1")).alias("file_name1"),
        pl.when(pl.col("dataset2") == "audiocaption")
        .then(pl.col("file2").str.split("/").list[-2] + "/" + pl.col("file2").str.split("/").list[-1])
        .otherwise(pl.col("file_name2")).alias("file_name2")
    ])

    # Extra case for soundingearth
    # split on _ and take the first part and remove only the last part but keep the extension
    df_data = df_data.with_columns([
        pl.when(pl.col("dataset1") == "soundingearth")
        .then(pl.col("file1").str.split("/").list[-1].str.split("_").list[0] + "_" + pl.col("file1").str.split("/").list[-1].str.split("_").list[1] + "_" + pl.col("file1").str.split("/").list[-1].str.split("_").list[1])
        .otherwise(pl.col("file_name1")).alias("file_name1"),
        pl.when(pl.col("dataset2") == "soundingearth")
        .then(pl.col("file2").str.split("/").list[-1].str.split("_").list[0] + "_" + pl.col("file2").str.split("/").list[-1].str.split("_").list[1] + "_" + pl.col("file2").str.split("/").list[-1].str.split("_").list[1])
        .otherwise(pl.col("file_name2")).alias("file_name2")
    ])

    print("Created columns")



    # Join with large_df to get the split information
    print("Shape", df_data.shape)
    df_data_merged = df_data.join(all_datasets, left_on=['file_name1', 'dataset1'], right_on=['file_name', 'dataset'], how='inner').rename({'split': 'split1'})
    print("Shape", df_data_merged.shape)
    df_data_merged = df_data_merged.unique()
    print("Shape", df_data_merged.shape)
    df_data_merged = df_data_merged.join(all_datasets, left_on=['file_name2', 'dataset2'], right_on=['file_name', 'dataset'], how='inner').rename({'split': 'split2'})
    print("Shape", df_data_merged.shape)
    df_data_merged = df_data_merged.unique()
    print("Shape", df_data_merged.shape)

    all_datasets = all_datasets.with_columns([
        (pl.col('dataset') + '_' + pl.col('split')).alias('combo')
    ])
    counts = all_datasets["combo"].value_counts()
    
    # Create a Cartesian product of the unique names
    cartesian_df = counts.join(counts, how='cross')


    # Calculate the product of counts for each pair
    cartesian_df = cartesian_df.with_columns([
        (pl.col('count') * pl.col('count_right')).cast(pl.UInt64).alias('product_of_counts')
    ])
        

    # Create a new column for combination of dataset and split
    df_data_merged = df_data_merged.with_columns([
        (pl.col('dataset1') + '_' + pl.col('split1')).alias('combo1'),
        (pl.col('dataset2') + '_' + pl.col('split2')).alias('combo2')
    ])

    # Count occurrences of each (dataset, split) combination
    cross_tab = df_data_merged.group_by(['combo1', 'combo2']).len()
    
    
    # divide by the counts to get the percentage
    cross_tab = cross_tab.join(cartesian_df, left_on=['combo1', 'combo2'], right_on=["combo", "combo_right"])
    cross_tab = cross_tab.with_columns([
        ((pl.col('len') / pl.col('product_of_counts')) * 100).alias('rel')
    ])
    cross_tab = cross_tab.drop(['count', 'count_right', 'product_of_counts'])

    # Convert to pandas DataFrame for plotting
    cross_tab_pandas = cross_tab.to_pandas()

    cross_tab_pandas.to_csv("cross_tab.csv", index=False)
else:
    cross_tab_pandas = pd.read_csv("cross_tab.csv")

def create_heatmap(df):
    
    df = df.sort_values(by=['combo1', 'combo2'])

    df_len = df.pivot(index='combo1', columns='combo2', values='len')
    df_rel = df.pivot(index='combo1', columns='combo2', values='rel')
    
    # # Replace NaN values with a small value
    df_len = df_len.fillna(0)
    
    df_len = df_len.astype(int)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(df_rel, dtype=bool))
    
    mask2 = np.where(df_len < 10, True, False)
    
    mask = (mask | mask2)

    plt.figure(figsize=(20, 15))

    # new_df = df.replace(0, 0.1)

    # # Set the log norm for the color scale with adjusted vmin
    vmin = 1e-4
    log_norm = LogNorm(vmin=vmin, vmax=df_rel.max().max())

    # Create the heatmap using seaborn with log scale and mask
    # sns.heatmap(new_df, mask=mask, annot=df, fmt='d', cmap='Reds', annot_kws={"size": 8}, linewidths=.5, norm=log_norm)
    # ax = sns.heatmap(df, annot=df, mask=mask, fmt=".4f", cmap='Reds', annot_kws={"size": 6}, linewidths=.5, norm=log_norm)
    ax = sns.heatmap(df_rel, annot=df_len, mask=mask, fmt="d", cmap='Reds', annot_kws={"size": 6}, linewidths=.5, norm=log_norm)
    ax.set(xlabel=None)
    ax.set(ylabel=None)

    # Rotates the x-axis labels so they fit better
    plt.xticks(rotation=90)

    # Ensures the y-axis labels are readable
    plt.yticks(rotation=0)

    plt.savefig('heatmap3.png', bbox_inches='tight', dpi=300)
    plt.show()

# Create the heatmap
create_heatmap(cross_tab_pandas)