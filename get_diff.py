import pickle
import os
import pandas as pd
from tqdm.contrib.concurrent import process_map
import polars as pl
from datasets import *

known_datasets = [
    "AnimalSpeak",
    "AudioCaps",
    "AudioCaption",
    "AudioDiffCaps",
    "Audioset",
    "CAPTDURE",
    "Clotho",
    "ClothoAQA",
    "ClothoDetail",
    "DAQA",
    "FAVDBench",
    "FSD50k",
    "MACS",
    "MULTIS",
    "SoundDescs",
    "SoundingEarth",
    "TextToAudioGrounding",
    "WavText5K",
    "mAQA",
]


def load_dataset(dataset_name):
    if dataset_name in globals():
        dataset = globals()[dataset_name]()
        dataset = dataset.dropna(subset=["caption"])
        dataset["dataset"] = dataset_name.lower()
        assert (
            "file_name" in dataset.columns
        ), f"Dataset {dataset_name} does not have a 'file_name' column"
        assert (
            "split" in dataset.columns
        ), f"Dataset {dataset_name} does not have a 'split' column"
        dataset = dataset[["file_name", "split", "dataset"]]
        print(f"Loaded dataset {dataset_name} with {len(dataset)} rows")
    else:
        raise ValueError(f"Dataset {dataset_name} not found!")

    return pl.from_pandas(dataset)


# Load all datasets
all_datasets = pl.concat(process_map(load_dataset, known_datasets))
print("Combined datasets")

# Load similar pairs
with open("/storage/embeddings/similar_pairs_5000_cpu.pkl", "rb") as f:
    similar_pairs = pickle.load(f)
print(f"Number of similar pairs: {len(similar_pairs)}")

# Create DataFrame from similar pairs
df_data = pl.DataFrame(similar_pairs)
df_data.columns = ["file1", "file2", "value"]
df_data = df_data.filter((pl.col("file1") != "/") & (pl.col("file2") != "/"))

# Extract dataset and filename
df_data = df_data.with_columns(
    [
        pl.col("file1").str.split("/").list[4].alias("dataset1"),
        pl.col("file1").str.split("/").list[-1].alias("file_name1"),
        pl.col("file2").str.split("/").list[4].alias("dataset2"),
        pl.col("file2").str.split("/").list[-1].alias("file_name2"),
    ]
)

# Handle special cases for audiocaption and soundingearth
df_data = df_data.with_columns(
    [
        pl.when(pl.col("dataset1") == "audiocaption")
        .then(
            pl.col("file1").str.split("/").list[-2]
            + "/"
            + pl.col("file1").str.split("/").list[-1]
        )
        .otherwise(pl.col("file_name1"))
        .alias("file_name1"),
        pl.when(pl.col("dataset2") == "audiocaption")
        .then(
            pl.col("file2").str.split("/").list[-2]
            + "/"
            + pl.col("file2").str.split("/").list[-1]
        )
        .otherwise(pl.col("file_name2"))
        .alias("file_name2"),
    ]
)

df_data = df_data.with_columns(
    [
        pl.when(pl.col("dataset1") == "soundingearth")
        .then(
            pl.col("file1").str.split("/").list[-1].str.split("_").list[0]
            + "_"
            + pl.col("file1").str.split("/").list[-1].str.split("_").list[1]
            + "_"
            + pl.col("file1").str.split("/").list[-1].str.split("_").list[1]
        )
        .otherwise(pl.col("file_name1"))
        .alias("file_name1"),
        pl.when(pl.col("dataset2") == "soundingearth")
        .then(
            pl.col("file2").str.split("/").list[-1].str.split("_").list[0]
            + "_"
            + pl.col("file2").str.split("/").list[-1].str.split("_").list[1]
            + "_"
            + pl.col("file2").str.split("/").list[-1].str.split("_").list[1]
        )
        .otherwise(pl.col("file_name2"))
        .alias("file_name2"),
    ]
)
# Calculate the count for each dataset and filter df_data
all_datasets_count = all_datasets.group_by('dataset').agg(pl.count()).rename({'count': 'dataset_count'})

df_data = df_data.join(all_datasets_count, left_on='dataset1', right_on='dataset')
df_data = df_data.with_columns([
    (pl.col('dataset_count') / pl.col('dataset_count').sum()).alias('rel_count')
])
print(f"Shape of df_data: {df_data.shape}")

# Filter based on the relative count of dataset1
df_data = df_data.filter(pl.col('rel_count') > 1e-04)

# Drop the temporary columns
df_data = df_data.drop(['dataset_count', 'rel_count'])

print(f"Shape of df_data: {df_data.shape}")

# Function to create inclusion files for a dataset
def create_inclusion_files(dataset):
    for other_dataset in known_datasets:
        if dataset.lower() != other_dataset.lower():
            # Filter similar pairs where dataset1 is the current dataset and dataset2 is the other dataset
            inclusions = df_data.filter(
                (pl.col("dataset1") == dataset.lower())
                & (pl.col("dataset2") == other_dataset.lower())
            ).select(pl.col("file_name1").alias("file_name"))

            # Also include pairs where dataset2 is the current dataset and dataset1 is the other dataset
            inclusions = pl.concat(
                [
                    inclusions,
                    df_data.filter(
                        (pl.col("dataset2") == dataset.lower())
                        & (pl.col("dataset1") == other_dataset.lower())
                    ).select(pl.col("file_name2").alias("file_name")),
                ]
            )

            # Remove duplicates
            inclusions = inclusions.unique()

            print(f"Number of inclusions for {dataset} -> {other_dataset}: {len(inclusions)}")

            if len(inclusions) == 0:
                continue

            # Write to CSV
            output_file = f"/storage/in/{dataset.lower()}_in_{other_dataset.lower()}.csv"
            inclusions.write_csv(output_file)
            print(f"Created {output_file} with {len(inclusions)} rows")


# Create inclusion files for each dataset
for dataset in known_datasets:
    create_inclusion_files(dataset)

print("Finished creating all inclusion files.")
