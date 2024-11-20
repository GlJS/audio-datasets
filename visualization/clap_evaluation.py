import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import polars as pl
import argparse
from collections import Counter
import matplotlib.cm as cm
import seaborn as sns
from numba import jit
import itertools
from cuml.metrics import pairwise_distances
import json
tqdm.pandas()
sns.set_theme()

parser = argparse.ArgumentParser(description='PCA visualization of CLAP embeddings')
parser.add_argument('--components', type=int, default=2, help='Number of components for dimensionality reduction')
parser.add_argument('--embeddings_dir', type=str, default='/scratch-shared/gwijngaard/embeddings', help='Base directory containing audio and text embeddings')
args = parser.parse_args()

import cudf
from cuml.decomposition import PCA as cumlPCA
import cupy as cp

def load_categories():
    with open("visualization/categories.json") as f:
        return json.load(f)

def load_embeddings(embeddings_dir, modality):
    """Load all embeddings for a given modality (audio/text)"""
    embeddings_path = os.path.join(embeddings_dir, modality)
    all_embeddings = []
    all_labels = []
    
    for dataset in os.listdir(embeddings_path):
        dataset_path = os.path.join(embeddings_path, dataset)
        for split_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, split_file), 'rb') as f:
                data = pickle.load(f)
                paths, embeds = zip(*data)
                all_embeddings.extend(embeds)
                all_labels.extend([dataset] * len(embeds))
                
    return np.array(all_embeddings), np.array(all_labels)

def compute_chunk_distances(group1, group2_chunk):
    group1_gpu = group1.to_cupy()
    group2_chunk_gpu = group2_chunk.to_cupy()
    distances = pairwise_distances(group1_gpu, group2_chunk_gpu, metric='euclidean')
    return cp.sum(distances), distances.size

def calculate_average_distance(group1, group2, label1, label2, chunk_size=1000):
    total_distance = 0.0
    total_count = 0

    for i in tqdm(range(0, group2.shape[0], chunk_size), desc=f"Between {label1} and {label2}", position=1, leave=False):
        group2_chunk = group2[i:i+chunk_size]
        chunk_distance, chunk_count = compute_chunk_distances(group1, group2_chunk)
        total_distance += chunk_distance.get()
        total_count += chunk_count

    return total_distance / total_count

def average_distance_to_centroid(group):
    group = group.to_cupy()
    centroid = cp.mean(group, axis=0)
    distances = cp.linalg.norm(group - centroid, axis=1)
    return cp.mean(distances).get(), centroid.get()

def plot_pca(labels, embeddings, title, ax, label_to_color):
    # Convert embeddings to GPU
    embeddings_dask = cudf.DataFrame.from_records(embeddings)
    reducer = cumlPCA(n_components=args.components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_dask)
    
    # Calculate variability and centroids
    label_variability = {}
    for label in label_to_color.keys():
        group = reduced_embeddings[labels == label]
        if len(group) > 0:
            variability, centroid = average_distance_to_centroid(group)
            label_variability[label] = variability
            ax.scatter(centroid[0], centroid[1], color=label_to_color[label], 
                      s=150, zorder=40, marker="*", edgecolors='black', linewidths=0.5)
    
    # Save distances between groups
    distances_file = f"pca/{title.lower().replace(' ', '-')}_distances.csv"
    if not os.path.exists(distances_file):
        distances = []
        for label1, label2 in itertools.combinations(label_to_color.keys(), 2):
            group1 = reduced_embeddings[labels == label1]
            group2 = reduced_embeddings[labels == label2]
            if len(group1) > 0 and len(group2) > 0:
                distance = calculate_average_distance(group1, group2, label1, label2)
                distances.append([label1, label2, distance])
        
        distances_df = pd.DataFrame(distances, columns=["label1", "label2", "distance"])
        distances_df = distances_df.sort_values("distance", ascending=True)
        distances_df.to_csv(distances_file, index=False)
    
    # Plot points
    reduced_embeddings = reduced_embeddings.to_pandas().values
    for idx, label in enumerate(sorted(label_to_color.keys())):
        mask = labels == label
        if np.any(mask):
            ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                      color=label_to_color[label], label=label, s=0.01, zorder=idx+3)
    
    # Save variability metrics
    label_variability_df = pd.DataFrame(label_variability.items(), columns=["label", "variability"])
    label_variability_df = label_variability_df.sort_values("variability", ascending=False)
    label_variability_df.to_csv(f"pca/{title.lower().replace(' ', '-')}_variability.csv", index=False)

    # Set labels and title
    explained_variance = reducer.explained_variance_ratio_
    ax.set_xlabel(f"Component 1 ({explained_variance[0]*100:.2f}%)")
    ax.set_ylabel(f"Component 2 ({explained_variance[1]*100:.2f}%)")
    ax.set_title(title)

    return list(label_to_color.keys())

def plot_embeddings_comparison():
    # Load embeddings
    print("Loading audio and text embeddings...")
    audio_embeddings, audio_labels = load_embeddings(args.embeddings_dir, "audio")
    text_embeddings, text_labels = load_embeddings(args.embeddings_dir, "text")
    
    # Setup colors with more than 20 colors
    all_labels = np.unique(np.concatenate([audio_labels, text_labels]))
    num_labels = len(all_labels)
    colors = []
    colors.extend(plt.cm.tab20(np.linspace(0, 1, 20)))  # First 20 colors from tab20
    colors.extend(plt.cm.Set3(np.linspace(0, 1, 12)))   # 12 more colors from Set3
    colors.extend(plt.cm.Pastel1(np.linspace(0, 1, 9))) # 9 more colors if needed
    
    # Ensure we have enough colors
    colors = colors[:num_labels]  # Trim to exact number needed
    label_to_color = {label: colors[i] for i, label in enumerate(all_labels)}
    
    # Create plot
    print("Creating plot layout...")
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    print("Plotting PCA for audio embeddings...")
    unique_labels = plot_pca(audio_labels, audio_embeddings, "CLAP Audio Embeddings PCA", axs[0], label_to_color)
    print("Plotting PCA for text embeddings...")
    plot_pca(text_labels, text_embeddings, "CLAP Text Embeddings PCA", axs[1], label_to_color)
    
    # Add legend
    custom_lines = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=label_to_color[label], markersize=10)
                   for label in unique_labels]
    fig.legend(custom_lines, unique_labels, title="Datasets",
              loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig("pca/clap_embeddings_pca.png", dpi=150, bbox_inches='tight')
    plt.close()

# Create output directory
os.makedirs("pca", exist_ok=True)

# Run visualization
plot_embeddings_comparison()
