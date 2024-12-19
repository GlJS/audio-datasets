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

parser = argparse.ArgumentParser(description='t-SNE visualization of CLAP embeddings')
parser.add_argument('--perplexity', type=float, default=30.0, help='Perplexity parameter for t-SNE')
parser.add_argument('--embeddings_dir', type=str, default='/scratch-shared/gwijngaard/embeddings', help='Base directory containing audio and text embeddings')
args = parser.parse_args()

import cudf
from cuml.manifold import TSNE
import cupy as cp

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
    distances = pairwise_distances(group1, group2_chunk, metric='euclidean')
    return cp.sum(distances), distances.size

def calculate_average_distance(group1, group2, label1, label2, chunk_size=1000):
    total_distance = 0.0
    total_count = 0

    for i in tqdm(range(0, group2.shape[0], chunk_size), desc=f"Between {label1} and {label2}", position=1, leave=False):
        group2_chunk = group2[i:i+chunk_size]
        chunk_distance, chunk_count = compute_chunk_distances(group1, group2_chunk)
        total_distance += chunk_distance
        total_count += chunk_count

    return float(total_distance / total_count)

def average_distance_to_centroid(group):
    # Convert numpy array to cupy array
    group_gpu = cp.asarray(group)
    centroid = cp.mean(group_gpu, axis=0)
    distances = cp.linalg.norm(group_gpu - centroid, axis=1)
    return float(cp.mean(distances)), cp.asnumpy(centroid)

def plot_tsne(labels, embeddings, title, ax, label_to_color):
    print("Converting embeddings to GPU...")
    n_samples = len(embeddings)
    
    # Reduce dimensionality first using CPU-based PCA to 50 dimensions
    print("Reducing dimensions with PCA first...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Convert to GPU for TSNE
    embeddings_gpu = cp.asarray(embeddings_pca)
    
    batch_perplexity = 100
    reducer = TSNE(
        n_components=2,
        perplexity=batch_perplexity,
        random_state=42,
        init='random',  # Changed from 'pca' to 'random' for better memory usage
        n_iter=250,     # Reduced iterations for faster processing
        method='fft',
        verbose=True
    )
            
    reduced_embeddings = reducer.fit_transform(embeddings_gpu)
    
    # Normalize the embeddings to spread them out
    reduced_embeddings = (reduced_embeddings - cp.mean(reduced_embeddings, axis=0)) / cp.std(reduced_embeddings, axis=0)
    # Scale up the spread
    reduced_embeddings = reduced_embeddings * 20
            
    # Calculate variability and centroids
    print("Calculating variability and centroids...")
    label_variability = {}
    for label in label_to_color.keys():
        mask = labels == label
        group = reduced_embeddings[mask]
        if len(group) > 0:
            variability, centroid = average_distance_to_centroid(group)
            label_variability[label] = variability
            ax.scatter(centroid[0], centroid[1], color=label_to_color[label], 
                      s=150, zorder=40, marker="*", edgecolors='black', linewidths=0.5)
    
    # Save distances between groups
    print("Saving distances between groups...")
    distances_file = f"tsne/{title.lower().replace(' ', '-')}_distances.csv"
    if not os.path.exists(distances_file):
        distances = []
        for label1, label2 in itertools.combinations(label_to_color.keys(), 2):
            mask1 = labels == label1
            mask2 = labels == label2
            group1 = reduced_embeddings[mask1]
            group2 = reduced_embeddings[mask2]
            if len(group1) > 0 and len(group2) > 0:
                distance = calculate_average_distance(group1, group2, label1, label2)
                distances.append([label1, label2, distance])
        
        distances_df = pd.DataFrame(distances, columns=["label1", "label2", "distance"])
        distances_df = distances_df.sort_values("distance", ascending=True)
        distances_df.to_csv(distances_file, index=False)
    
    # Convert back to CPU for plotting
    reduced_embeddings_cpu = cp.asnumpy(reduced_embeddings)
    
    # Plot points
    print("Plotting points...")
    for idx, label in enumerate(sorted(label_to_color.keys())):
        mask = labels == label
        if np.any(mask):
            ax.scatter(reduced_embeddings_cpu[mask, 0], reduced_embeddings_cpu[mask, 1],
                      color=label_to_color[label], label=label, s=0.01, zorder=idx+3)
    
    # Save variability metrics
    print("Saving variability metrics...")
    label_variability_df = pd.DataFrame(label_variability.items(), columns=["label", "variability"])
    label_variability_df = label_variability_df.sort_values("variability", ascending=False)
    label_variability_df.to_csv(f"tsne/{title.lower().replace(' ', '-')}_variability.csv", index=False)

    print("Setting plot labels and title...")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(title)

    return list(label_to_color.keys())

def plot_embeddings_comparison():
    # Load embeddings
    print("Loading audio and text embeddings...")
    audio_embeddings, audio_labels = load_embeddings(args.embeddings_dir, "audio")
    text_embeddings, text_labels = load_embeddings(args.embeddings_dir, "text")

    all_labels = np.unique(np.concatenate([audio_labels, text_labels]))
    num_labels = len(all_labels)
    colors = []
    colors.extend(plt.get_cmap('tab20')(np.linspace(0, 1, 20)))  
    colors.extend(plt.get_cmap('Set3')(np.linspace(0, 1, 12)))   
    colors.extend(plt.get_cmap('Pastel1')(np.linspace(0, 1, 9))) 
    
    # Ensure we have enough colors
    colors = colors[:num_labels]  # Trim to exact number needed
    label_to_color = {label: colors[i] for i, label in enumerate(all_labels)}
    
    # Create plot
    print("Creating plot layout...")
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    print("Plotting t-SNE for audio embeddings...")
    unique_labels = plot_tsne(audio_labels, audio_embeddings, "CLAP Audio Embeddings t-SNE", axs[0], label_to_color)
    print("Plotting t-SNE for text embeddings...")
    plot_tsne(text_labels, text_embeddings, "CLAP Text Embeddings t-SNE", axs[1], label_to_color)
    
    # Add legend
    custom_lines = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=label_to_color[label], markersize=10)
                   for label in unique_labels]
    fig.legend(custom_lines, unique_labels, title="Datasets",
              loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig("tsne/clap_embeddings_tsne_w_pca.png", dpi=150, bbox_inches='tight')
    plt.close()

# Create output directory
os.makedirs("tsne", exist_ok=True)

# Run visualization
plot_embeddings_comparison()
