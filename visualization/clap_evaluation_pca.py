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
import matplotlib.transforms as mtrans
tqdm.pandas()

parser = argparse.ArgumentParser(description='PCA visualization of CLAP embeddings')
parser.add_argument('--components', type=int, default=2, help='Number of components for dimensionality reduction')
parser.add_argument('--embeddings_dir', type=str, default='/gpfs/work4/0/einf6190/embeddings', help='Base directory containing audio and text embeddings')
parser.add_argument('--filter_datasets', action='store_true', default=True, help='Only include datasets from unique_datasets.json')
args = parser.parse_args()

import cudf
from cuml.decomposition import PCA as cumlPCA
import cupy as cp

def load_embeddings(embeddings_dir, modality):
    """Load all embeddings for a given modality (audio/text)"""
    embeddings_path = os.path.join(embeddings_dir, modality)
    all_embeddings = []
    all_labels = []
    label_counts = Counter()
    
    # Load allowed datasets if filtering
    allowed_datasets = None
    if args.filter_datasets:
        with open('visualization/unique_datasets.json', 'r') as f:
            allowed_datasets = set(json.load(f))
    
    # Define dataset remapping
    datasets_to_remap = {
        'FreeToUseSounds': 'LAION-Audio-630k',
        'WeSoundEffects': 'LAION-Audio-630k', 
        'Audiostock': 'LAION-Audio-630k',
        'EpidemicSoundEffects': 'LAION-Audio-630k',
        'Paramount': 'LAION-Audio-630k',
        'macs': 'MACS'
    }
    
    for dataset in os.listdir(embeddings_path):
        # Skip if filtering and dataset not in allowed list
        if args.filter_datasets and dataset not in allowed_datasets:
            continue
            
        dataset_path = os.path.join(embeddings_path, dataset)
        for split_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, split_file), 'rb') as f:
                data = pickle.load(f)
                paths, embeds = zip(*data)
                all_embeddings.extend(embeds)
                # Apply dataset remapping
                label = datasets_to_remap.get(dataset, dataset)
                all_labels.extend([label] * len(embeds))
                label_counts[label] += len(embeds)
                
    return np.array(all_embeddings), np.array(all_labels), label_counts

def average_distance_to_centroid(group):
    group = group.to_cupy()
    centroid = cp.mean(group, axis=0)
    distances = cp.linalg.norm(group - centroid, axis=1)
    return cp.mean(distances).get(), centroid.get()

def compute_pca_data(labels, embeddings, title, label_to_color, label_counts):
    # Calculate original space variability and centroids first
    embeddings_gpu = cudf.DataFrame.from_records(embeddings)
    orig_label_variability = {}
    orig_centroids = {}
    for label in label_to_color.keys():
        group = embeddings_gpu[labels == label]
        if len(group) > 0:
            variability, centroid = average_distance_to_centroid(group)
            orig_label_variability[label] = variability
            orig_centroids[label] = centroid
    
    # Save original space metrics
    orig_variability_df = pd.DataFrame([(label, var, label_counts[label]) 
                                      for label, var in orig_label_variability.items()],
                                     columns=["label", "variability", "num_sounds"])
    orig_variability_df = orig_variability_df.sort_values("variability", ascending=False)
    orig_variability_df.to_csv(f"pca/{title.lower().replace(' ', '-')}_original_variability.csv", index=False)
    
    # Perform PCA reduction
    reducer = cumlPCA(n_components=args.components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_gpu)
    
    # Calculate PCA space variability and centroids
    label_variability = {}
    centroids = {}
    for label in label_to_color.keys():
        group = reduced_embeddings[labels == label]
        if len(group) > 0:
            variability, centroid = average_distance_to_centroid(group)
            label_variability[label] = variability
            centroids[label] = centroid
    
    # Convert reduced embeddings to pandas
    reduced_embeddings_pd = reduced_embeddings.to_pandas().values
    
    # Save PCA space variability metrics
    label_variability_df = pd.DataFrame([(label, var, label_counts[label]) 
                                       for label, var in label_variability.items()],
                                      columns=["label", "variability", "num_sounds"])
    label_variability_df = label_variability_df.sort_values("variability", ascending=False)
    label_variability_df.to_csv(f"pca/{title.lower().replace(' ', '-')}_pca_variability.csv", index=False)

    return reduced_embeddings_pd, centroids, label_variability, reducer.explained_variance_ratio_

def plot_pca(ax, title, reduced_embeddings, labels, centroids, label_to_color, explained_variance, show_labels=False):
    ax.set_facecolor('white')
    
    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Plot centroids and labels
    for label in label_to_color.keys():
        if label in centroids:
            centroid = centroids[label]
            ax.scatter(centroid[0], centroid[1], color=label_to_color[label], 
                      s=150, zorder=40, marker="*", edgecolors='black', linewidths=0.5)
            if show_labels:
                ax.annotate(label, (centroid[0], centroid[1]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, zorder=41)
    
    # Get dataset sizes and sort labels by size
    dataset_sizes = {label: np.sum(labels == label) for label in label_to_color.keys()}
    sorted_labels = sorted(label_to_color.keys(), key=lambda x: dataset_sizes[x], reverse=True)
    
    # Plot points in order of dataset size (largest first)
    for idx, label in enumerate(sorted_labels):
        mask = labels == label
        if np.any(mask):
            ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                      color=label_to_color[label], label=label, s=0.01, zorder=idx+3)

    # Set labels and title
    ax.set_xlabel(f"Component 1 ({explained_variance[0]*100:.2f}%)")
    ax.set_ylabel(f"Component 2 ({explained_variance[1]*100:.2f}%)")
    ax.set_title(title)

    return sorted_labels

def plot_embeddings_comparison():
    # Load embeddings
    print("Loading audio and text embeddings...")
    audio_embeddings, audio_labels, audio_counts = load_embeddings(args.embeddings_dir, "audio")
    text_embeddings, text_labels, text_counts = load_embeddings(args.embeddings_dir, "text")
    
    # Setup pastel colors
    all_labels = np.unique(np.concatenate([audio_labels, text_labels]))
    num_labels = len(all_labels)
    
    # Generate pastel colors
    pastel_colors = plt.cm.Pastel1(np.linspace(0, 1, 9))
    pastel_colors2 = plt.cm.Pastel2(np.linspace(0, 1, 8))
    set2 = plt.cm.Set2(np.linspace(0, 1, 8))
    set3 = plt.cm.Set3(np.linspace(0, 1, 12))
    
    colors = np.vstack([pastel_colors, pastel_colors2, set2, set3])
    colors = colors[:num_labels]
    
    label_to_color = {label: colors[i] for i, label in enumerate(all_labels)}
    
    # Compute PCA data once for each modality
    print("Computing PCA for audio embeddings...")
    audio_reduced, audio_centroids, audio_variability, audio_variance = compute_pca_data(
        audio_labels, audio_embeddings, "CLAP Audio Embeddings PCA", label_to_color, audio_counts)
    
    print("Computing PCA for text embeddings...")
    text_reduced, text_centroids, text_variability, text_variance = compute_pca_data(
        text_labels, text_embeddings, "CLAP Text Embeddings PCA", label_to_color, text_counts)
    
    # Create plots - one with labels, one without
    for show_labels in [True, False]:
        print(f"Creating plot layout (labels: {show_labels})...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'wspace': 0.1}, constrained_layout=True)
        fig.patch.set_facecolor('white')
        
        print("Plotting PCA visualizations...")
        sorted_labels = plot_pca(ax1, "CLAP Audio Embeddings PCA", 
                               audio_reduced, audio_labels, audio_centroids, 
                               label_to_color, audio_variance, show_labels)
        
        plot_pca(ax2, "CLAP Text Embeddings PCA",
                text_reduced, text_labels, text_centroids,
                label_to_color, text_variance, show_labels)
        
        # Add legend to the right of both plots, closer to the figure
        custom_lines = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=label_to_color[label], markersize=10)
                       for label in sorted_labels]
        fig.legend(custom_lines, sorted_labels, title="Datasets",
                  loc='center left', frameon=True, bbox_to_anchor=(0.9, 0.5))
                
        # Get bounding boxes and draw vertical line between plots
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = list(map(get_bbox, [ax1, ax2]))
        
        # Get the minimum and maximum extent, get the coordinate half-way between plots
        xmax = bboxes[0].x1
        xmin = bboxes[1].x0
        x = (xmax + xmin) / 2 - 0.005
        
        # Draw vertical line
        line = plt.Line2D([x, x], [0.05, 0.95], transform=fig.transFigure, color="#808080", linestyle='-', linewidth=0.5)
        fig.add_artist(line)
        
        plt.tight_layout()
        plt.margins(x=0)
        plt.savefig(f"pca/clap_embeddings_pca{'_labels' if show_labels else ''}{'_filtered' if args.filter_datasets else ''}.png", dpi=150, bbox_inches='tight')
        plt.close()

os.makedirs("pca", exist_ok=True)
plot_embeddings_comparison()