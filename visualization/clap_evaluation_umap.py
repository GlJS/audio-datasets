import os
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import itertools
from cuml.manifold import UMAP
import cudf
import cupy as cp
from cuml.metrics import pairwise_distances
import seaborn as sns

tqdm.pandas()
sns.set_theme()

parser = argparse.ArgumentParser(description='UMAP visualization of CLAP embeddings')
parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors parameter for UMAP')
parser.add_argument('--min_dist', type=float, default=0.1, help='min_dist parameter for UMAP')
parser.add_argument('--embeddings_dir', type=str, default='/scratch-shared/gwijngaard/embeddings', 
                    help='Base directory containing audio and text embeddings')
args = parser.parse_args()

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

    for i in tqdm(range(0, group2.shape[0], chunk_size), 
                 desc=f"Between {label1} and {label2}", position=1, leave=False):
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

def plot_umap(labels, embeddings, title, ax, label_to_color):
    print("Converting embeddings to GPU...")
    
    # Clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    embeddings_dask = cudf.DataFrame.from_records(embeddings)
    reducer = UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=0.5,  # Increased from 0.1 to spread points more
        spread=2.0,    # Added spread parameter to increase separation
        random_state=42,
        verbose=True
    )

    reduced_embeddings = reducer.fit_transform(embeddings_dask)

    # # Calculate variability and centroids
    # print("Calculating variability and centroids...")
    # label_variability = {}
    # for label in label_to_color.keys():
    #     group = reduced_embeddings[labels == label]
    #     if len(group) > 0:
    #         variability, centroid = average_distance_to_centroid(group)
    #         label_variability[label] = variability
    #         ax.scatter(centroid[0], centroid[1], color=label_to_color[label], 
    #                   s=150, zorder=40, marker="*", edgecolors='black', linewidths=0.5)
    
    # # Save distances between groups
    # print("Saving distances between groups...")
    # distances_file = f"umap/{title.lower().replace(' ', '-')}_distances.csv"
    # if not os.path.exists(distances_file):
    #     distances = []
    #     for label1, label2 in itertools.combinations(label_to_color.keys(), 2):
    #         group1 = reduced_embeddings[labels == label1]
    #         group2 = reduced_embeddings[labels == label2]
    #         if len(group1) > 0 and len(group2) > 0:
    #             distance = calculate_average_distance(group1, group2, label1, label2)
    #             distances.append([label1, label2, distance])
        
        # distances_df = pd.DataFrame(distances, columns=["label1", "label2", "distance"])
        # distances_df = distances_df.sort_values("distance", ascending=True)
        # distances_df.to_csv(distances_file, index=False)
    
    # Plot points
    reduced_embeddings = reduced_embeddings.to_pandas().values
    print("Plotting points...")
    for idx, label in enumerate(sorted(label_to_color.keys())):
        mask = labels == label
        if np.any(mask):
            ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1],
                      color=label_to_color[label], label=label, s=5, zorder=idx+3, alpha=0.6)  # Increased point size and added transparency
    
    # Save variability metrics
    # print("Saving variability metrics...")
    # label_variability_df = pd.DataFrame(label_variability.items(), columns=["label", "variability"])
    # label_variability_df = label_variability_df.sort_values("variability", ascending=False)
    # label_variability_df.to_csv(f"umap/{title.lower().replace(' ', '-')}_variability.csv", index=False)

    print("Setting plot labels and title...")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
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
    colors = colors[:num_labels]
    label_to_color = {label: colors[i] for i, label in enumerate(all_labels)}
    
    # Create plot
    print("Creating plot layout...")
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    print("Plotting UMAP for audio embeddings...")
    unique_labels = plot_umap(audio_labels, audio_embeddings, 
                            "CLAP Audio Embeddings UMAP", axs[0], label_to_color)
    print("Plotting UMAP for text embeddings...")
    plot_umap(text_labels, text_embeddings, 
             "CLAP Text Embeddings UMAP", axs[1], label_to_color)
    
    # Add legend
    custom_lines = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=label_to_color[label], markersize=10)
                   for label in unique_labels]
    fig.legend(custom_lines, unique_labels, title="Datasets",
              loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig("umap/clap_embeddings_umap.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create output directory
    os.makedirs("umap", exist_ok=True)

    # Run visualization
    plot_embeddings_comparison() 