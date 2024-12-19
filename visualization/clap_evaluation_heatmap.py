import os
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from cuml.metrics import pairwise_distances
import torch
from torch.utils.data import Dataset, DataLoader
import json
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx]

def load_embeddings(embeddings_dir, modality):
    """Load all embeddings for a given modality (audio/text)"""
    embeddings_path = os.path.join(embeddings_dir, modality)
    dataset_embeddings = {}
    
    # Load allowed datasets if filtering
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
        if dataset not in allowed_datasets:
            continue
            
        dataset_path = os.path.join(embeddings_path, dataset)
        embeddings = []
        for split_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, split_file), 'rb') as f:
                data = pickle.load(f)
                _, embeds = zip(*data)
                embeddings.extend(embeds)
        
        # Apply dataset remapping
        label = datasets_to_remap.get(dataset, dataset)
        if label not in dataset_embeddings:
            dataset_embeddings[label] = []
        dataset_embeddings[label].extend(embeddings)
                
    return {k: np.array(v) for k,v in dataset_embeddings.items()}

def compute_centroid_distances(embeddings_dict, modality):
    """Compute distances between dataset centroids"""
    centroids = {}
    distances = []
    
    # Compute centroids
    for dataset, embeds in embeddings_dict.items():
        centroids[dataset] = cp.mean(cp.array(embeds), axis=0)
    
    # Compute pairwise distances
    for d1, d2 in itertools.combinations(centroids.keys(), 2):
        dist = float(cp.linalg.norm(centroids[d1] - centroids[d2]))
        distances.append([d1, d2, dist])
    
    distances_df = pd.DataFrame(distances, columns=["dataset1", "dataset2", "distance"])
    distances_df = distances_df.sort_values("distance")
    distances_df.to_csv(f"heatmap/{modality}_centroid_distances.csv", index=False)
    return distances_df

def compute_pairwise_similarities(embeddings_dict, modality, batch_size=2048):
    """Compute pairwise similarities between all embeddings"""
    # Check if results already exist
    output_file = f"heatmap/{modality}_pairwise_similarities.csv"
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
        
    similarity_stats = []
    
    for d1, d2 in tqdm(list(itertools.combinations(embeddings_dict.keys(), 2)), 
                       desc=f"Computing {modality} similarities"):
        # Create dataloaders
        dataset1 = EmbeddingDataset(embeddings_dict[d1])
        dataset2 = EmbeddingDataset(embeddings_dict[d2])
        loader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=0)
        loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=0)
        
        total_sims = 0
        sims_50 = 0
        sims_90 = 0
        sims_95 = 0
        sims_99 = 0
        
        # Compute similarities in batches
        for batch1 in tqdm(loader1, leave=False):
            for batch2 in loader2:
                batch1_norm = batch1 / batch1.norm(dim=1, keepdim=True)
                batch2_norm = batch2 / batch2.norm(dim=1, keepdim=True)
                
                # Compute cosine similarities
                sims = torch.mm(batch1_norm, batch2_norm.t())
                
                # Update counters
                total_sims += sims.numel()
                sims_50 += torch.sum(sims > 0.5).item()
                sims_90 += torch.sum(sims > 0.9).item() 
                sims_95 += torch.sum(sims > 0.95).item()
                sims_99 += torch.sum(sims > 0.99).item()
                
                del sims
                torch.cuda.empty_cache()
        
        similarity_stats.append({
            'dataset1': d1,
            'dataset2': d2,
            'total_comparisons': total_sims,
            'sim_50': sims_50,
            'sim_90': sims_90,
            'sim_95': sims_95,
            'sim_99': sims_99
        })
        
    # Save results
    stats_df = pd.DataFrame(similarity_stats)
    stats_df.to_csv(output_file, index=False)
    return stats_df

def plot_similarity_heatmap(similarity_stats, sim_threshold, modality, centroid_distances):
    """Create heatmap for a specific similarity threshold"""
    # Get unique datasets
    datasets = sorted(list(set(similarity_stats['dataset1'].unique()) | set(similarity_stats['dataset2'].unique())))
    n_datasets = len(datasets)
    
    # Create matrices for distances and similarities
    dist_matrix = np.zeros((n_datasets, n_datasets))
    sim_matrix = np.zeros((n_datasets, n_datasets))
    
    # Create mapping from dataset names to indices
    dataset_to_idx = {dataset: i for i, dataset in enumerate(datasets)}
    
    # Fill similarity matrix
    for _, row in similarity_stats.iterrows():
        i = dataset_to_idx[row['dataset1']]
        j = dataset_to_idx[row['dataset2']]
        # Convert to percentage
        sim_value = (row[sim_threshold] / row['total_comparisons']) * 100
        sim_matrix[i,j] = sim_value
        sim_matrix[j,i] = sim_value
    
    # Fill distance matrix
    max_dist = 0
    for _, row in centroid_distances.iterrows():
        i = dataset_to_idx[row['dataset1']]
        j = dataset_to_idx[row['dataset2']]
        dist_value = 1 - row['distance'] # Invert the distance value
        max_dist = max(max_dist, dist_value)
        dist_matrix[i,j] = dist_value
        dist_matrix[j,i] = dist_value
        
    # Create mask for lower triangle and diagonal
    mask = np.tril(np.ones_like(dist_matrix), k=-1)  # Changed k=0 to k=-1 to exclude diagonal
    
    # Plot heatmap with pastel colors from blue to red
    plt.figure(figsize=(12, 10))
    sns.heatmap(dist_matrix,
                xticklabels=datasets,
                yticklabels=datasets,
                cmap='coolwarm',  # Changed to a pastel color scale
                vmin=0,
                vmax=1,
                annot=sim_matrix,
                fmt='.1f',
                mask=mask)
    
    plt.title(f'{modality.capitalize()} {sim_threshold} Similarity Ratios (%)\n(Color: Centroid Distance)', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'heatmap/{modality}_{sim_threshold}_heatmap.png')
    plt.close()

def process_modality(embeddings_dir, modality):
    """Process a single modality (audio/text)"""
    print(f"Loading {modality} embeddings...")
    embeddings_dict = load_embeddings(embeddings_dir, modality)
    
    print(f"Computing {modality} centroid distances...")
    centroid_distances = compute_centroid_distances(embeddings_dict, modality)
    
    print(f"Computing {modality} pairwise similarities...")
    similarity_stats = compute_pairwise_similarities(embeddings_dict, modality)
    
    # Plot heatmaps for different similarity thresholds
    for sim_threshold in ['sim_50', 'sim_90', 'sim_95', 'sim_99']:
        plot_similarity_heatmap(similarity_stats, sim_threshold, modality, centroid_distances)
    
    return centroid_distances, similarity_stats

def main():
    os.makedirs("heatmap", exist_ok=True)
    embeddings_dir = "/scratch-shared/gwijngaard/embeddings"
    
    # Process both modalities
    for modality in ['audio', 'text']:
        centroid_distances, similarity_stats = process_modality(embeddings_dir, modality)
        
        # Plot heatmaps for different similarity thresholds
        for sim_threshold in ['sim_50', 'sim_90', 'sim_95', 'sim_99']:
            plot_similarity_heatmap(similarity_stats, sim_threshold, modality, centroid_distances)

if __name__ == "__main__":
    main()
