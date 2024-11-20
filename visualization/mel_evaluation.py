import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import argparse
from matplotlib.colors import LogNorm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader

# Constants
BETA = 0.5
K_NEAREST = 5
TAU_MEL = 0.5005
BACKGROUND_NUM = 5000

parser = argparse.ArgumentParser()
parser.add_argument('--embeddings_dir', type=str, 
                   default='/scratch-shared/gwijngaard/embeddings/mel',
                   help='Directory containing mel embeddings')
args = parser.parse_args()

def load_embeddings(embeddings_dir):
    """Load all mel embeddings"""
    all_embeddings = []
    all_labels = []
    
    for dataset in os.listdir(embeddings_dir):
        dataset_path = os.path.join(embeddings_dir, dataset)
        for split_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, split_file), 'rb') as f:
                data = pickle.load(f)
                paths, embeds = zip(*data)
                all_embeddings.extend(embeds)
                all_labels.extend([dataset] * len(embeds))
                
    return np.array(all_embeddings), np.array(all_labels)

def compute_similarity_matrix(embeddings1, embeddings2, background_set):
    """Compute similarity matrix between two sets of embeddings"""
    n1, n2 = len(embeddings1), len(embeddings2)
    similarity_matrix = np.zeros((n1, n2))
    
    # Compute background similarities
    background_sims = cosine_similarity(embeddings1, background_set)
    background_bias = np.mean(np.sort(background_sims, axis=1)[:, -K_NEAREST:], axis=1)
    
    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings1, embeddings2)
    
    # Apply bias correction
    similarities = similarities - BETA * background_bias.reshape(-1, 1)
    
    return similarities

def create_overlap_matrix(embeddings, labels):
    """Create overlap matrix between all datasets"""
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    overlap_matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
    count_matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
    
    # Select background set
    all_indices = np.arange(len(embeddings))
    background_indices = np.random.choice(all_indices, BACKGROUND_NUM, replace=False)
    background_set = embeddings[background_indices]
    
    for i, label1 in enumerate(tqdm(unique_labels)):
        mask1 = labels == label1
        embeddings1 = embeddings[mask1]
        
        for j, label2 in enumerate(unique_labels[i:], i):
            mask2 = labels == label2
            embeddings2 = embeddings[mask2]
            
            similarities = compute_similarity_matrix(embeddings1, embeddings2, background_set)
            overlaps = (similarities > TAU_MEL).sum()
            
            if i != j:
                total_possible = len(embeddings1) * len(embeddings2)
                overlap_matrix.loc[label1, label2] = overlaps
                overlap_matrix.loc[label2, label1] = overlaps
                count_matrix.loc[label1, label2] = total_possible
                count_matrix.loc[label2, label1] = total_possible
            else:
                total_possible = len(embeddings1) * (len(embeddings1) - 1) / 2
                overlap_matrix.loc[label1, label1] = overlaps
                count_matrix.loc[label1, label1] = total_possible
    
    # Convert to percentages
    percentage_matrix = (overlap_matrix / count_matrix) * 100
    return percentage_matrix

def plot_heatmap(matrix):
    """Create and save heatmap visualization"""
    plt.figure(figsize=(15, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(matrix), k=1)
    
    # Create heatmap
    sns.heatmap(matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='Reds',
                norm=LogNorm(vmin=0.0001, vmax=matrix.max().max()),
                square=True,
                cbar_kws={'label': 'Overlap Percentage'})
    
    plt.title('Dataset Overlap Based on Mel Spectrograms')
    plt.tight_layout()
    plt.savefig('mel_overlap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load embeddings
    print("Loading mel embeddings...")
    embeddings, labels = load_embeddings(args.embeddings_dir)
    
    # Create overlap matrix
    print("Computing overlap matrix...")
    overlap_matrix = create_overlap_matrix(embeddings, labels)
    
    # Save overlap matrix
    overlap_matrix.to_csv('mel_overlap_matrix.csv')
    
    # Create and save visualization
    print("Creating heatmap...")
    plot_heatmap(overlap_matrix)

if __name__ == "__main__":
    main() 