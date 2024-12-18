import os
import pickle
import json
from collections import defaultdict

def count_embeddings(embeddings_dir):
    """Count embeddings for both audio and text modalities"""
    counts = defaultdict(int)
    counts_all = defaultdict(int)
    
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

    # Count embeddings for both modalities
    for modality in ['audio', 'text']:
        embeddings_path = os.path.join(embeddings_dir, modality)
        
        for dataset in os.listdir(embeddings_path):
            dataset_path = os.path.join(embeddings_path, dataset)
            
            # Count all datasets
            for split_file in os.listdir(dataset_path):
                with open(os.path.join(dataset_path, split_file), 'rb') as f:
                    data = pickle.load(f)
                    counts_all[f"{dataset}_{modality}"] += len(data)
                    counts_all[f"total_{modality}"] += len(data)
            
            # Skip if not in allowed datasets for filtered counts
            if dataset not in allowed_datasets:
                continue
                
            # Count allowed datasets
            for split_file in os.listdir(dataset_path):
                with open(os.path.join(dataset_path, split_file), 'rb') as f:
                    data = pickle.load(f)
                    # Apply dataset remapping
                    label = datasets_to_remap.get(dataset, dataset)
                    counts[f"{label}_{modality}"] += len(data)
                    counts[f"total_{modality}"] += len(data)

    # Print results for allowed datasets
    print("\nEmbedding counts per allowed dataset and modality:")
    print("-" * 50)
    for key in sorted(counts.keys()):
        if not key.startswith('total'):
            print(f"{key}: {counts[key]:,}")
    
    print("\nTotal counts per modality (allowed datasets):")
    print("-" * 50)
    for modality in ['audio', 'text']:
        print(f"Total {modality}: {counts[f'total_{modality}']:,}")

    # Print results for all datasets
    print("\nEmbedding counts per dataset and modality (all datasets):")
    print("-" * 50)
    for key in sorted(counts_all.keys()):
        if not key.startswith('total'):
            print(f"{key}: {counts_all[key]:,}")
    
    print("\nTotal counts per modality (all datasets):")
    print("-" * 50)
    for modality in ['audio', 'text']:
        print(f"Total {modality}: {counts_all[f'total_{modality}']:,}")

if __name__ == "__main__":
    embeddings_dir = "/scratch-shared/gwijngaard/embeddings"
    count_embeddings(embeddings_dir)
