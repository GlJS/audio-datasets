import pickle
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import itertools

# Load allowed datasets from unique_datasets.json
with open('visualization/unique_datasets.json', 'r') as f:
    known_datasets = json.load(f)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings)
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        return self.embeddings[idx]

def load_embeddings(embeddings_dir):
    """Load audio embeddings"""
    embeddings_path = os.path.join(embeddings_dir, 'audio')
    dataset_embeddings = {}
    
    for dataset in os.listdir(embeddings_path):
        if dataset not in known_datasets:
            continue
            
        # Load name mapping if it exists
        name_map = {}
        name_map_path = f"name_maps/{dataset}_name_map.json"
        if os.path.exists(name_map_path):
            with open(name_map_path, 'r') as f:
                loaded_map = json.load(f)
                # Flatten the name map by combining train/valid/test dicts
                name_map = {}
                for split in loaded_map.keys():
                    name_map.update(loaded_map[split])
            
        dataset_path = os.path.join(embeddings_path, dataset)
        embeddings = []
        filenames = []
        
        for split_file in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, split_file), 'rb') as f:
                data = pickle.load(f)
                files, embeds = zip(*data)
                embeddings.extend(embeds)
                filenames.extend(files)
        
        dataset_embeddings[dataset] = {
            'embeddings': np.array(embeddings),
            'filenames': filenames,
            'name_map': name_map
        }
                
    return dataset_embeddings

def compute_similarities(embeddings_dict, batch_size=32768):
    """Compute pairwise similarities between datasets"""
    os.makedirs("splits", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(list(itertools.combinations(embeddings_dict.keys(), 2)), 
                desc="Computing audio similarities")
    for d1, d2 in pbar:
        pbar.set_description(f"Computing audio similarities between {d1} and {d2}") 
        # Create dataloaders
        dataset1 = EmbeddingDataset(embeddings_dict[d1]['embeddings'])
        dataset2 = EmbeddingDataset(embeddings_dict[d2]['embeddings'])
        loader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=8)
        loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=8)
        
        similar_pairs_d1 = []
        similar_pairs_d2 = []
        
        # Compute similarities in batches
        for i, batch1 in enumerate(tqdm(loader1, leave=False)):
            batch1 = batch1.to(device)
            start_idx1 = i * batch_size
            
            for j, batch2 in enumerate(loader2):
                batch2 = batch2.to(device)
                start_idx2 = j * batch_size
                
                batch1_norm = batch1 / batch1.norm(dim=1, keepdim=True)
                batch2_norm = batch2 / batch2.norm(dim=1, keepdim=True)
                
                # Compute cosine similarities
                sims = torch.mm(batch1_norm, batch2_norm.t())
                
                # Find pairs with >99% similarity
                similar_idx = torch.where(sims > 0.99)
                batch1_idx = similar_idx[0] + start_idx1
                batch2_idx = similar_idx[1] + start_idx2
                
                # Get filenames for similar pairs
                for idx1, idx2 in zip(batch1_idx.cpu(), batch2_idx.cpu()):
                    similar_pairs_d1.append(str(embeddings_dict[d1]['filenames'][idx1]))
                    similar_pairs_d2.append(str(embeddings_dict[d2]['filenames'][idx2]))
        
        if len(similar_pairs_d1) > 0:
            # Apply name mapping
            similar_pairs_d1_mapped = [str(embeddings_dict[d1]['name_map'].get(f, f)) for f in similar_pairs_d1]
            similar_pairs_d2_mapped = [str(embeddings_dict[d2]['name_map'].get(f, f)) for f in similar_pairs_d2]

            similar_pairs_d1_short = ["/".join(f.split('/')[-2:]) if len(f.split("/")) > 2 else f for f in similar_pairs_d1_mapped]
            similar_pairs_d2_short = ["/".join(f.split('/')[-2:]) if len(f.split("/")) > 2 else f for f in similar_pairs_d2_mapped]

            similar_pairs_d1_short = list(set(similar_pairs_d1_short))
            similar_pairs_d2_short = list(set(similar_pairs_d2_short))

            # Save similar pairs to CSV files        
            pd.DataFrame({'file_name': similar_pairs_d1_short}).to_csv(
                f'splits/{d1.lower()}_in_{d2.lower()}.csv', index=False)
            pd.DataFrame({'file_name': similar_pairs_d2_short}).to_csv(
                f'splits/{d2.lower()}_in_{d1.lower()}.csv', index=False)

def main():
    embeddings_dir = "/gpfs/work4/0/einf6190/embeddings"
    embeddings_dict = load_embeddings(embeddings_dir)
    compute_similarities(embeddings_dict)

if __name__ == "__main__":
    main()
