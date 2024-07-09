import os
import numpy as np
import librosa
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.contrib.concurrent import process_map
from glob import glob
from tqdm import tqdm
import pickle
import argparse
import os
from torchmetrics.functional import pairwise_cosine_similarity
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="audioset", help="Directory to search for mel files")
parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="Device to use for computation")
args = parser.parse_args()

# Set the device based on user input
device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Current directory:", args.dir)

# Parameters
SR = 16000
N_MELS = 16
WIN_LENGTH = int(0.128 * SR)  # 128ms window
HOP_LENGTH = WIN_LENGTH * 3 // 4  # 75% overlap
DB_CLIP_MIN = -40
BETA = 0.5
K_NEAREST = 5
TAU_MEL = 0.5005
TAU_CLAP = None  # Need to set this based on the results for parity
TARGET_LENGTH = 10.242
BATCH_SIZE = 10000  # Reduced batch size for memory efficiency
BACKGROUND_NUM = 5000

def compute_mel(paths, mel_list):
    paths = np.array(paths)
    mels = torch.tensor(np.array(mel_list), dtype=torch.float32).to(device)
    results = []

    # Compute background set for normalization
    background_set_mel = mels[torch.randperm(mels.shape[0])[:BACKGROUND_NUM]]

    n = mels.shape[0]
    biases = pairwise_cosine_similarity(background_set_mel, mels)
    biases = torch.mean(torch.topk(biases, K_NEAREST, dim=0).values, dim=0)

    # similarity_matrix = torch.zeros((n, n), dtype=torch.float32).to(device)
    # Compute similarity matrices
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Batch search", position=0):
        t2 = mels[i:i + BATCH_SIZE].clone().detach()
        cos_sim = pairwise_cosine_similarity(mels, t2)
        
        cos_sim = cos_sim.T

        cos_sim = cos_sim - BETA * biases
        mask = (cos_sim > TAU_MEL)
        
        indices_i = torch.arange(0, len(t2)).to(device)
        indices_j = torch.arange(i, i + len(t2)).to(device)
        
        
        mask[indices_i, indices_j] = False



        # similarity_matrix[i:i + BATCH_SIZE] = sim
        j_indices, k_indices = torch.nonzero(mask, as_tuple=True)
        
        if args.device == "gpu":
            filtered_cos_sim = cos_sim[j_indices, k_indices].cpu().detach().numpy()
        else:
            filtered_cos_sim = cos_sim[j_indices, k_indices].detach().numpy()
        j_indices += i
        if args.device == "gpu":
            j_indices = j_indices.cpu()
            k_indices = k_indices.cpu()
        first_path = paths[j_indices]
        second_path = paths[k_indices]
        

        results += list(zip(first_path, second_path, filtered_cos_sim))

    # Retrieve potential duplicates
    
    return results

def get_mel_pkls(path):
    all_files = []
    for f in os.listdir(path):
        if f.endswith(".pkl"):
            with open(os.path.join(path, f), "rb") as file:
                all_files.extend(pickle.load(file))
            print("Loaded", f)
    
    print(f"Loaded {len(all_files)} files")
    return all_files

if __name__ == "__main__":
    if args.dir == "all":
        audio_dir = "/home/gwijngaard/projects/clap-surveypaper/mel"
        mel_descriptors = get_mel_pkls(audio_dir)
    else:
        audio_dir = f"/home/gwijngaard/projects/clap-surveypaper/mel/{args.dir}.pkl"
        mel_descriptors = pickle.load(open(audio_dir, "rb"))
        print(f"Loaded {len(mel_descriptors)} files")

    paths, mel_list = zip(*mel_descriptors)
    results = compute_mel(paths, mel_list)
    
    print("Number of similar pairs found:", len(results))
    # Optionally, do something with `results`, like saving or further processing.
    with open(f"/scratch-shared/gwijngaard/embeddings/similar_pairs2_{BACKGROUND_NUM}.pkl", "wb") as f:
        pickle.dump(results, f)
