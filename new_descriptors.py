import numpy as np
import librosa
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.contrib.concurrent import process_map
from glob import glob
from tqdm import tqdm
import pickle
import numba as nb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="audiocaption")
parser.add_argument("--option", type=str, choices=["build", "compute"], default="build")
args = parser.parse_args()

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


def get_filepaths(directory):
    filepaths = glob(f"{directory}/**/*.wav", recursive=True) + \
                glob(f"{directory}/**/*.mp3", recursive=True) + \
                glob(f"{directory}/**/*.flac", recursive=True) + \
                glob(f"{directory}/**/*.ogg", recursive=True) + \
                glob(f"{directory}/**/*.m4a", recursive=True) + \
                glob(f"{directory}/**/*.aiff", recursive=True) + \
                glob(f"{directory}/**/*.aif", recursive=True) + \
                glob(f"{directory}/**/*.au", recursive=True) + \
                glob(f"{directory}/**/*.3gp", recursive=True) + \
                glob(f"{directory}/**/*.3gpp", recursive=True) + \
                glob(f"{directory}/**/*.mp4", recursive=True) + \
                glob(f"{directory}/**/*.mpeg", recursive=True) + \
                glob(f"{directory}/**/*.mpga", recursive=True) + \
                glob(f"{directory}/**/*.x-hx-aac-adts", recursive=True)
    return filepaths

def pad_or_truncate(audio, target_length):
    target_samples = int(target_length * SR)
    if len(audio) > target_samples:
        return audio[:target_samples]
    elif len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)))
    else:
        return audio


def compute_mel_spectrogram(audio_path):
    try:
        audio = librosa.load(audio_path, sr=SR)[0]
    except:
        return None
    if len(audio) < ((TARGET_LENGTH / 2) * SR):
        return None
    
    audio = pad_or_truncate(audio, TARGET_LENGTH)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.clip(mel_spec_db, DB_CLIP_MIN, 0)
    return [audio_path, mel_spec_db.flatten()]

def normalized_similarity(query, reference, background_set):
    raw_similarity = cosine_similarity(query.reshape(1, -1), reference.reshape(1, -1))[0][0]
    biases = [cosine_similarity(query.reshape(1, -1), bg.reshape(1, -1))[0][0] for bg in background_set]
    bias_term = np.mean(sorted(biases, reverse=True)[:K_NEAREST])
    return raw_similarity - BETA * bias_term

@nb.njit(nogil=True, parallel=True)
def compute_similarity_matrix_nb(descriptors, background_set, beta, k_nearest, n):
    similarity_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in nb.prange(n):
        for j in range(n):
            raw_similarity = cosine_similarity(descriptors[i], descriptors[j])
            biases = np.empty((background_set.shape[0],), dtype=np.float32)
            
            for k in range(background_set.shape[0]):
                biases[k] = cosine_similarity(descriptors[i], background_set[k])
            
            bias_term = np.mean(np.sort(biases)[-k_nearest:])
            similarity_matrix[i, j] = raw_similarity - beta * bias_term

    return similarity_matrix


def deduplicate(audio_files):
    # Load and process audio files

    # Compute descriptors
    data = process_map(compute_mel_spectrogram, audio_files, max_workers=32, chunksize=100)
    file_names, mel_descriptors = zip(*[d for d in data if d is not None])
    
    mel_descriptors = np.array(mel_descriptors)
    
    assert mel_descriptors.shape[0] == len(file_names)
    
    return list(zip(file_names, mel_descriptors))
        
    # # Compute background set for normalization


# Example usage

def calculate_overlap(filenames, mel_descriptors):
    background_set_mel = mel_descriptors[np.random.choice(mel_descriptors.shape[0], 1000, replace=False)]
    
    # Compute similarity matrices
    n = mel_descriptors.shape[0]
    similarity_matrix_mel = compute_similarity_matrix_nb(mel_descriptors, background_set_mel, BETA, K_NEAREST, n)
    
    # Retrieve potential duplicates
    potential_duplicates_mel = np.argwhere(similarity_matrix_mel > TAU_MEL)
    
    potential_duplicates_mel = [(filenames[i], filenames[j], similarity_matrix_mel[i, j]) for i, j in potential_duplicates_mel]
    
    
    return potential_duplicates_mel


if args.option == "build":

    audio_dir = f"/scratch-shared/gwijngaard/data/{args.dir}"
    audio_files = get_filepaths(audio_dir)
    # audio_files = audio_files[:1000]  # Limit to first 100 files for demonstration
    # with open("tmp_dump.pkl", "wb") as f:
    #     pickle.dump(audio_files, f)
    mel_descriptors = deduplicate(audio_files)

    print("Length mel descriptors", len(mel_descriptors))
    with open(f"/home/gwijngaard/projects/clap-surveypaper/mel/{args.dir}.pkl", "wb") as f:
        pickle.dump(mel_descriptors, f)
else:
    if args.dir == "all":
        all_data = []
        for file in os.listdir("/home/gwijngaard/projects/clap-surveypaper/mel/"):
            with open(f"/home/gwijngaard/projects/clap-surveypaper/mel/{file}", "rb") as f:
                data = pickle.load(f)
                print("Length mel descriptors", len(data))
                all_data.extend(data)
        filenames, mel_descriptors = zip(*all_data)
    else:
        with open(f"/home/gwijngaard/projects/clap-surveypaper/mel/{args.dir}.pkl", "rb") as f:
            data = pickle.load(f)
            filenames, mel_descriptors = zip(*data)
            print("Length mel descriptors", len(data))
    overlap = calculate_overlap(filenames, np.array(mel_descriptors)) 
    
    with open(f"/home/gwijngaard/projects/clap-surveypaper/overlap/{args.dir}.pkl", "wb") as f:
        pickle.dump(overlap, f)
                


# print("Potential duplicates using CLAP Descriptors:", duplicates_clap)
