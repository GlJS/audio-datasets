import os
import torch
import numpy as np
import torchlibrosa as tl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import argparse
from dotenv import load_dotenv
import webdataset as wds
import json
import io
import torchaudio

load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="TextToAudioGrounding", help='Name of dataset to process')
args = parser.parse_args()

# Set the device based on user input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
SR = 48000
N_MELS = 16
WIN_LENGTH = int(0.128 * SR)  # 128ms window
HOP_LENGTH = WIN_LENGTH * 3 // 4  # 75% overlap
DB_CLIP_MIN = -40
TARGET_LENGTH = 10.242


# Spectrogram extractor
spectrogram_extractor = tl.Spectrogram(
    n_fft=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    window='hann',
    center=True,
    pad_mode='reflect',
    freeze_parameters=True
).to(device)

# Log mel spectrogram extractor
logmel_extractor = tl.LogmelFilterBank(
    sr=SR,
    n_fft=WIN_LENGTH,
    n_mels=N_MELS
).to(device)

def pad_or_truncate(audio, target_length):
    """Pad or truncate audio tensor to target length"""
    target_samples = int(target_length * SR)
    if audio.size(-1) > target_samples:
        return audio[..., :target_samples]
    elif audio.size(-1) < target_samples:
        padding = torch.zeros(target_samples - audio.size(-1), device=device)
        return torch.cat([audio, padding], dim=-1)
    else:
        return audio

def compute_mel_spectrogram(audio):
    """Compute mel spectrogram from audio data"""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    audio = audio.to(device)
    
    # Check if audio is too short
    if audio.size(-1) < ((TARGET_LENGTH / 4) * SR):
        return None
        
    # Pad or truncate audio to target length
    audio = pad_or_truncate(audio, TARGET_LENGTH)
    
    # Compute spectrogram
    spec = spectrogram_extractor(audio.unsqueeze(0))
    
    # Compute log mel spectrogram
    mel_spec = logmel_extractor(spec)
    mel_spec = torch.clamp(mel_spec, min=DB_CLIP_MIN, max=0)
    
    return mel_spec.squeeze().flatten().cpu().numpy()

def collate_fn(batch):
    """Collate function for the dataloader"""
    audio_paths = []
    waveforms = []
    
    for sample in batch:
        if "__key__" not in sample:
            continue
            
        # Skip if flac data is missing
        if "flac" not in sample:
            continue
            
        # Validate and load audio data
        try:
            audio_bytes = sample["flac"]
            if audio_bytes is None:
                continue
                
            # Convert bytes to waveform
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != SR:
                resampler = torchaudio.transforms.Resample(sample_rate, SR)
                waveform = resampler(waveform)
            
            audio_paths.append(sample["__key__"])
            waveforms.append(waveform)
            
        except:
            print(sample["__key__"], "failed to load")
            continue
            
    return audio_paths, waveforms

# Initialize paths
root_dir = '/scratch-shared/gwijngaard/data'
directory = args.dataset

# Process each split
for split in tqdm(os.listdir(os.path.join(root_dir, directory)), desc="Splits"):
    
    # Skip if already processed
    if os.path.exists(f"/scratch-shared/gwijngaard/embeddings/mel/{directory}/{split}_mel.pkl"):
        continue
    
    # Count the number of tar files
    num_shards = len([f for f in os.listdir(os.path.join(root_dir, directory, split)) 
                     if f.endswith('.tar')]) - 1
    
    # Create dataset path pattern
    dataset_path = os.path.join(root_dir, directory, split, f"{{0..{num_shards}}}.tar")
    dataset = wds.WebDataset(dataset_path, empty_check=False)
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, 
                           num_workers=18, pin_memory=True)

    mel_embeddings = []
    all_audio_paths = []

    # Process batches
    for batch in tqdm(data_loader, desc=f"Processing {directory}/{split}"):
        audio_paths, audios = batch
        
        # Compute mel spectrograms
        for path, audio in zip(audio_paths, audios):
            mel_spec = compute_mel_spectrogram(audio)
            if mel_spec is not None:
                mel_embeddings.append(mel_spec)
                all_audio_paths.append(path)

    # Combine paths with embeddings
    mel_combined = list(zip(all_audio_paths, mel_embeddings))

    # Create output directory
    os.makedirs(f"/scratch-shared/gwijngaard/embeddings/mel/{directory}", exist_ok=True)
    
    # Save results
    with open(f"/scratch-shared/gwijngaard/embeddings/mel/{directory}/{split}_mel.pkl", 'wb') as f:
        pickle.dump(mel_combined, f) 