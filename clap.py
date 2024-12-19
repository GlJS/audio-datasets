import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import ClapModel, ClapProcessor, set_seed
from tqdm import tqdm
import torchaudio
from collections import defaultdict
from pydub import AudioSegment
from dotenv import load_dotenv
import numpy as np
import pickle
import webdataset as wds
import json
import io
import argparse
load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="TextToAudioGrounding", help='Name of dataset to process')
args = parser.parse_args()

set_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

storage_path = os.getenv("STORAGE_PATH")

def collate_fn(batch):
    audio_paths = []
    waveforms = []
    texts = []
    
    for sample in batch:
        if "__key__" not in sample:
            continue
            
        # Skip if flac or json data is missing
        if "flac" not in sample or "json" not in sample:
            continue
            
        # Get text from json and validate
        try:
            json_data = json.loads(sample["json"].decode())
            text = json_data["text"]
            if not isinstance(text, str):
                print(sample["__key__"], "is not a string")
                continue
            if not text.strip():  # Skip empty strings
                continue
        except:
            print(sample["__key__"], "is missing json")
            continue
            
        # Validate and load audio data
        try:
            audio_bytes = sample["flac"]
            if audio_bytes is None:
                print(sample["__key__"], "is missing flac")
                continue
            # Convert bytes to waveform
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample if needed
            if sample_rate != 48000:
                resampler = torchaudio.transforms.Resample(sample_rate, 48000)
                waveform = resampler(waveform)
            
            # Convert to numpy array
            waveform = waveform.squeeze().numpy()
        except:
            print(sample["__key__"], "is missing flac")
            continue
            
        audio_paths.append(sample["__key__"])
        waveforms.append(waveform)
        texts.append(text)
            
    return audio_paths, waveforms, texts

# Load CLAP model and processor
model = ClapModel.from_pretrained("laion/larger_clap_general").to(device)
processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

model.eval()

# Initialize dataset and dataloader
root_dir = '/scratch-shared/gwijngaard/data'
directory = args.dataset
for split in tqdm(os.listdir(os.path.join(root_dir, directory)), desc="Splits", position=1):
    
    if os.path.exists(f"audio_embeddings/{directory}/{split}_audio.pkl"):
        continue
    
    # Count the number of tar files in the directory
    num_shards = len([f for f in os.listdir(os.path.join(root_dir, directory, split)) if f.endswith('.tar')]) - 1
    
    # Create the appropriate pattern based on actual number of files
    dataset_path = os.path.join(root_dir, directory, split, f"{{0..{num_shards}}}.tar")
    dataset = wds.WebDataset(dataset_path, empty_check=False)
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=18, pin_memory=True)

    audio_embeddings = np.zeros((0, 512))
    text_embeddings = np.zeros((0, 512))
    all_audio_paths = []

    for idx, batch in enumerate(tqdm(data_loader, position=1, desc=f"Extracting embeddings for {directory}")):
        audio_paths, audios, texts = batch
        
        # Process audio and move to GPU
        audio_inputs = processor(audios=audios, return_tensors="pt", sampling_rate=48_000)
        text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to GPU
        audio_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in audio_inputs.items()}
        text_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in text_inputs.items()}
        
        with torch.no_grad():
            audio_embed = model.get_audio_features(**audio_inputs)
            text_embed = model.get_text_features(**text_inputs)
            
        audio_embeddings = np.concatenate((audio_embeddings, audio_embed.cpu().numpy()), axis=0)
        text_embeddings = np.concatenate((text_embeddings, text_embed.cpu().numpy()), axis=0)
        all_audio_paths += audio_paths
        
    audio_combined = [(all_audio_paths[i], audio_embeddings[i]) for i in range(len(all_audio_paths))]
    text_combined = [(all_audio_paths[i], text_embeddings[i]) for i in range(len(all_audio_paths))]

    # Create directories if they don't exist
    os.makedirs(f"/scratch-shared/gwijngaard/embeddings/audio/{directory}", exist_ok=True)
    os.makedirs(f"/scratch-shared/gwijngaard/embeddings/text/{directory}", exist_ok=True)
    
    with open(f"/scratch-shared/gwijngaard/embeddings/audio/{directory}/{split}_audio.pkl", 'wb') as f:
        pickle.dump(audio_combined, f)
        
    with open(f"/scratch-shared/gwijngaard/embeddings/text/{directory}/{split}_text.pkl", 'wb') as f:
        pickle.dump(text_combined, f)
