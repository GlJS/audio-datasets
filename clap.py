import os
import torch
from glob import glob
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
load_dotenv()

set_seed(42)

storage_path = os.getenv("STORAGE_PATH")


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.samples = glob(f"{root_dir}/**/*.wav", recursive=True) + \
                    glob(f"{root_dir}/**/*.mp3", recursive=True) + \
                    glob(f"{root_dir}/**/*.flac", recursive=True) + \
                    glob(f"{root_dir}/**/*.ogg", recursive=True) + \
                    glob(f"{root_dir}/**/*.m4a", recursive=True) + \
                    glob(f"{root_dir}/**/*.aiff", recursive=True) + \
                    glob(f"{root_dir}/**/*.aif", recursive=True) + \
                    glob(f"{root_dir}/**/*.au", recursive=True) + \
                    glob(f"{root_dir}/**/*.3gp", recursive=True) + \
                    glob(f"{root_dir}/**/*.3gpp", recursive=True) + \
                    glob(f"{root_dir}/**/*.mp4", recursive=True) + \
                    glob(f"{root_dir}/**/*.mpeg", recursive=True) + \
                    glob(f"{root_dir}/**/*.mpga", recursive=True) + \
                    glob(f"{root_dir}/**/*.x-hx-aac-adts", recursive=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading file {audio_path}: {e}")
            return audio_path, None
        if sr != 48_000:
            waveform = torchaudio.transforms.Resample(sr, 48_000)(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)
        return audio_path, waveform

def collate_fn(batch):
    batch = list(filter(lambda x: x[1] is not None, batch))

    audio_paths, waveforms = zip(*batch)
    waveforms = [waveform.numpy() for waveform in waveforms]
    return list(audio_paths), waveforms


# Load CLAP model and processor
model = ClapModel.from_pretrained("laion/larger_clap_general")
processor = ClapProcessor.from_pretrained("laion/larger_clap_general")

model.eval()

# Initialize dataset and dataloader
root_dir = '/storage/data/'
for directory in tqdm(os.listdir(root_dir), desc="Directories", position=0):
    if directory not in ['SoundingEarth']:
        continue
    if os.path.isdir(os.path.join(root_dir, directory)):
        os.makedirs(f"embeddings/{directory}", exist_ok=True)
        root_sub_dir = os.path.join(root_dir, directory)
        if os.path.exists(f"embeddings/{directory}/0.pkl"):
            continue

        dataset = AudioDataset(root_sub_dir)
        data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=48)

        together = np.zeros((0, 512))
        all_audio_paths = []

        for idx, batch in enumerate(tqdm(data_loader, position=1, desc=f"Extracting embeddings for {directory}")):
            audio_paths, audios = batch
            inputs = processor(audios=audios, return_tensors="pt", sampling_rate=48_000)
            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs)
            together = np.concatenate((together, audio_embed.cpu().numpy()), axis=0)
            all_audio_paths += audio_paths
        combined = [(all_audio_paths[i], together[i]) for i in range(len(all_audio_paths))]
        with open(f"embeddings/{directory}/0.pkl", 'wb') as f:
            pickle.dump(combined, f)

