from glob import glob 
import os
import string
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
from dotenv import load_dotenv
from collections import Counter
from pydub import AudioSegment
from datasets import *
from multiprocessing import Pool
load_dotenv()

class AudioStats():
    def __init__(self, dataset_name, storage_path):
        self.dataset_name = dataset_name
        self.storage_path = storage_path

    def __call__(self):
        directory = f"{self.storage_path}/{self.dataset_name}"
        total_count, total_duration, total_bytes, frequency_count, extension_count = self.calculate_audio_metrics(directory)
        total_gb = total_bytes / (1024**3)  # convert bytes to gigabytes
        total_hours = total_duration / 3600  # convert seconds to hours

        print(f"Total duration of audio files: {total_hours:.2f} hours")
        print(f"Total size of audio files: {total_gb:.2f} GB")
        print("Frequency distribution (Hz):")
        for freq, count in frequency_count.items():
            print(f"{freq} Hz: {count} files")
        print("File extension distribution:")
        for ext, count in extension_count.items():
            print(f"{ext}: {count} files")
        
        return total_count, total_hours, total_gb, frequency_count, extension_count

    def get_audio_properties(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            duration = len(audio) / 1000  # duration in seconds
            frame_rate = audio.frame_rate
            size = os.path.getsize(file_path)
            return duration, frame_rate, size
        except Exception as e:
            return 0, 0, 0

    def calculate_audio_metrics(self, directory):
        total_duration = 0
        total_size = 0  # in bytes
        total_count = 0
        frequency_count = {}  # Dictionary for frequency count
        extension_count = {}  # Dictionary for file extension count

        audio_files = glob(f"{directory}/audio/**/*.wav", recursive=True) + \
                    glob(f"{directory}/audio/**/*.mp3", recursive=True) + \
                    glob(f"{directory}/audio/**/*.flac", recursive=True) + \
                    glob(f"{directory}/audio/**/*.ogg", recursive=True) + \
                    glob(f"{directory}/audio/**/*.m4a", recursive=True) + \
                    glob(f"{directory}/audio/**/*.aiff", recursive=True) + \
                    glob(f"{directory}/audio/**/*.aif", recursive=True) + \
                    glob(f"{directory}/audio/**/*.au", recursive=True) + \
                    glob(f"{directory}/audio/**/*.3gp", recursive=True) + \
                    glob(f"{directory}/audio/**/*.3gpp", recursive=True) + \
                    glob(f"{directory}/audio/**/*.mp4", recursive=True) + \
                    glob(f"{directory}/audio/**/*.mpeg", recursive=True) + \
                    glob(f"{directory}/audio/**/*.mpga", recursive=True) + \
                    glob(f"{directory}/audio/**/*.x-hx-aac-adts", recursive=True)
        # audio_files = glob(f"{directory}/audio/**/*.wav", recursive=True)
        audio_files = audio_files
        if len(audio_files) > 10_000:
            with Pool(16) as pool:
                out = []
                for file_path in tqdm(audio_files, total=len(audio_files)):
                    result = pool.apply_async(self.get_audio_properties, (file_path,))
                    out.append(result.get())
        else:
            out = list(map(self.get_audio_properties, audio_files))
        print("Done with audio properties")
        # duration, frame_rate = zip(*out)
        total_duration = 0
        total_size = 0
        # Process file sizes and extensions in bulk first
        extensions = [os.path.splitext(f)[1].lower() for f in audio_files]
        
        # Calculate totals and counts
        durations, frame_rates, file_sizes = zip(*out)
        total_duration = sum(durations)
        total_size = sum(file_sizes)
        total_count = len(audio_files)
        
        # Count frequencies using Counter
        frequency_count = Counter(frame_rates)
        extension_count = Counter(extensions)

        return total_count, total_duration, total_size, frequency_count, extension_count

class TextStats():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = globals()[dataset_name]() # run function with the same name
    
    def __call__(self):
        # Prepare captions
        self.dataset = self.dataset.dropna(subset=['caption'])

        captions_cleaned = self.dataset['caption'].astype(str).str.lower() \
            .str.replace('[{}]'.format(string.punctuation), '')
        
        caption_cleaned_split = captions_cleaned.str.split()

        average_words = caption_cleaned_split.apply(len).mean()

        average_characters = captions_cleaned.str.len().mean()
        average_characters_std = captions_cleaned.str.len().std()
        
        # standard deviation of the number of words in the captions
        std_dev = caption_cleaned_split.apply(len).std()

        # Calculate number of unique words
        unique_words = set(word for caption in caption_cleaned_split for word in caption)
        

        # Calculate average amount of words in caption column
        print(f"Total number of captions: {len(self.dataset)}")
        print(f"Average number of words per caption: {average_words:.2f}")
        print(f"Standard deviation of the number of words in the captions: {std_dev:.2f}")
        print(f"Number of unique words: {len(unique_words)}")

        return len(self.dataset), average_characters, average_characters_std, len(unique_words)


if __name__ == '__main__':
    storage_path = os.getenv("STORAGE_PATH")
    dataset_path = os.getenv("DATASET_PATH")
    parser = argparse.ArgumentParser(description='Calculate metrics of audio files in a directory')
    parser.add_argument('--dataset', type=str, default="MULTIS", help='Dataset name')
    parser.add_argument('--storage_path', type=str, default=storage_path, help="Base path")
    parser.add_argument('--dataset_path', type=str, default=dataset_path, help="Base path")
    

    args = parser.parse_args()
    print("Calculating ... ", args.dataset)
    text_out = TextStats(args.dataset)()
    audio_out = AudioStats(args.dataset, args.storage_path)()
    dataset_length, average_characters, average_characters_std, unique_words = text_out
    total_count, total_duration, total_size, frequency_count, extension_count = audio_out

    # with open(f"{args.dataset_path}/stats_words.csv", "a") as f:
    #     f.write(f"{args.dataset}, {dataset_length}, {average_characters}, {average_characters_std}, {unique_words}\n")
    
    with open(f"{args.dataset_path}/stats.csv", "a") as f:
        f.write(f"{args.dataset}, {dataset_length}, {average_characters}, {average_characters_std}, {unique_words}, {total_count}, {total_duration}, {total_size}, {frequency_count}, {extension_count}\n")

    # AudioStats("SoundingEarth", "/storage/data/")()
    # AudioStats("VGGSound", "/storage/data/")()    
    # TextStats("VGGSound", "/storage/data/")()
    # AudioStats("ClothoAQA", "/storage/data/")()
    # TextStats("ClothoAQA", "/storage/data/")()
