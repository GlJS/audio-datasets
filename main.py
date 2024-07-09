from glob import glob 
import os
import string
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import argparse
from dotenv import load_dotenv
from pydub import AudioSegment
from datasets import *
load_dotenv()

class AudioStats():
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

    def __call__(self):
        directory = f"{self.dataset_path}/{self.dataset_name}"
        total_hours, total_bytes, frequency_count, extension_count = self.calculate_audio_metrics(directory)
        total_gb = total_bytes / (1024**3)  # convert bytes to gigabytes
        total_hours = total_hours / 3600  # convert seconds to hours

        print(f"Total duration of audio files: {total_hours:.2f} hours")
        print(f"Total size of audio files: {total_gb:.2f} GB")
        print("Frequency distribution (Hz):")
        for freq, count in frequency_count.items():
            print(f"{freq} Hz: {count} files")
        print("File extension distribution:")
        for ext, count in extension_count.items():
            print(f"{ext}: {count} files")

    def get_audio_properties(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            duration = len(audio) / 1000  # duration in seconds
            frame_rate = audio.frame_rate
            return duration, frame_rate
        except Exception as e:
            return 0, 0

    def calculate_audio_metrics(self, directory):
        total_duration = 0
        total_size = 0  # in bytes
        frequency_count = {}  # Dictionary for frequency count
        extension_count = {}  # Dictionary for file extension count

        audio_files = glob(f"{directory}/**/*.wav", recursive=True) + \
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
        # audio_files = glob(f"{directory}/**/*.wav", recursive=True)
        # audio_files = audio_files[:1000]
        if len(audio_files) > 100_000:
            out = process_map(self.get_audio_properties, audio_files, max_workers=92, total=len(audio_files), chunksize=1000)
        else:
            out = map(self.get_audio_properties, audio_files)
        duration, frame_rate = zip(*out)
        total_duration = sum(duration)
        total_size = sum(map(os.path.getsize, audio_files))
        for duration, frame_rate, audio_file in zip(duration, frame_rate, audio_files):
            frequency_count[frame_rate] = frequency_count.get(frame_rate, 0) + 1

            extension = os.path.splitext(audio_file)[1].lower()
            extension_count[extension] = extension_count.get(extension, 0) + 1

        return total_duration, total_size, frequency_count, extension_count

class TextStats():
    def __init__(self, dataset_name, dataset_path):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset = globals()[dataset_name]() # run function with the same name
    
    def __call__(self):
        # Prepare captions
        self.dataset = self.dataset.dropna(subset=['caption'])

        captions_cleaned = self.dataset['caption'].astype(str).str.lower() \
            .str.replace('[{}]'.format(string.punctuation), '') \
            .str.split()

        average_words = captions_cleaned.apply(len).mean()
        
        # standard deviation of the number of words in the captions
        std_dev = captions_cleaned.apply(len).std()

        # Calculate number of unique words
        unique_words = set(word for caption in captions_cleaned for word in caption)
        

        # Calculate average amount of words in caption column
        print(f"Total number of captions: {len(self.dataset)}")
        print(f"Average number of words per caption: {average_words:.2f}")
        print(f"Standard deviation of the number of words in the captions: {std_dev:.2f}")
        print(f"Number of unique words: {len(unique_words)}")


if __name__ == '__main__':
    dataset_path = os.getenv("DATASET_PATH")
    storage_path = os.getenv("STORAGE_PATH")
    folder_path = os.getenv("FOLDER_PATH")

    parser = argparse.ArgumentParser(description='Calculate metrics of audio files in a directory')
    parser.add_argument('--folder_path', type=str, default=folder_path, help='Path to the folder containing audio files')
    parser.add_argument('--dataset_path', type=str, default=dataset_path, help="Base path")
    parser.add_argument('--storage_path', type=str, default=storage_path, help="Base path")
    args = parser.parse_args()
    print("Calculating ... ", args.folder_path)
    # AudioStats(args.folder_path, args.dataset_path)()
    TextStats(args.folder_path, args.dataset_path)()
    # AudioStats("SoundingEarth", "/storage/data/")()
    # AudioStats("VGGSound", "/storage/data/")()    
    # TextStats("VGGSound", "/storage/data/")()
    # AudioStats("ClothoAQA", "/storage/data/")()
    # TextStats("ClothoAQA", "/storage/data/")()
