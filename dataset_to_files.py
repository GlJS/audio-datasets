import os
import json
import pandas as pd
from tqdm import tqdm
import datasets
import argparse
from dotenv import load_dotenv
import subprocess
import multiprocessing
from functools import partial

load_dotenv()

dataset_path = os.getenv("DATASET_PATH")
storage_path = os.getenv("STORAGE_PATH")
num_cores = 16

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def convert_audio_to_flac(input_path, output_path):
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-ac', '1', '-ar', '48000', '-sample_fmt', 's16', '-c:a', 'flac', output_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        try: 
            ext = input_path.split('.')[-1]
            subprocess.run(['ffmpeg', '-f', ext, '-i', input_path, '-ac', '1', '-ar', '48000', '-sample_fmt', 's16', '-c:a', 'flac', output_path], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error converting {input_path} to FLAC: {e}")
            raise
    # delete the original file
    # os.remove(input_path)

def process_row(row, dataset_name, output_folder):
    original_file_name = row['file_name']
    caption = row['caption']
    split = row['split']

    # Create split folder
    split_folder = os.path.join(output_folder, split)

    # Generate new file names
    new_file_name = f"{row['index']}"
    
    # Convert audio to FLAC
    input_audio_path = os.path.join(storage_path, dataset_name, "audio", original_file_name)
    output_audio_path = os.path.join(split_folder, f"{new_file_name}.flac")
    
    try:
        convert_audio_to_flac(input_audio_path, output_audio_path)
    except Exception as e:
        print(f"Error converting audio file {original_file_name}: {str(e)}")
        return None

    # Create JSON metadata
    metadata = {
        "file_name": f"{new_file_name}.flac",
        "original_file_name": original_file_name,
        "caption": caption,
        "dataset": dataset_name,
        "split": split
    }

    json_path = os.path.join(split_folder, f"{new_file_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return split

def process_dataset(dataset_name, output_folder):
    print(f"Processing dataset: {dataset_name}")
    dataset_func = getattr(datasets, dataset_name)
    df = dataset_func()

    ensure_dir(output_folder)

    # Convert DataFrame to list of dictionaries
    data_list = df.reset_index().to_dict('records')

    splits = df['split'].unique()
    for split in splits:
        split_folder = os.path.join(output_folder, split)
        ensure_dir(split_folder)

    # Create a pool of workers
    pool = multiprocessing.Pool(processes=num_cores)

    # Prepare the partial function for multiprocessing
    process_func = partial(process_row, dataset_name=dataset_name, output_folder=output_folder)

    # Process rows in parallel
    results = list(tqdm(pool.imap(process_func, data_list, chunksize=10), total=len(data_list)))

    # Close the pool
    pool.close()
    pool.join()

    # Count the number of files in each split
    file_counters = {'train': 0, 'valid': 0, 'test': 0}
    for split in results:
        if split is not None:
            file_counters[split] += 1

    print(f"Processed files: Train: {file_counters['train']}, Valid: {file_counters['valid']}, Test: {file_counters['test']}")

def main():
    parser = argparse.ArgumentParser(description="Convert datasets to FLAC and JSON files")
    parser.add_argument("--dataset", type=str, default="Freesound", help="Name of the dataset to process")
    parser.add_argument("--output_folder", type=str, default=None, help="Path to the output folder")
    args = parser.parse_args()

    if args.output_folder is None:
        output_folder = os.path.join(storage_path, args.dataset, "processed")
    else:
        output_folder = os.path.join(args.output_folder, args.dataset, "processed")

    if args.dataset:
        process_dataset(args.dataset, output_folder)
    else:
        print("Please specify a dataset name using the --dataset argument")

if __name__ == "__main__":
    main()
