import os
import webdataset as wds
from tqdm import tqdm
import json
import tarfile
import io
import math
import argparse

def main():
    parser = argparse.ArgumentParser(description='Fix data in tar files for a specific dataset')
    parser.add_argument('--dataset', default="BBCSoundEffects", help='Name of the dataset directory to process')
    args = parser.parse_args()

    root_dir = '/scratch-shared/gwijngaard/data'
    dataset_dir = os.path.join(root_dir, args.dataset)

    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist")
        return

    for split in os.listdir(dataset_dir):
        try:

            # Load sizes.json
            sizes_path = os.path.join(dataset_dir, split, 'sizes.json')
            if os.path.exists(sizes_path):
                with open(sizes_path, 'r') as f:
                    sizes_data = json.load(f)

            tar_files = [f for f in os.listdir(os.path.join(dataset_dir, split)) if f.endswith('.tar')]
            
            for tar_file in tqdm(tar_files, desc=f"Processing {args.dataset}/{split}"):
                tar_path = os.path.join(dataset_dir, split, tar_file)
                
                # Create a temporary tar file
                temp_tar_path = tar_path + '.tmp'
                
                removed_count = 0
                with tarfile.open(tar_path, 'r') as src_tar, tarfile.open(temp_tar_path, 'w') as dst_tar:
                    # Group files by base name (without extension)
                    files_by_base = {}
                    for member in src_tar.getmembers():
                        base = os.path.splitext(member.name)[0]
                        if base not in files_by_base:
                            files_by_base[base] = []
                        files_by_base[base].append(member)

                    # Process each group of files
                    for base, members in files_by_base.items():
                        should_keep = True
                        json_data = None
                        
                        # First check if we need to keep this group
                        for member in members:
                            if member.name.endswith('.json'):
                                f = src_tar.extractfile(member)
                                json_data = json.load(f)
                                
                                if "caption" in json_data:
                                    json_data["text"] = json_data.pop("caption")
                                    
                                if "text" in json_data:
                                    if isinstance(json_data["text"], (float, int)) and (isinstance(json_data["text"], float) and math.isnan(json_data["text"]) or isinstance(json_data["text"], int)):
                                        should_keep = False
                                        removed_count += 1
                                        break
                                    text = json_data["text"].lower()
                                    
                                    # Apply dataset-specific transformations
                                    if args.dataset == "SoundDescs" and text.startswith("generate metadata "):
                                        text = text[len("generate metadata "):]
                                    elif args.dataset == "WavText" and text.startswith("generate metadata "):
                                        text = text[len("generate metadata "):]
                                    elif args.dataset == "Audiocaps" and text.startswith("generate audio caption "):
                                        text = text[len("generate audio caption "):]
                                    elif args.dataset == "FSD50k" and text.startswith("this is a sound of "):
                                        text = text[len("this is a sound of "):]
                                    elif args.dataset == "Audioset" and text.startswith("this is a sound of "):
                                        text = text[len("this is a sound of "):]
                                    elif args.dataset == "CochlScene" and text.startswith("this acoustic scene is "):
                                        text = text[len("this acoustic scene is "):]
                                        
                                    json_data["text"] = text

                        # If we should keep this group, write all its files
                        if should_keep:
                            for member in members:
                                if member.name.endswith('.json'):
                                    # Write modified JSON
                                    json_bytes = json.dumps(json_data).encode('utf-8')
                                    json_io = io.BytesIO(json_bytes)
                                    new_member = tarfile.TarInfo(name=member.name)
                                    new_member.size = len(json_bytes)
                                    dst_tar.addfile(new_member, json_io)
                                else:
                                    # Copy non-JSON files as-is
                                    f = src_tar.extractfile(member)
                                    dst_tar.addfile(member, f)
                
                # Update sizes.json if files were removed
                if removed_count > 0 and os.path.exists(sizes_path):
                    sizes_data[tar_file] = sizes_data[tar_file] - removed_count
                    with open(sizes_path, 'w') as f:
                        json.dump(sizes_data, f, indent=4)
                
                # Replace original tar with modified version
                os.replace(temp_tar_path, tar_path)
                print(f"Processed {tar_path}")
                
        except Exception as e:
            print(f"Error processing {args.dataset}/{split}: {str(e)}")
            print(f"json_data: {json_data}")
            if os.path.exists(temp_tar_path):
                os.remove(temp_tar_path)
            continue

if __name__ == "__main__":
    main()
