import os
import webdataset as wds
from tqdm import tqdm
import json
import tarfile
import io
import math
import argparse
import shutil
import glob

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Name of dataset to process')
parser.add_argument('--batch_size', type=int, default=512, help='Number of samples per tar file')
args = parser.parse_args()

# Define old and new root directories
old_root_dir = '/scratch-shared/gwijngaard/data'
scratch_dirs = glob.glob('/scratch-node/gwijngaard*')
new_root_dir = sorted(scratch_dirs)[-1]

for directory in os.listdir(old_root_dir):
    if os.path.isdir(os.path.join(old_root_dir, directory)):
        if directory != args.dataset:
            continue
            
        for split in os.listdir(os.path.join(old_root_dir, directory)):
            try:
                # Get list of tar files
                tar_files = [f for f in os.listdir(os.path.join(old_root_dir, directory, split)) if f.endswith('.tar')]
                                
                # Create new directory for output
                new_split_dir = os.path.join(new_root_dir, directory, split)
                os.makedirs(new_split_dir, exist_ok=True)
                
                # Track total samples and current batch
                total_samples = 0
                current_batch = []
                current_batch_size = 0
                batch_number = 0
                
                # Load sizes.json if it exists
                sizes_path = os.path.join(new_split_dir, 'sizes.json')
                sizes_data = {}
                
                for tar_file in tqdm(tar_files, desc=f"Processing {directory}/{split}"):
                    tar_path = os.path.join(old_root_dir, directory, split, tar_file)
                    
                    with tarfile.open(tar_path, 'r') as src_tar:
                        # Group files by base name
                        files_by_base = {}
                        for member in src_tar.getmembers():
                            base = os.path.splitext(member.name)[0]
                            if base not in files_by_base:
                                files_by_base[base] = []
                            files_by_base[base].append(member)
                        
                        # Process each group
                        for base, members in files_by_base.items():
                            # Extract the files' content while the tar is still open
                            member_contents = []
                            for member in members:
                                f = src_tar.extractfile(member)
                                member_contents.append((member, f.read()))
                            current_batch.append(member_contents)
                            current_batch_size += 1
                            
                            # When batch is full, write to new tar file
                            if current_batch_size >= args.batch_size:
                                new_tar_path = os.path.join(new_split_dir, f"{batch_number}.tar")
                                with tarfile.open(new_tar_path, 'w') as dst_tar:
                                    for members_content in current_batch:
                                        for member, content in members_content:
                                            info = member
                                            f = io.BytesIO(content)
                                            dst_tar.addfile(info, f)
                                
                                sizes_data[f"{batch_number}.tar"] = current_batch_size
                                batch_number += 1
                                current_batch = []
                                current_batch_size = 0
                
                # Write remaining samples
                if current_batch:
                    new_tar_path = os.path.join(new_split_dir, f"{batch_number}.tar")
                    with tarfile.open(new_tar_path, 'w') as dst_tar:
                        for members_content in current_batch:
                            for member, content in members_content:
                                info = member
                                f = io.BytesIO(content)
                                dst_tar.addfile(info, f)
                    
                    sizes_data[f"{batch_number}.tar"] = current_batch_size
                
                # Write new sizes.json
                with open(sizes_path, 'w') as f:
                    json.dump(sizes_data, f, indent=4)
                                
                # Clean up old tar files
                old_split_dir = os.path.join(old_root_dir, directory, split)
                for tar_file in tar_files:
                    os.remove(os.path.join(old_split_dir, tar_file))

                # Move new files back to old location
                for filename in os.listdir(new_split_dir):
                    src_path = os.path.join(new_split_dir, filename)
                    dst_path = os.path.join(old_split_dir, filename)
                    shutil.move(src_path, dst_path)

                # Remove temporary directory
                shutil.rmtree(new_split_dir)
                
            except Exception as e:
                print(f"Error processing {directory}/{split}: {str(e)}")
                continue
