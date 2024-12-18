import os
import webdataset as wds
from tqdm import tqdm
import json
import tarfile

root_dir = '/scratch-shared/gwijngaard/data'

# Track statistics
files_with_caption = []

for directory in os.listdir(root_dir):
    if os.path.isdir(os.path.join(root_dir, directory)):
        for split in os.listdir(os.path.join(root_dir, directory)):
            # Get list of tar files
            tar_files = [f for f in os.listdir(os.path.join(root_dir, directory, split)) if f.endswith('.tar')]
            
            if tar_files:  # Only check if there are any tar files
                # Just check the first tar file
                found_caption = False
                for tar_file in tar_files:
                    tar_path = os.path.join(root_dir, directory, split, tar_file)
                    
                    try:
                        with tarfile.open(tar_path, 'r') as tar:
                            for member in tar.getmembers():
                                if member.name.endswith('.json'):
                                    f = tar.extractfile(member)
                                    json_data = json.load(f)
                                    
                                    if "caption" in json_data:
                                        print(f"{tar_path} still uses 'caption' instead of 'text'")
                                        files_with_caption.append(tar_path)
                                        found_caption = True
                                        break

                    except Exception as e:
                        print(f"Error processing {tar_path}: {str(e)}")

                        
            if not found_caption:
                print(f"Checked {directory}, all items in {split} have text instead of caption")



print(f"\nTotal files with caption: {len(files_with_caption)}")
for f in files_with_caption:
    print(f)