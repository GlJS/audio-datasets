import os
import re
# os.environ["HF_HOME"] = "/mnt/um-share-gijs/gijs/cache"
# os.environ["HF_HOME"] = "/root/share/cache"
os.environ["HF_HOME"] = "/scratch-shared/gwijngaard/cache"
os.environ["HF_TOKEN"] = "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"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import sys
from datasets import *
import json
from typing import List

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="AdobeAuditionSFX")
parser.add_argument("--full_data", default=True, action="store_true")
args = parser.parse_args()

print(args)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

if args.full_data:
    with open("visualization/categories200.json", "r") as f:
        categories = json.load(f)
else:
    categories = [
        "Human sounds",
        "Source-ambiguous sounds",
        "Animal",
        "Sounds of things",
        "Music", 
        "Natural sounds",
        "Channel, environment and background"
    ]

known_datasets = ["ACalt4", "AdobeAuditionSFX", "AFAudioSet", "AnimalSpeak", "AudioAlpaca",
                 "AudioCaps", "AudioCaption", "AudioCondition", "AudioDiffCaps", "AudioEgoVLP",
                 "AudioHallucination", "AudioLog", "Audioset", "AudiosetStrong", "Audiostock",
                 "AudioTime", "AutoACD", "BATON", "BBCSoundEffects", "BigSoundBank", "CAPTDURE",
                 "CHiMEHome", "Clotho", "ClothoAQA", "ClothoChatGPTMixup", "ClothoDetail",
                 "ClothoEntailment", "ClothoMoment", "ClothoV2GPT", "CompAR", "DAQA",
                 "EpidemicSoundEffects", "ESC50", "EzAudioCaps", "FAVDBench", "FindSounds",
                 "Freesound", "FreeToUseSounds", "FSD50k", "LAION630k", "LASS", "MACS",
                 "MULTIS", "NonSpeech7k", "Paramount", "PicoAudio", "ProSoundEffects",
                 "RichDetailAudioTextSimulation", "SonnissGameEffects", "SonycUST",
                 "SoundBible", "SoundDescs", "SoundingEarth", "SoundJay", "SoundVECaps",
                 "SpatialSoundQA", "Syncaps", "TextToAudioGrounding", "VGGSound",
                 "WavCaps", "WavText5K", "WeSoundEffects", "Zapsplat", "mAQA"]
if args.dataset not in known_datasets:
    raise ValueError(f"Dataset {args.dataset} not found!")

model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model.config.pad_token_id = tokenizer.pad_token_id
# resize token embeddings to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

def create_structured_prompt(sentence: str, categories: List[str]) -> str:
    categories_list = "\n".join([f"- {cat}" for cat in categories])
    return f"""You are a helpful chatbot who helps label a dataset. Classify the following sentence into one of the AudioSet categories listed below.

Sentence: "{sentence}"

Available categories:
{categories_list}

Instructions:
1. Choose exactly one category from the list above
2. Return only the exact full category name, nothing else. Not only part of the category name.
3. If unsure, choose the most relevant category

Category:"""

def load_dataset(dataset_name):
    if dataset_name in globals():
        dataset = globals()[dataset_name]()
    else:
        raise ValueError(f"Dataset {dataset_name} not found!")
    return dataset



class AudioData(Dataset):
    def __init__(self, dataset):
        self.data = load_dataset(dataset)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]["caption"]
        categories_info = "\n".join([f"- {key}" for key in categories])
        
        # We'll just return the raw inputs now, as we'll handle the LLM call in the main loop
        return self.data.iloc[idx]["file_name"], sentence, categories_info, self.data.iloc[idx]["split"]

def collate_fn(batch):
    file_names = [x[0] for x in batch]
    sentences = [x[1] for x in batch]
    categories_info = batch[0][2]  # Same for all items in batch
    splits = [x[3] for x in batch]
    return {"file_names": file_names, "sentences": sentences, "categories_info": categories_info, "splits": splits}

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# for dataset in tqdm(known_datasets, desc="Datasets", position=0):
dataset_labels = []
dataset = args.dataset
loader = DataLoader(AudioData(dataset), batch_size=64, shuffle=False, num_workers=16, collate_fn=collate_fn)


for data in tqdm(loader, desc=dataset, position=1):
    file_names = data["file_names"]
    sentences = data["sentences"]
    splits = data["splits"]

    # Process in batches
    batch_prompts = [create_structured_prompt(sentence, categories) for sentence in sentences]
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=24,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    outputs = tokenizer.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    for idx, output in enumerate(outputs):
        # Clean and match the output to valid categories
        output = output.strip()
        # Find first occurrence of any category in the output
        category = None
        output_lower = output.lower()
        min_pos = float('inf')
        for cat in categories:
            cat_lower = cat.lower()
            # Check if the category appears as a whole word using word boundaries
            for match in re.finditer(r'\b' + re.escape(cat_lower) + r'\b', output_lower):
                pos = match.start()
                if pos < min_pos:
                    min_pos = pos
                    category = cat
        # Remove newlines from output
        output = output.replace('\n', ' ').strip()
        dataset_labels.append([dataset, file_names[idx], sentences[idx], splits[idx], output, category])
    

df = pd.DataFrame(dataset_labels, columns=["dataset", "file_name", "sentence", "split", "output", "label"])
df.to_csv(f"output_cats/{dataset}.csv", index=False)
