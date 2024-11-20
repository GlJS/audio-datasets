import os
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="AnimalSpeak")
parser.add_argument("--all_labels", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

print(args)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

with open('visualization/cli.json', 'r') as f:
    categories_dict = json.load(f)

if not args.all_labels:
    categories = [
        "Human sounds",
        "Source-ambiguous sounds",
        "Animal",
        "Sounds of things",
        "Music", 
        "Natural sounds",
        "Channel, environment and background"
    ]
    categories_dict = [k for k in categories_dict.keys() if k in categories]

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
if args.data not in known_datasets:
    raise ValueError(f"Dataset {args.data} not found!")

model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model.config.pad_token_id = tokenizer.pad_token_id
# resize token embeddings to match new tokenizer
model.resize_token_embeddings(len(tokenizer))


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
            
        categories_info = "\n".join([f"{key}" for key in categories_dict])
        message = f"You are a helpful chatbot who helps label a dataset. Classify the following sentence into one of the AudioSet categories based on their descriptions:\n\nSentence: \"{sentence}\"\n\nCategories and their descriptions:\n{categories_info}\n\nOnly return the exact category name and nothing else.\n\nCategory:"
        
        data = tokenizer(message, return_tensors="pt", padding="max_length", max_length=256)
        return self.data.iloc[idx]["file_name"], sentence, data, self.data.iloc[idx]["split"]

def collate_fn(batch):
    file_names = [x[0] for x in batch]
    sentences = [x[1] for x in batch]
    input_ids = torch.cat([x[2]["input_ids"] for x in batch])
    attention_mask = torch.cat([x[2]["attention_mask"] for x in batch])
    splits = [x[3] for x in batch]
    return {"file_names": file_names, "sentences": sentences, "input_ids": input_ids, "attention_mask": attention_mask, "splits": splits}

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# for dataset in tqdm(known_datasets, desc="Datasets", position=0):
dataset_labels = []
dataset = args.data
loader = DataLoader(AudioData(dataset), batch_size=128, shuffle=False, num_workers=18, collate_fn=collate_fn)


for data in tqdm(loader, desc=dataset, position=1):
    file_names = data["file_names"]
    sentences = data["sentences"]
    input_ids = data["input_ids"].to(model.device)
    attention_mask = data["attention_mask"].to(model.device)
    splits = data["splits"]

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        
    )
    generated_ids = generated_ids[:, input_ids.size(1):]
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = [o.strip() for o in outputs]
    for idx, res in enumerate(outputs):
        all_res = [cat for cat in categories_dict.keys() if cat.lower() in res.lower()]
        if len(all_res) > 0:
            res = all_res[0]
        else:
            res = None
        dataset_labels.append([dataset, file_names[idx], sentences[idx], splits[idx], res])
    

df = pd.DataFrame(dataset_labels, columns=["dataset", "file_name", "sentence", "split", "label"])
df.to_csv(f"output_cats/{dataset}.csv", index=False)