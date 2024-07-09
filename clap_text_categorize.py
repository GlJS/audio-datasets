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
from datasets import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="all")
parser.add_argument("--speech", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

print(args)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


known_datasets = ["AnimalSpeak", "AudioCaps", "AudioCaption", "AudioDiffCaps", "Audioset", 
                    "CAPTDURE", "Clotho", "ClothoAQA", "ClothoDetail", "DAQA", "FAVDBench", 
                    "FSD50k", "MACS", "MULTIS", "SoundDescs", "SoundingEarth", 
                    "TextToAudioGrounding", "WavText5K", "mAQA"]
if args.data not in known_datasets and args.data != "all":
    raise ValueError(f"Dataset {args.data} not found!")

if args.speech:
    categories = [
        "Human speech",
        "Other human sounds",
        "Source-ambiguous sounds",
        "Animal",
        "Sounds of things",
        "Music",
        "Natural sounds",
        "Channel, environment and background"
    ]
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

def load_dataset_output():
    dfs = []
    for f in os.listdir("output_cats/"):
        if f.endswith(".csv"):
            df = pd.read_csv(f"output_cats/{f}")
            dfs.append(df)
    df = pd.concat(dfs)
    return df

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

class AudioData(Dataset):
    def __init__(self, dataset):
        if args.data == "all":
            self.data = load_dataset_output()
        else:
            self.data = load_dataset(dataset)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if args.data == "all":
            sentence = self.data.iloc[idx]["sentence"]
        else:
            sentence = self.data.iloc[idx]["caption"]
        message = f"Classify the following sentence into one of the AudioSet top-level ontology classes:\n\nSentence: \"{sentence}\"\n\nOntology classes: {'; '.join(categories)}\n\nOnly return the 1 category and nothing else. Give the full class name. \n\nCategory:"
        data = tokenizer(message, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
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
loader = DataLoader(AudioData(dataset), batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_fn)


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
        top_p=0.9
        
    )
    generated_ids = generated_ids[:, input_ids.size(1):]
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = [o.strip() for o in outputs]
    for idx, res in enumerate(outputs):
        all_res = [cat for cat in categories if cat.lower() in res.lower()]
        if len(all_res) > 0:
            res = all_res[0]
        else:
            res = None
        dataset_labels.append([dataset, file_names[idx], sentences[idx], splits[idx], res])
    

df = pd.DataFrame(dataset_labels, columns=["dataset", "file_name", "sentence", "split", "label"])
if args.speech:
    df.to_csv(f"output_cats_speech/{dataset}.csv", index=False)
else:
    df.to_csv(f"output_cats/{dataset}.csv", index=False)