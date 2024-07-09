import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import argparse
from collections import Counter
import matplotlib.cm as cm
import seaborn as sns
sns.set_theme()

parser = argparse.ArgumentParser(description='PCA on audio and text embeddings')
parser.add_argument('--components', type=int, default=2, help='Number of components for dimensionality reduction')
parser.add_argument('--audio_path', type=str, default='/root/share/clap-surveypaper/audio_embeddings', help='Path to audio embeddings')
parser.add_argument('--text_path', type=str, default='/root/share/clap-surveypaper/text_embeddings', help='Path to text embeddings')
parser.add_argument("--category_labels", type=bool, default=True, help="Whether to replace labels with categories")
args = parser.parse_args()

def load_audio_data(directory):
    if directory in ['audiomatches']:
        return []
    elif directory == "Audioset":
        audioset = []
        for subdir in ['balanced_train_segments', 'eval_segments', 'unbalanced_train_segments']:
            with open(f"{args.audio_path}/{directory}/{subdir}/0.pkl", "rb") as f:
                data = pickle.load(f)
                audioset.extend(data)
        return audioset
    if os.path.exists(f"{args.audio_path}/{directory}/0.pkl"):
        with open(f"{args.audio_path}/{directory}/0.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    return []

def load_text_data(directory):
    if directory in ['audiomatches']:
        return []
    if os.path.exists(f"{args.text_path}/{directory}"):
        with open(f"{args.text_path}/{directory}", "rb") as f:
            data = pickle.load(f)
        return data
    return []

def get_all_category_dfs():
    dfs = []
    for f in os.listdir("output_cats/"):
        if f.endswith(".csv"):
            df = pd.read_csv(f"output_cats/{f}")
            df["file_name"] = df["file_name"].astype(str)
            dfs.append(df)
    df = pd.concat(dfs)
    return df

def replace_labels(x):
    if x.startswith("/storage/data/"):
        x = x.replace("/storage/data/", "")
    elif x.startswith("/scratch-shared/gwijngaard/"):
        x = x.replace("/scratch-shared/gwijngaard/", "")
    if args.category_labels:
        x = x.replace("data/", "").replace("dataset/", "").replace("audio/", "").replace("audios/", "")
        x = x.replace("storage/", "").replace("gwijngaard/", "").replace("train/", "").replace("val/", "")
        x = x.replace("test/", "")
        if x.startswith("SoundingEarth/aporee"):
            x = "_".join(x.split("_")[:-1])
        x = x.replace("FSD50K.dev_", "FSD50K.dev_audio/").replace("FSD50K.eval_", "FSD50K.eval_audio/")
    if "unbalanced_train_segments" in x:
        x = x.replace("unbalanced_train_segments", "Audioset")
    elif "TextToAudioGrounding" in x:
        x = x.replace("TextToAudioGrounding", "AudioGrounding")
    elif "balanced_train_segments" in x:
        x = x.replace("balanced_train_segments", "Audioset")
    elif "eval_segments" in x:
        x = x.replace("eval_segments", "Audioset")
    return x

def process_embeddings(data_path, load_data_func):
    directories = os.listdir(data_path)
    all_data = list(map(load_data_func, directories))
    all_data = [item for sublist in all_data for item in sublist]
    labels, embeddings = zip(*all_data)
    if args.category_labels:
        assert all(["/" in l for l in labels])
        labels = [replace_labels(x) for x in labels]
        splitted = [l.split("/") for l in labels]
        dataset = [s[0] for s in splitted]
        labels = ["/".join(s[1:]) for s in splitted]
        dataset_w_labels = pd.DataFrame(list(zip(dataset, labels)), columns=["dataset", "label"])
        category_df = get_all_category_dfs()
        new_df = dataset_w_labels.reset_index().merge(category_df, left_on=["label", "dataset"], right_on=["file_name", "dataset"], how="left")
        new_df = new_df.drop_duplicates("index").sort_values("index")
        new_df["label_y"] = new_df["label_y"].fillna("Unknown")
        labels = new_df["label_y"].values
    else:
        labels = [replace_labels(x).split("/")[0] for x in labels]
    return np.array(labels), np.array(embeddings)

def plot_category_distribution(labels, title, ax):
    label_counts = Counter(labels)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    categories, counts = zip(*sorted_labels)
    
    ax.bar(categories, counts, color=cm.tab20(np.arange(len(categories))))
    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xlabel('Category')
    ax.set_xticklabels(categories, rotation=90)

audio_labels, audio_embeddings = process_embeddings(args.audio_path, load_audio_data)
text_labels, text_embeddings = process_embeddings(args.text_path, load_text_data)

print(f"Audio embeddings shape: {audio_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")

# Combine all labels to ensure consistent coloring
all_labels = np.concatenate([audio_labels, text_labels])
label_counts = Counter(all_labels)
sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
cmap = cm.get_cmap('tab20', len(sorted_labels))
label_to_color = {label: cmap(i) for i, label in enumerate(sorted_labels)}

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

plot_category_distribution(audio_labels, "Audio Data Category Distribution", axs[0])
plot_category_distribution(text_labels, "Text Data Category Distribution", axs[1])

# Create a combined legend
custom_lines = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label], markersize=10) for label in sorted_labels]
fig.legend(custom_lines, sorted_labels, title="Labels", loc='center right', bbox_to_anchor=(1.1, 0.5))

plt.tight_layout()
is_categories = "categories" if args.category_labels else "labels"
plt.savefig(f"pca/combined_pca_plots_{is_categories}_bar.png", dpi=150, bbox_inches='tight')
plt.close()
