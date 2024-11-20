import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import polars as pl
import argparse
from collections import Counter
import matplotlib.cm as cm
import seaborn as sns
from numba import jit
import itertools
from cuml.metrics import pairwise_distances
tqdm.pandas()
sns.set_theme()

parser = argparse.ArgumentParser(description='PCA on audio and text embeddings')
parser.add_argument('--components', type=int, default=2, help='Number of components for dimensionality reduction')
parser.add_argument('--audio_path', type=str, default='/home/gwijngaard/g1/audio_embeddings', help='Path to audio embeddings')
parser.add_argument('--text_path', type=str, default='/home/gwijngaard/g1/text_embeddings', help='Path to text embeddings')
parser.add_argument("--remove_audioset", action=argparse.BooleanOptionalAction, default=True, help="Whether to remove audioset")
parser.add_argument("--calculate_distances", action=argparse.BooleanOptionalAction, default=True, help="Whether to calculate distances")
# parser.add_argument("--category_labels", default=False, he`lp="Whether to replace labels with categories")
args = parser.parse_args()

import cudf
from cuml.decomposition import PCA as cumlPCA
import cupy as cp


def capt(filename):
    if "multiple_source_sound/" in filename:
        return filename.replace("multiple_source_sound/", "mixture_sound/")
    else:
        fn_removed = filename.split("/")[-1]
        first_part = "_".join(fn_removed.split("_")[:-2])
        rest = "/".join(filename.split("/")[:-2])
        return f"{rest}/{first_part}/{fn_removed}"
    
def load_audio_data(directory):
    data = []
    if directory in ['audiomatches']:
        return []
    elif directory == "Audioset":
        audioset = []
        for subdir in ['balanced_train_segments', 'eval_segments', 'unbalanced_train_segments']:
            with open(f"{args.audio_path}/{directory}/{subdir}/0.pkl", "rb") as f:
                audios = pickle.load(f)
                audioset.extend(audios)
        return audioset
    if os.path.exists(f"{args.audio_path}/{directory}/0.pkl"):
        with open(f"{args.audio_path}/{directory}/0.pkl", "rb") as f:
            data = pickle.load(f)
    if directory == "FAVDBench":
        # row["label"] = f'{row["label"]}.wav'
        # df["file_name"] = df["file_name"] + ".wav"
        data = [(x + ".wav", y) for x, y in data]
    elif directory == "CAPTDURE":
        data = [(capt(x), y) for x, y in data]
    
    return data

def load_text_data(directory):
    data = []
    if directory in ['audiomatches']:
        return []        
    if os.path.exists(f"{args.text_path}/{directory}"):
        with open(f"{args.text_path}/{directory}", "rb") as f:
            data = pickle.load(f)
        
    return data

def get_all_category_dfs():
    dfs = []
    for f in os.listdir("output_cats/"):
        if f.endswith(".csv"):
            df = pd.read_csv(f"output_cats/{f}")
            df["file_name"] = df["file_name"].astype(str)
    
            dfs.append(df)
    df = pd.concat(dfs)
    return df

def compute_chunk_distances(group1, group2_chunk):
    group1_gpu = group1.to_cupy()
    group2_chunk_gpu = group2_chunk.to_cupy()
    distances = pairwise_distances(group1_gpu, group2_chunk_gpu, metric='euclidean')
    return cp.sum(distances), distances.size

def calculate_average_distance(group1, group2, label1, label2, chunk_size=1000):
    # We chunk the data because the GPU memory is limited
    total_distance = 0.0
    total_count = 0

    # Process group2 in chunks
    for i in tqdm(range(0, group2.shape[0], chunk_size), desc=f"Between {label1} and {label2}", position=1, leave=False):
        group2_chunk = group2[i:i+chunk_size]
        chunk_distance, chunk_count = compute_chunk_distances(group1, group2_chunk)
        total_distance += chunk_distance.get()  # Transfer from GPU to CPU
        total_count += chunk_count

    # Calculate the average distance
    average_distance = total_distance / total_count
    return average_distance

def average_distance_to_centroid(group):
    group = group.to_cupy()
    centroid = cp.mean(group, axis=0)
    distances = cp.linalg.norm(group - centroid, axis=1)
    return cp.mean(distances).get(), centroid.get()


def replace_labels(x, is_category):
    if x.startswith("/storage/data/"):
        x = x.replace("/storage/data/", "")
    elif x.startswith("/scratch-shared/gwijngaard/"):
        x = x.replace("/scratch-shared/gwijngaard/", "")
    if is_category:
        x = x.replace("data/", "").replace("dataset/", "").replace("audio/", "").replace("audios/", "")
        x = x.replace("storage/", "").replace("gwijngaard/", "").replace("train/", "").replace("val/", "")
        x = x.replace("test/", "").replace("evaluation/", "").replace("development/", "").replace("validation/", "")
        x = x.replace("audio_files/", "")
        if x.startswith("SoundingEarth/aporee") and len(x.split("_")) > 3:
            x = "_".join(x.split("_")[:-1])
        if x.startswith("FAVDBench/") and x.endswith(".wav.wav"):
            x = x.replace(".wav.wav", "")
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
    embeddings = np.array(embeddings)

    assert all(["/" in l for l in labels])
    categories = [replace_labels(x, True) for x in labels]
    splitted = [l.split("/") for l in categories]
    dd = [s[0] for s in splitted]
    categories = ["/".join(s[1:]) for s in splitted]
    dataset_w_labels = pd.DataFrame(list(zip(dd, categories)), columns=["dataset", "label"])
    category_df = get_all_category_dfs()
    if args.remove_audioset:
        category_df = category_df[category_df["dataset"] != "Audioset"]
    new_df = dataset_w_labels.reset_index().merge(category_df, left_on=["label", "dataset"], right_on=["file_name", "dataset"], how="inner")
    new_df = new_df.drop_duplicates("index").sort_values("index")
    new_df["label_y"] = new_df["label_y"].fillna("Unknown")
    categories = new_df["label_y"].values
    embeddings = embeddings[new_df["index"].values]
    labels = new_df["dataset"].values
    
    # There are in total 18 different datasets we want to include in our visualisation
    assert len(new_df["dataset"].unique()) == 18, f"Unique datasets: {new_df['dataset'].unique()}"


    return labels, categories, embeddings

def plot_pca(labels, embeddings, title, ax, label_to_color):
    embeddings_dask = cudf.DataFrame.from_records(embeddings)
    reducer = cumlPCA(n_components=args.components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings_dask)
    
    label_variability = {}
    for label in label_to_color.keys():
        group = reduced_embeddings[labels == label]
        variability, centroid = average_distance_to_centroid(group)
        label_variability[label] = variability
        ax.scatter(centroid[0], centroid[1], color=label_to_color[label], s=150, zorder=40, marker="*", edgecolors='black', linewidths=0.5)
    
    if args.calculate_distances and not os.path.exists(f"pca/{title.lower().replace(' ', '-')}_distances.csv"):
        distances = []
        for label1, label2 in tqdm(itertools.combinations(label_to_color.keys(), 2), 
                                total=len(list(itertools.combinations(label_to_color.keys(), 2))), 
                                desc=f"Calculating distances {title}", position=0):
            group1 = reduced_embeddings[labels == label1]
            group2 = reduced_embeddings[labels == label2]
            distance = calculate_average_distance(group1, group2, label1, label2)
            distances.append([label1, label2, distance])
        
        distances_df = pd.DataFrame(distances, columns=["label1", "label2", "distance"])
        distances_df = distances_df.sort_values("distance", ascending=True)
        distances_df.to_csv(f"pca/{title.lower().replace(' ', '-')}_distances.csv", index=False)
            
        
    reduced_embeddings = reduced_embeddings.to_pandas().values


    for idx, label in enumerate(sorted(label_to_color.keys(), key=lambda x: label_to_color[x])):
        mask = labels == label
        ax.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], color=label_to_color[label], label=f'{label}', s=0.01, zorder=idx+3)

    
    label_variability_df = pd.DataFrame(label_variability.items(), columns=["label", "variability"])
    label_variability_df = label_variability_df.sort_values("variability", ascending=False)
    label_variability_df.to_csv(f"pca/{title.lower().replace(' ', '-')}.csv", index=False)


    explained_variance = reducer.explained_variance_ratio_
    ax.set_xlabel(f"Component 1 ({explained_variance[0]*100:.2f}%)")
    ax.set_ylabel(f"Component 2 ({explained_variance[1]*100:.2f}%)")
    ax.set_title(title)

    unique_labels = list(label_to_color.keys())
    return unique_labels

def plot_pca_plot(audio_labels, text_labels, audio_embeddings, text_embeddings, is_categories=False):
    # Combine all labels to ensure consistent coloring
    all_labels = np.concatenate([audio_labels, text_labels])
    label_counts = Counter(all_labels)
    sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
    cmap = cm.get_cmap('tab20', len(sorted_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(sorted_labels)}

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    lbl = "Categories" if is_categories else "Labels"
    unique_labels = plot_pca(audio_labels, audio_embeddings, f"PCA of Audio Data Embeddings With {lbl}", axs[0], label_to_color)
    unique_labels = plot_pca(text_labels, text_embeddings, f"PCA of Text Data Embeddings With {lbl}", axs[1], label_to_color)

    # Create a combined legend
    custom_lines = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_to_color[label], markersize=10) for label in sorted_labels]
    if is_categories:
        bbox = 1.175
    else:
        bbox = 1.09
    fig.legend(custom_lines, unique_labels, title="Labels", loc='center right', bbox_to_anchor=(bbox, 0.5))


    plt.tight_layout()
    is_categories = "categories" if is_categories else "labels"
    is_audioset = "noaudioset" if args.remove_audioset else "audioset"
    plt.savefig(f"pca/combined_pca_plots_{is_categories}_{bbox}_{is_audioset}.png", dpi=150, bbox_inches='tight')
    plt.close()

audio_labels, audio_categories, audio_embeddings = process_embeddings(args.audio_path, load_audio_data)
assert audio_labels.shape[0] == audio_embeddings.shape[0]
print(f"Audio embeddings shape: {audio_embeddings.shape}")

text_labels, text_categories, text_embeddings = process_embeddings(args.text_path, load_text_data)
assert text_labels.shape[0] == text_embeddings.shape[0]
print(f"Text embeddings shape: {text_embeddings.shape}")

plot_pca_plot(audio_categories, text_categories, audio_embeddings, text_embeddings, is_categories=True)
plot_pca_plot(audio_labels, text_labels, audio_embeddings, text_embeddings, is_categories=False)
