# Audio-Language Datasets of Scenes and Events: A Survey

Pre-print: https://arxiv.org/abs/2407.06947. An updated version is currently under review.

This repository contains various scripts used for creating the survey paper on audio-language datasets. Also, it includes useful splits to mitigate overlap between datasets. Furthermore, we provide a bash script to easily download all the data in `download.sh`.

We also share all the audio-dataset text files used in the paper on our HuggingFace dataset repository: [https://huggingface.co/datasets/gijs/audio-datasets](https://huggingface.co/datasets/gijs/audio-datasets).

## Script files

- clap.py: Extracts CLAP audio/text embeddings from WebDataset files.

- dataset_to_files.py: Converts audio files to FLAC format with JSON metadata.

- datasets.py: Functions to load and standardize 40+ audio datasets into a common file/caption/split format.

- get_diff.py: Identifies overlapping audio files between datasets using CLAP embeddings similarity analysis.

- get_dogs.py: Checks each dataset's captions for presence of "dog" and prints first matching caption.

- laion-tarify.py: Creates tar archives from audio-text pairs with configurable batch size and data splits.

- main.py: Calculates audio and text statistics for different datasets.

- make_tar_utils.py: Utilities for creating tar archives of dataset files.

### Visualization Scripts

- analysis.py: Performs various analyses on dataset statistics.

- barplot.py: Creates bar plots for visualizing dataset distributions.

- calc_audioset_class_unique.py: Calculates unique classes in AudioSet dataset.

- clap_evaluation_heatmap.py and clap_evaluation_heatmap_combined.py: Generate heatmaps for CLAP embedding similarities.

- clap_evaluation_pca.py: Performs PCA analysis on CLAP embeddings.

- clap_evaluation_tsne.py and clap_evaluation_tsne_batchsearch.py: Performs t-SNE visualization of CLAP embeddings.

- clap_evaluation_umap.py: Generates GPU-accelerated UMAP plots comparing audio and text embeddings across datasets.

- count.py: Counts audio and text embeddings per dataset, including remapped LAION-Audio-630k datasets.

- heatmap_evaluation.py: Generates heatmaps for CLAP embedding similarities.

- pca_analysis.py: Analyzes correlation between dataset size and embedding variability for CLAP embeddings.

## New splits

When training on one dataset, and evaluating on another dataset, the training dataset should not include the ids present in the specific file in the `splits` directory (based on 99\% similarity, we also will upload a 95% similarity split soon).
For example, for training on AudioCaps and evaluating on FAVDBench, one should remove the audios from the AudioCaps dataset that are present in the `audiocaps_in_favdbench.csv`. 
Of course, one should be more careful when training and evaluating on datasets that share the same origin.
