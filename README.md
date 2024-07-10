# Audio-Language Datasets of Scenes and Events: A Survey

Corresponding paper: [https://arxiv.org/abs/2407.06947](https://arxiv.org/abs/2407.06947).

This repository contains various scripts used for creating the survey paper on audio-language datasets. Also, it includes useful splits to mitigate overlap between datasets. Furthermore, we provide a bash script to easily download all the data in `download.sh`.

## Script files

- new_descriptors.py: Processes audio files to compute mel spectrograms and perform audio deduplication.

- new_cosine.py: Calculates cosine similarity between audio embeddings using GPU acceleration.

- datasets.py: Contains functions to load and preprocess various audio datasets.

- clap.py: Extracts CLAP (Contrastive Language-Audio Pretraining) embeddings from audio files.

- main.py: Calculates audio and text statistics for different datasets.

- clap_text_categorize.py: Categorizes text descriptions of audio using a language model.

- barplot.py: Generates bar plots to visualize the distribution of audio and text categories.

- clap_evaluation.py: Evaluates CLAP embeddings and performs various analyses on audio and text data.

 - get_dogs.py: Retrieves of first occurrence of the word dog in each dataset.

 - calc_overlap_mel.py: Creates heatmap of melspectogram overlaps.

 - get_in.py: Generates the new dataset splits based on overlap with other datasets.

## New splits

When training on one dataset, and evaluating on another dataset, the training dataset should not include the ids present in the specific file in the `splits` directory. 
For example, for training on AnimalSpeak and evaluating on AudioCaps, one should remove the audios from the AnimalSpeak dataset that are present in the `animalspeak_in_audiocaps.csv`. 

