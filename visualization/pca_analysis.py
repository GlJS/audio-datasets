import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Read the original text embeddings CSV file
df_text_orig = pd.read_csv('/gpfs/work4/0/einf6190/audio-datasets/pca/clap-text-embeddings-pca_original_variability.csv')

# Calculate Pearson correlation
pearson_corr, pearson_p = stats.pearsonr(df_text_orig['variability'], df_text_orig['num_sounds'])


# Check statistical significance (p < 0.05)
pearson_significant = pearson_p < 0.05

print(f"Original text embeddings:")
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Pearson p-value: {pearson_p:.3f}")
print(f"Pearson correlation is{' ' if pearson_significant else ' not '}statistically significant")

# Read the PCA text embeddings CSV file
df_text_pca = pd.read_csv('/gpfs/work4/0/einf6190/audio-datasets/pca/clap-text-embeddings-pca_pca_variability.csv')

# Calculate Pearson correlation
pearson_corr, pearson_p = stats.pearsonr(df_text_pca['variability'], df_text_pca['num_sounds'])


# Check statistical significance (p < 0.05)
pearson_significant = pearson_p < 0.05

print(f"\nPCA text embeddings:")
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Pearson p-value: {pearson_p:.3f}")
print(f"Pearson correlation is{' ' if pearson_significant else ' not '}statistically significant")

# Read the original audio embeddings CSV file
df_audio_orig = pd.read_csv('/gpfs/work4/0/einf6190/audio-datasets/pca/clap-audio-embeddings-pca_original_variability.csv')

# Calculate Pearson correlation
pearson_corr, pearson_p = stats.pearsonr(df_audio_orig['variability'], df_audio_orig['num_sounds'])


# Check statistical significance (p < 0.05)
pearson_significant = pearson_p < 0.05

print(f"\nOriginal audio embeddings:")
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Pearson p-value: {pearson_p:.3f}")
print(f"Pearson correlation is{' ' if pearson_significant else ' not '}statistically significant")

# Read the PCA audio embeddings CSV file
df_audio_pca = pd.read_csv('/gpfs/work4/0/einf6190/audio-datasets/pca/clap-audio-embeddings-pca_pca_variability.csv')

# Calculate Pearson correlation
pearson_corr, pearson_p = stats.pearsonr(df_audio_pca['variability'], df_audio_pca['num_sounds'])


# Check statistical significance (p < 0.05)
pearson_significant = pearson_p < 0.05

print(f"\nPCA audio embeddings:")
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Pearson p-value: {pearson_p:.3f}")
print(f"Pearson correlation is{' ' if pearson_significant else ' not '}statistically significant")