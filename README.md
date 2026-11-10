# Music Genre Classification — Messy Mashup

A deep-learning pipeline for classifying audio mashups into one of 10 music genres. The project was developed for the **January 2026 DLGenAI “Messy Mashup”** challenge and is implemented in a Kaggle notebook.

The pipeline generates synthetic training examples by recombining isolated audio stems, adds environmental noise and audio augmentations, trains multiple audio classifiers, and combines their predictions for final inference.

## Problem Statement

Given an audio mashup, predict its music genre. The notebook supports these 10 classes:

- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Approach

### 1. Synthetic training-data generation

The notebook creates training examples from the provided genre-separated stems:

- Combines stems from multiple songs within a genre.
- Applies random gain, tempo variation, and pitch shifting.
- Occasionally introduces a low-volume stem from another genre to make examples more challenging.
- Adds environmental noise sampled from the ESC-50 audio collection.
- Converts the resulting audio into log-mel spectrogram features.

The notebook is configured to generate up to **30,000 synthetic samples** across the 10 genres.

### 2. Audio representation and augmentation

- Sample rate: 22,050 Hz
- Clip duration: 10 seconds
- Mel frequency bins: 128
- FFT size: 2,048
- Hop length: 512
- Training-time SpecAugment and MixUp are used to improve generalisation.

### 3. Model training

The notebook trains three complementary neural-network classifiers:

| Model | Role |
|---|---|
| ResNet-50 | CNN classifier operating on mel spectrograms |
| EfficientNet-B2 | CNN classifier operating on mel spectrograms |
| Audio Spectrogram Transformer (AST) | Transformer-based audio classifier, initialized from pretrained AudioSet weights |

### 4. Prediction ensemble

The three models' output probabilities are combined using a CatBoost stacking classifier. Test-time augmentation (TTA) is also applied during inference to reduce prediction variance. A weighted probability blend is included as a fallback path in the notebook.

### 5. Submission validation

The notebook checks that the generated submission contains the expected `id` and `label` columns, has the same number of rows as the test data, and contains only valid genre labels.

## Repository Structure

```text
Music-Genre-Classificattion/
├── README.md
└── kaggle_notebook.ipynb
```

The notebook creates intermediate data, model checkpoints, and the submission file when executed. These generated artifacts are not included in this repository.

## Dataset

The notebook expects the challenge dataset to be available in the Kaggle environment. It references the following data:

- `genres_stems/` — genre/song folders containing audio stems such as `drums.wav`, `vocals.wav`, `bass.wav`, and `others.wav`.
- `ESC-50-master/audio/` — environmental noise audio files.
- `mashups/` — test mashup audio files.
- `test.csv` — test sample metadata.

The original dataset is not included in this GitHub repository. You must have access to the competition dataset and attach it to your Kaggle notebook. If your Kaggle dataset mount path differs, update `STEMS_DIR`, `NOISE_DIR`, `MASHUPS_DIR`, and `TEST_CSV` in the notebook.

## Requirements

The first notebook cell installs several dependencies. The main libraries used include:

- Python
- NumPy
- pandas
- PyTorch and torchvision
- librosa
- soundfile
- Transformers
- timm
- CatBoost
- scikit-learn
- tqdm

A CUDA-enabled GPU is recommended. Training all models can require substantial GPU memory, disk space, and execution time. The notebook selects CUDA when available and otherwise falls back to CPU; CPU execution will be significantly slower.

## How to Run

1. Open [Kaggle](https://www.kaggle.com/) and create a new notebook.
2. Attach the Messy Mashup challenge dataset and ensure the ESC-50 noise files are accessible.
3. Import or upload `kaggle_notebook.ipynb`.
4. Check and update the dataset paths in the configuration cell.
5. Select a GPU accelerator in the notebook settings if available.
6. Run the notebook cells in order, allowing synthetic data generation and model training to complete.
7. Run inference and submission validation.
8. Download the generated `submission.csv` from the Kaggle working directory and submit it to the competition.

## Reproducibility and Configuration

The notebook sets a random seed of `42` for Python, NumPy, and PyTorch. Key configuration values—including the sample count, batch sizes, training epochs, audio parameters, and TTA count—are defined near the beginning of the notebook and can be adjusted based on available compute.

Results can vary with the dataset version, runtime environment, hardware, and training configuration. No performance score is claimed in this README; use the competition leaderboard and the notebook's validation outputs for the measured result of a particular run.

## Notes

- The dataset and trained model checkpoints are not committed to this repository.
- The notebook is designed around Kaggle input paths and may need path changes in other environments.
- Review the dataset and pretrained-model licences and competition rules before redistributing data or model artifacts.

## Author

**Sandhya Gupta**

