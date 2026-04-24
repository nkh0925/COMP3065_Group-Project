# Reproducing the Experiments and Figures

## 1. Overview

This repository implements a fake news detection system based on a **2-layer Bidirectional LSTM (Bi-LSTM)** with **300-dimensional GloVe embeddings** on the **ISOT Fake News Dataset**. This document explains how to reproduce the full experimental pipeline, including dataset download, training, saved artifacts, and the figures used in the report.

The main reproduction path used for this project is:

1. install dependencies,
2. configure Kaggle credentials,
3. download the dataset,
4. run `run_project.ipynb` from top to bottom,
5. inspect the outputs in the `checkpoints/` directory.

---

## 2. Repository Structure

```text
COMP3065_Group-Project/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── glove/
├── src/
│   ├── dataset.py
│   ├── matrics.py
│   ├── model.py
│   ├── trainer.py
│   ├── utils.py
│   └── visualize.py
├── checkpoints/
├── download_dataset.py
├── main.py
├── requirements.txt
├── run_project.ipynb
└── training.log
```

---

## 3. Environment Setup

### 3.1 Python and package requirements

Install the required Python packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The project depends on the following libraries:

- `kaggle>=1.6.14`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `python-dotenv>=1.0.0`
- `tqdm>=4.65.0`
- `pyyaml>=6.0`
- `scikit-learn>=1.3.0`
- `torch>=2.0.0`
- `seaborn>=0.12.0`
- `matplotlib>=3.7.0`

### 3.2 Notebook runner

The experiments were reproduced through `run_project.ipynb`. To run the notebook directly, install one of the following tools if needed:

```bash
pip install notebook
```

or

```bash
pip install jupyterlab
```

### 3.3 Optional GPU check

If you want to train with a GPU, verify that PyTorch detects CUDA correctly.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If the command returns `True`, the training notebook can use GPU acceleration. If it returns `False`, the project can still be run on CPU.

---

## 4. Dataset Download

The project uses the **ISOT Fake News Dataset** from Kaggle.

### 4.1 Configure Kaggle credentials

Create a `.env` file in the project root with your Kaggle API credentials:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

### 4.2 Download the dataset

Run:

```bash
python download_dataset.py
```

The dataset is downloaded from:
- owner: `emineyetm`
- dataset: `fake-news-detection-datasets`
- https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

### 4.3 Important path note

There is an important path inconsistency in the current project:

- `download_dataset.py` downloads and extracts `True.csv` and `Fake.csv` into `./data/`
- `configs/config.yaml` expects the raw files in `data/raw/`

To make the training pipeline work with the current configuration, move the downloaded CSV files into `data/raw/` after download:

```bash
mkdir -p data/raw
mv data/True.csv data/raw/True.csv
mv data/Fake.csv data/raw/Fake.csv
```

After this step, the raw dataset files should be available at:

- `data/raw/True.csv`
- `data/raw/Fake.csv`

### 4.4 Prepare the GloVe embeddings

The reported experiments use the file below:

```text
data/glove/glove.6B.300d.txt
```

Place `glove.6B.300d.txt` inside `data/glove/` before running the notebook. If this file is missing, the code falls back to random embedding initialization, which means the reproduced results may differ from the reported results.

---

## 5. Configuration Used in the Experiments

The experiments reported in this project were run with the settings in `configs/config.yaml`.

### 5.1 Data and preprocessing

- text column: `text`
- maximum sequence length: `256`
- minimum word frequency: `2`

### 5.2 Dataset split

- training set: `70%`
- validation set: `15%`
- test set: `15%`
- random seed: `42`
- splitting strategy: **stratified split**

### 5.3 Data loading

- batch size: `64`
- num workers: `2`
- pin memory: `true`

### 5.4 Embeddings and model

- GloVe directory: `data/glove`
- GloVe dimension: `300`
- embedding dimension: `300`
- hidden dimension: `128`
- number of LSTM layers: `2`
- bidirectional: `true`
- number of classes: `2`
- embedding dropout: `0.3`
- LSTM dropout: `0.3`
- fully connected dropout: `0.5`
- embeddings frozen: `false`

### 5.5 Training

- number of epochs: `20`
- optimizer: `Adam`
- learning rate: `0.001`
- weight decay: `1e-5`
- early stopping patience: `5`
- scheduler: `ReduceLROnPlateau`
  - factor: `0.5`
  - patience: `3`
  - minimum learning rate: `1e-6`

---

## 6. How to Reproduce the Experiments

### Recommended workflow

The most faithful way to reproduce the project is to run the notebook:

```text
run_project.ipynb
```

Open the notebook in Jupyter Notebook, JupyterLab, or VS Code and execute all cells from top to bottom.

This is the recommended path because it reproduces the exact training and visualization workflow used in the project.

### What the notebook does

The notebook workflow should perform the following steps:

1. load the project configuration,
2. load and merge `True.csv` and `Fake.csv`,
3. assign labels (`0` for real news, `1` for fake news),
4. perform stratified train/validation/test splitting,
5. build the vocabulary on the training set only,
6. save `word2idx.json` to `data/processed/`,
7. convert texts into padded token sequences,
8. initialize the Bi-LSTM model,
9. load GloVe embeddings if available,
10. train the model,
11. save the best checkpoint and training history,
12. generate evaluation plots.

### Expected output files

After a successful run, the following files should be created in `checkpoints/`:

- `best_model.pt`
- `training_history.json`
- `loss_curve.png`
- `recall_curve.png`
- `combined_loss_recall.png`
- `val_confusion_matrix.png`
- `test_confusion_matrix.png`
- `metrics_bar_chart.png`
- `metrics_radar_chart.png`
- `test_labels.npy`
- `test_preds.npy`

---

## 7. Reproducing the Figures

The visualization utilities are implemented in `src/visualize.py`.

### 7.1 Figures generated by the project

The codebase contains functions to generate the following figures:

- **Loss curve**: training loss vs. validation loss
- **Recall curve**: fake-news recall over epochs
- **Combined curve**: loss and fake-news recall on a dual-axis figure
- **Confusion matrices**: validation and test heatmaps
- **Metrics bar chart**: class-wise validation vs. test precision, recall, and F1
- **Radar chart**: class-wise validation vs. test performance

### 7.2 Important note about the radar chart

Although `src/visualize.py` includes a radar chart function and the file `metrics_radar_chart.png` can be generated, **this figure was not used in the final report** because its visual presentation was not satisfactory for formal analysis.

## 8. Final Recommendation

For anyone reproducing this work, the most reliable procedure is:

1. install the dependencies,
2. configure the Kaggle API,
3. download the dataset,
4. move the CSV files into `data/raw/`,
5. ensure the GloVe file is present in `data/glove/`,
6. run `run_project.ipynb` from start to finish,
7. collect the generated model checkpoint, JSON logs, and figures from `checkpoints/`.

