"""
dataset.py — Data loading, splitting, and PyTorch Dataset / DataLoader for
the ISOT Fake News Detection project.

Pipeline overview
-----------------
1. ``load_and_merge_data``  — Read True.csv & Fake.csv, add labels, merge.
2. ``split_dataset``        — Stratified train / val / test split (BEFORE
                              any vocabulary fitting to prevent data leakage).
3. ``build_vocab``          — Build vocabulary on training texts ONLY (in utils.py).
4. ``FakeNewsDataset``      — Custom ``torch.utils.data.Dataset``.
5. ``create_dataloaders``   — End-to-end factory returning three DataLoaders.

Anti-data-leakage guarantee
---------------------------
* The dataset is split **before** vocabulary construction.
* ``build_vocab`` is called **only** on the training split.
* The resulting ``word2idx`` is applied uniformly to train / val / test.

Author : COMP3065 Group
"""

import os
import json
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils import (
    load_config,
    set_seed,
    clean_text,
    tokenize,
    build_vocab,
    texts_to_sequences,
    PAD_IDX,
)

logger = logging.getLogger(__name__)

# ======================================================================
# 1. Data Loading
# ======================================================================

def load_and_merge_data(
    raw_dir: str,
    true_csv: str = "True.csv",
    fake_csv: str = "Fake.csv",
) -> pd.DataFrame:
    """Load True.csv and Fake.csv, add label columns, and merge into one
    shuffled DataFrame.

    Labels:
        - Real (truthful) news → ``0``
        - Fake news            → ``1``

    Args:
        raw_dir: Directory containing the CSV files.
        true_csv: Filename of the real-news CSV.
        fake_csv: Filename of the fake-news CSV.

    Returns:
        Merged and shuffled ``pd.DataFrame`` with an added ``label`` column.

    Raises:
        FileNotFoundError: If either CSV file is missing.
    """
    true_path = os.path.join(raw_dir, true_csv)
    fake_path = os.path.join(raw_dir, fake_csv)

    if not os.path.isfile(true_path):
        raise FileNotFoundError(f"True-news CSV not found: {true_path}")
    if not os.path.isfile(fake_path):
        raise FileNotFoundError(f"Fake-news CSV not found: {fake_path}")

    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # Assign labels: Real=0, Fake=1
    true_df["label"] = 0
    fake_df["label"] = 1

    # Merge & shuffle
    merged = pd.concat([true_df, fake_df], axis=0, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Drop rows with empty/NaN text (about 3 % of fake news)
    text_col = "text"  # will be parameterised later via config
    original_len = len(merged)
    merged = merged.dropna(subset=[text_col])
    merged = merged[merged[text_col].str.strip().astype(bool)].reset_index(drop=True)
    dropped = original_len - len(merged)
    if dropped > 0:
        logger.warning("Dropped %d rows with empty/NaN text.", dropped)

    logger.info(
        "Loaded %d samples (Real: %d | Fake: %d)",
        len(merged),
        (merged["label"] == 0).sum(),
        (merged["label"] == 1).sum(),
    )
    return merged

# ======================================================================
# 2. Stratified Splitting
# ======================================================================

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train / validation / test split.

    The split is done **before** any vocabulary or preprocessor fitting to
    guarantee no information leaks from val/test into training.

    Strategy:
        1. Split off ``test_size`` fraction as the test set.
        2. From the remainder, split off ``val_size / (1 - test_size)`` as
           the validation set.

    Args:
        df: Full merged DataFrame with a ``label`` column.
        test_size: Fraction of data for the test set.
        val_size: Fraction of data for the validation set.
        random_seed: Seed for reproducibility.

    Returns:
        ``(train_df, val_df, test_df)`` — three DataFrames.
    """
    # Step 1: separate test set
    remaining, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Step 2: separate validation set from remainder
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        remaining,
        test_size=adjusted_val_size,
        stratify=remaining["label"],
        random_state=random_seed,
    )

    logger.info(
        "Split sizes — Train: %d (%.1f%%) | Val: %d (%.1f%%) | Test: %d (%.1f%%)",
        len(train_df), 100 * len(train_df) / len(df),
        len(val_df),   100 * len(val_df)   / len(df),
        len(test_df),  100 * len(test_df)  / len(df),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# ======================================================================
# 3. Custom PyTorch Dataset
# ======================================================================

class FakeNewsDataset(Dataset):
    """PyTorch Dataset for the ISOT Fake News Detection task.

    Each sample is a tuple ``(sequence, label, length)`` where:
        - ``sequence`` : ``torch.LongTensor`` of shape ``(max_seq_len,)``
        - ``label``    : ``torch.LongTensor`` scalar (0 = Real, 1 = Fake)
        - ``length``   : ``torch.LongTensor`` scalar (actual token count
                         before padding, capped at ``max_seq_len``)
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray,
    ) -> None:
        """
        Args:
            sequences: Integer-encoded, padded sequences — shape ``(N, max_len)``.
            labels: Binary labels — shape ``(N,)``.
            lengths: Actual sequence lengths — shape ``(N,)``.
        """
        super().__init__()
        self.sequences = torch.from_numpy(sequences).long()
        self.labels = torch.from_numpy(labels).long()
        self.lengths = torch.from_numpy(lengths).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# ======================================================================
# 4. End-to-End DataLoader Factory
# ======================================================================

def create_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int]:
    """Build train / val / test DataLoaders from raw CSV files.

    Full pipeline:
        1. Load & merge CSVs  →  2. Stratified split  →  3. Build vocab
        (train only)  →  4. Encode all splits  →  5. Wrap in DataLoader.

    Args:
        config: Configuration dictionary (as returned by ``load_config``).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader, word2idx, vocab_size)``.
    """
    # ---- Unpack config ----
    data_cfg   = config["data"]
    pre_cfg    = config["preprocessing"]
    split_cfg  = config["split"]
    dl_cfg     = config["dataloader"]

    raw_dir       = data_cfg["raw_dir"]
    processed_dir = data_cfg["processed_dir"]
    true_csv      = data_cfg["true_csv"]
    fake_csv      = data_cfg["fake_csv"]

    text_col   = pre_cfg["text_column"]
    max_len    = pre_cfg["max_seq_len"]
    min_freq   = pre_cfg["min_word_freq"]

    test_size  = split_cfg["test_size"]
    val_size   = split_cfg["val_size"]
    seed       = split_cfg["random_seed"]

    batch_size   = dl_cfg["batch_size"]
    num_workers  = dl_cfg["num_workers"]
    pin_memory   = dl_cfg["pin_memory"]

    # ---- Reproducibility ----
    set_seed(seed)

    # ---- Step 1: Load & merge ----
    logger.info("=" * 60)
    logger.info("Step 1 / 5 — Loading and merging raw data …")
    logger.info("=" * 60)
    df = load_and_merge_data(raw_dir, true_csv, fake_csv)

    # ---- Step 2: Split BEFORE fitting anything ----
    logger.info("=" * 60)
    logger.info("Step 2 / 5 — Stratified dataset splitting …")
    logger.info("=" * 60)
    train_df, val_df, test_df = split_dataset(df, test_size, val_size, seed)

    # ---- Step 3: Build vocabulary on TRAINING set only ----
    logger.info("=" * 60)
    logger.info("Step 3 / 5 — Building vocabulary (train-only) …")
    logger.info("=" * 60)
    train_texts = train_df[text_col].tolist()
    word2idx, idx2word = build_vocab(train_texts, min_freq=min_freq)
    vocab_size = len(word2idx)

    # Save word2idx for reproducibility / downstream use
    os.makedirs(processed_dir, exist_ok=True)
    w2i_path = os.path.join(processed_dir, "word2idx.json")
    with open(w2i_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    logger.info("Saved word2idx (%d entries) to '%s'", vocab_size, w2i_path)

    # ---- Step 4: Encode all splits ----
    logger.info("=" * 60)
    logger.info("Step 4 / 5 — Encoding sequences …")
    logger.info("=" * 60)

    train_seqs, train_lens = texts_to_sequences(
        train_df[text_col].tolist(), word2idx, max_len
    )
    val_seqs, val_lens = texts_to_sequences(
        val_df[text_col].tolist(), word2idx, max_len
    )
    test_seqs, test_lens = texts_to_sequences(
        test_df[text_col].tolist(), word2idx, max_len
    )

    train_labels = train_df["label"].values.astype(np.int64)
    val_labels   = val_df["label"].values.astype(np.int64)
    test_labels  = test_df["label"].values.astype(np.int64)

    # ---- Step 5: Build DataLoaders ----
    logger.info("=" * 60)
    logger.info("Step 5 / 5 — Creating DataLoaders …")
    logger.info("=" * 60)

    train_dataset = FakeNewsDataset(train_seqs, train_labels, train_lens)
    val_dataset   = FakeNewsDataset(val_seqs,   val_labels,   val_lens)
    test_dataset  = FakeNewsDataset(test_seqs,  test_labels,  test_lens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # no shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,         # no shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "DataLoaders ready — "
        "Train batches: %d | Val batches: %d | Test batches: %d | "
        "Vocab size: %d",
        len(train_loader), len(val_loader), len(test_loader), vocab_size,
    )

    return train_loader, val_loader, test_loader, word2idx, vocab_size

