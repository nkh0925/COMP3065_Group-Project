"""
utils.py — Utility functions for the ISOT Fake News Detection project.

Provides:
    - Configuration loading (YAML)
    - Text cleaning & tokenisation
    - Vocabulary building (fitted on training data ONLY to prevent data leakage)
    - Sequence encoding with padding / truncation
    - Reproducibility helpers (seed setting)

Author : COMP3065 Group
"""

import os
import re
import random
import logging
from collections import Counter
from typing import List, Tuple, Dict

import yaml
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A nested dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Basic validation — ensure top-level sections exist
    required_sections = ["data", "preprocessing", "split", "dataloader"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    logger.info("Configuration loaded successfully from '%s'", config_path)
    return config

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour on CUDA (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)

# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a raw news article text string.

    Steps:
        1. Convert to lowercase
        2. Remove Reuters header patterns, e.g. "CITY (Reuters) -"
        3. Remove URLs (http / https / www)
        4. Remove email addresses
        5. Remove HTML tags
        6. Remove non-alphabetic characters (keep spaces)
        7. Collapse multiple whitespace into a single space
        8. Strip leading/trailing whitespace

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Lowercase
    text = text.lower()

    # Remove Reuters-style header: "city (reuters) -"
    text = re.sub(r"^.*?\(reuters\)\s*-\s*", "", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> List[str]:
    """Tokenise cleaned text by whitespace splitting.

    Args:
        text: A cleaned text string (output of ``clean_text``).

    Returns:
        List of word tokens.
    """
    if not text:
        return []
    return text.split()

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(
    texts: List[str],
    min_freq: int = 2,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word-to-index and index-to-word mappings from a list of texts.

    .. important::
        This function must be called **only on the training split** to prevent
        data leakage.  The returned ``word2idx`` mapping is then applied to the
        validation and test sets without modification.

    Args:
        texts: List of raw text strings (will be cleaned & tokenised internally).
        min_freq: Minimum frequency for a word to be included in the vocabulary.

    Returns:
        A tuple ``(word2idx, idx2word)`` where both are dictionaries.
    """
    counter: Counter = Counter()
    for text in texts:
        tokens = tokenize(clean_text(text))
        counter.update(tokens)

    # Filter by minimum frequency
    filtered_words = [w for w, c in counter.items() if c >= min_freq]

    # Build mappings — reserve 0 for <PAD>, 1 for <UNK>
    word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for idx, word in enumerate(sorted(filtered_words), start=2):
        word2idx[word] = idx

    idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}

    logger.info(
        "Vocabulary built: %d unique words (min_freq=%d). "
        "Total vocab size (incl. special tokens): %d",
        len(filtered_words),
        min_freq,
        len(word2idx),
    )
    return word2idx, idx2word

# ---------------------------------------------------------------------------
# Sequence Encoding
# ---------------------------------------------------------------------------

def texts_to_sequences(
    texts: List[str],
    word2idx: Dict[str, int],
    max_len: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of raw texts to padded / truncated integer sequences.

    Each text is cleaned, tokenised, then mapped to indices using ``word2idx``.
    Out-of-vocabulary tokens are replaced with ``<UNK>`` index.

    Args:
        texts: List of raw text strings.
        word2idx: Word-to-index mapping (from ``build_vocab``).
        max_len: Maximum sequence length; longer sequences are truncated,
                 shorter ones are post-padded with ``<PAD>`` index.

    Returns:
        A tuple ``(sequences, lengths)`` where:
            - ``sequences``: np.ndarray of shape ``(N, max_len)`` with int64 dtype.
            - ``lengths``: np.ndarray of shape ``(N,)`` recording the actual
              (pre-padding) length of each sequence, capped at ``max_len``.
    """
    sequences = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)

    for i, text in enumerate(texts):
        tokens = tokenize(clean_text(text))
        # Map tokens to indices (use UNK for unknown words)
        encoded = [word2idx.get(t, UNK_IDX) for t in tokens]

        # Truncate if necessary
        actual_len = min(len(encoded), max_len)
        sequences[i, :actual_len] = encoded[:actual_len]
        lengths[i] = actual_len

    return sequences, lengths