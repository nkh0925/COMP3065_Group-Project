"""
model.py — LSTM-based Neural Network for Fake News Detection.

Architecture
------------
    Input (token IDs, lengths)
        │
        ▼
    Embedding Layer  (vocab_size × embed_dim, optional GloVe init)
        │
        ▼
    Embedding Dropout
        │
        ▼
    Bidirectional LSTM  (num_layers, hidden_dim)
        │
        ▼
    LSTM Output Dropout
        │
        ▼
    Fully Connected Layer  (hidden_dim × 2  →  num_classes)
        │
        ▼
    Output Logits  (batch_size, num_classes)

Design rationale
----------------
* **Bidirectional LSTM** captures both forward and backward context in
  news articles, which is critical for detecting subtle linguistic cues
  of fake news (e.g. sensational phrasing, inconsistencies).
* **Multi-layer LSTM** (default 2 layers) learns hierarchical
  representations — the lower layer captures local patterns while the
  upper layer captures longer-range dependencies.
* **Dropout** is applied at three points (embedding, between LSTM layers,
  and on the LSTM output) to mitigate overfitting, which is a common
  concern given the large vocabulary and relatively homogeneous news text.
* **Last-hidden-state extraction** uses the actual sequence lengths
  (via ``pack_padded_sequence``) so padding tokens do not influence the
  representation.
* The final representation concatenates the last hidden states from
  **both directions** [h_forward; h_backward] before classification.

Author : COMP3065 Group
"""

import os
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


# ======================================================================
# GloVe Embedding Loader
# ======================================================================

def load_glove_embeddings(
    glove_dir: str,
    glove_dim: int,
    word2idx: dict,
) -> torch.Tensor:
    """Load pre-trained GloVe vectors and build an embedding matrix aligned
    with the project vocabulary.

    Args:
        glove_dir: Directory containing the GloVe text file
                   (e.g. ``glove.6B.100d.txt``).
        glove_dim: Dimensionality of the GloVe vectors (50/100/200/300).
        word2idx: Word-to-index mapping from ``build_vocab``.

    Returns:
        ``torch.FloatTensor`` of shape ``(vocab_size, glove_dim)`` where
        rows corresponding to words found in GloVe are initialised with
        the pre-trained vectors, and the rest are randomly initialised.
    """
    glove_file = os.path.join(glove_dir, f"glove.6B.{glove_dim}d.txt")
    if not os.path.isfile(glove_file):
        logger.warning(
            "GloVe file not found at '%s'. "
            "Falling back to random initialisation.",
            glove_file,
        )
        return None

    logger.info("Loading GloVe vectors from '%s' …", glove_file)
    glove_vectors: dict = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                vector = np.array(parts[1:], dtype=np.float32)
                glove_vectors[word] = vector

    # Build the embedding matrix
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(
        scale=0.6, size=(vocab_size, glove_dim)
    ).astype(np.float32)

    # Index 0 (<PAD>) should be all zeros
    embedding_matrix[0] = np.zeros(glove_dim)

    matched = 0
    for word, idx in word2idx.items():
        if word in glove_vectors:
            embedding_matrix[idx] = glove_vectors[word]
            matched += 1

    coverage = 100.0 * matched / vocab_size
    logger.info(
        "GloVe coverage: %d / %d words (%.1f%%)",
        matched, vocab_size, coverage,
    )
    return torch.from_numpy(embedding_matrix)


# ======================================================================
# LSTM Model
# ======================================================================

class FakeNewsLSTM(nn.Module):
    """Bidirectional LSTM classifier for binary fake-news detection.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (including <PAD> and <UNK> tokens).
    embed_dim : int
        Dimensionality of the token embedding vectors.
    hidden_dim : int
        Number of features in each LSTM hidden state.  The actual
        representation is ``hidden_dim × 2`` when ``bidirectional=True``.
    num_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes (default 2: Real / Fake).
    dropout_embed : float
        Dropout probability applied after the embedding layer.
    dropout_lstm : float
        Dropout probability applied between LSTM layers (only active
        when ``num_layers > 1``).
    dropout_fc : float
        Dropout probability applied before the fully connected output
        layer.
    bidirectional : bool
        Whether the LSTM reads the sequence in both directions.
    pad_idx : int
        Index of the ``<PAD>`` token (embedding for this index is
        frozen at zero).
    pretrained_embeddings : torch.Tensor or None
        Optional pre-trained embedding matrix to initialise the
        embedding layer (e.g. GloVe).
    freeze_embeddings : bool
        If ``True`` and ``pretrained_embeddings`` is provided, the
        embedding weights are frozen (not updated during training).

    Input
    -----
    sequences : torch.LongTensor, shape ``(batch, max_seq_len)``
        Padded integer-encoded token sequences.
    lengths : torch.LongTensor, shape ``(batch,)``
        Actual (pre-padding) lengths of each sequence.

    Output
    ------
    logits : torch.FloatTensor, shape ``(batch, num_classes)``
        Raw (un-normalised) class scores.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_embed: float = 0.3,
        dropout_lstm: float = 0.3,
        dropout_fc: float = 0.5,
        bidirectional: bool = True,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ----- Embedding Layer -----
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            logger.info("Embedding layer initialised with pre-trained vectors.")
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                logger.info("Embedding weights are FROZEN (not trainable).")

        # ----- Dropout after Embedding -----
        self.embed_dropout = nn.Dropout(p=dropout_embed)

        # ----- LSTM Encoder -----
        # When num_layers > 1, PyTorch applies dropout between layers automatically
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        # ----- Dropout before FC -----
        self.fc_dropout = nn.Dropout(p=dropout_fc)

        # ----- Fully Connected Output Layer -----
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, num_classes)

        # Log model summary
        logger.info(
            "FakeNewsLSTM initialised — "
            "vocab: %d | embed: %d | hidden: %d | layers: %d | "
            "bidir: %s | classes: %d | "
            "dropout(embed=%.2f, lstm=%.2f, fc=%.2f)",
            vocab_size, embed_dim, hidden_dim, num_layers,
            bidirectional, num_classes,
            dropout_embed, dropout_lstm, dropout_fc,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        lengths: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Forward pass.

        Dimension flow (assuming bidirectional=True, D=2)::

            sequences : (B, L)           — B=batch, L=max_seq_len
                │
            Embedding : (B, L, E)        — E=embed_dim
                │
            embed_drop: (B, L, E)
                │
            pack       → PackedSequence
                │
            LSTM       → PackedSequence  — hidden: (num_layers×D, B, H)
                │
            extract last hidden states from both directions
                │
            concat  : (B, H×2)
                │
            fc_drop : (B, H×2)
                │
            FC      : (B, C)            — C=num_classes

        Args:
            sequences: Padded token IDs, shape ``(batch, max_seq_len)``.
            lengths: Actual lengths, shape ``(batch,)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        batch_size = sequences.size(0)

        # --- 1. Embedding ---
        # (B, L) → (B, L, E)
        embedded = self.embedding(sequences)
        embedded = self.embed_dropout(embedded)

        # --- 2. Pack padded sequences ---
        # Clamp lengths to [1, max_seq_len] to avoid zero-length errors
        lengths_clamped = lengths.clamp(min=1).cpu()

        packed = pack_padded_sequence(
            embedded,
            lengths_clamped,
            batch_first=True,
            enforce_sorted=False,  # no need to pre-sort by length
        )

        # --- 3. LSTM ---
        # packed_output: PackedSequence
        # h_n: (num_layers × D, B, H)
        # c_n: (num_layers × D, B, H)
        packed_output, (h_n, c_n) = self.lstm(packed)

        # --- 4. Extract final hidden state ---
        # Take the hidden state from the LAST layer
        if self.bidirectional:
            # h_n shape: (num_layers*2, B, H)
            # Forward direction last layer: h_n[-2]  → (B, H)
            # Backward direction last layer: h_n[-1] → (B, H)
            h_forward = h_n[-2]   # (B, H)
            h_backward = h_n[-1]  # (B, H)
            hidden = torch.cat([h_forward, h_backward], dim=1)  # (B, H*2)
        else:
            hidden = h_n[-1]  # (B, H)

        # --- 5. Dropout + FC ---
        hidden = self.fc_dropout(hidden)
        logits = self.fc(hidden)  # (B, num_classes)

        return logits


# ======================================================================
# Model Factory
# ======================================================================

def build_model(
    config: dict,
    vocab_size: int,
    word2idx: Optional[dict] = None,
) -> FakeNewsLSTM:
    """Instantiate FakeNewsLSTM from a configuration dictionary.

    If GloVe vectors are available and ``word2idx`` is provided, the
    embedding layer will be initialised with pre-trained weights.

    Args:
        config: Project configuration (from ``load_config``).
        vocab_size: Size of the vocabulary (from ``create_dataloaders``).
        word2idx: Optional word-to-index mapping for GloVe loading.

    Returns:
        An initialised ``FakeNewsLSTM`` model.
    """
    model_cfg = config.get("model", {})
    embed_cfg = config.get("embedding", {})

    embed_dim       = model_cfg.get("embed_dim", 100)
    hidden_dim      = model_cfg.get("hidden_dim", 128)
    num_layers      = model_cfg.get("num_layers", 2)
    num_classes     = model_cfg.get("num_classes", 2)
    dropout_embed   = model_cfg.get("dropout_embed", 0.3)
    dropout_lstm    = model_cfg.get("dropout_lstm", 0.3)
    dropout_fc      = model_cfg.get("dropout_fc", 0.5)
    bidirectional   = model_cfg.get("bidirectional", True)
    freeze_embed    = model_cfg.get("freeze_embeddings", False)

    # Attempt to load GloVe embeddings
    pretrained = None
    if word2idx is not None:
        glove_dir = embed_cfg.get("glove_dir", "data/glove")
        glove_dim = embed_cfg.get("glove_dim", 100)
        if embed_dim == glove_dim:
            pretrained = load_glove_embeddings(glove_dir, glove_dim, word2idx)
        else:
            logger.warning(
                "embed_dim (%d) ≠ glove_dim (%d). Skipping GloVe init.",
                embed_dim, glove_dim,
            )

    model = FakeNewsLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_embed=dropout_embed,
        dropout_lstm=dropout_lstm,
        dropout_fc=dropout_fc,
        bidirectional=bidirectional,
        pad_idx=0,
        pretrained_embeddings=pretrained,
        freeze_embeddings=freeze_embed,
    )

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters — Total: %s | Trainable: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    return model
