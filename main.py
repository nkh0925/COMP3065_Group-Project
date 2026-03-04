"""
main.py — Entry point for the ISOT Fake News Detection pipeline.

Usage:
    python main.py                          # train with default config
    python main.py --config configs/config.yaml  # specify config path

This script orchestrates:
    1. Configuration loading
    2. Random seed fixing
    3. Data loading & preprocessing (Task 6)
    4. Model building (Task 7)
    5. Model training with validation & early stopping (Task 8)

The **test set is NOT evaluated here** — that is reserved for Task 9.

Author : COMP3065 Group
"""

import os
import sys
import argparse
import logging

import torch

from src.utils import load_config, set_seed
from src.dataset import create_dataloaders
from src.model import build_model
from src.trainer import train

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(
        description="ISOT Fake News Detection — LSTM Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # --- 1. Load configuration ---
    logger.info("=" * 70)
    logger.info("  ISOT Fake News Detection — Training Pipeline")
    logger.info("=" * 70)
    config = load_config(args.config)

    # --- 2. Fix all random seeds ---
    seed = config["split"]["random_seed"]
    set_seed(seed)

    # --- 3. Device selection ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # --- 4. Data pipeline ---
    logger.info("Building data pipeline …")
    train_loader, val_loader, test_loader, word2idx, vocab_size = create_dataloaders(config)
    logger.info("Data pipeline ready. Vocab size: %d", vocab_size)

    # --- 5. Build model ---
    logger.info("Building LSTM model …")
    model = build_model(config, vocab_size, word2idx)

    # --- 6. Train ---
    logger.info("Starting training …")
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # --- Summary ---
    best = results["best_metrics"]
    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE")
    logger.info("  Best epoch : %d", results["best_epoch"])
    logger.info("  Recall_Fake: %.4f  (★ primary metric)", best.get("recall_fake", 0))
    logger.info("  F1_Fake    : %.4f", best.get("f1_fake", 0))
    logger.info("  Precision_F: %.4f", best.get("precision_fake", 0))
    logger.info("  Accuracy   : %.4f", best.get("accuracy", 0))
    logger.info("  Val Loss   : %.4f", best.get("val_loss", 0))
    logger.info("=" * 70)
    logger.info("Model checkpoint: checkpoints/best_model.pt")
    logger.info("Training history: checkpoints/training_history.json")
    logger.info("Training log    : training.log")


if __name__ == "__main__":
    main()
