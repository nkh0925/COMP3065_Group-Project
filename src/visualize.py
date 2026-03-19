"""
visualize.py — Visualization utilities for training history.
"""

import os
import json
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def plot_training_history(history_json_path: str, save_dir: str = "checkpoints") -> None:
    """
    Reads the training history JSON file and generates high-quality 
    plots for Loss and Recall.

    Args:
        history_json_path: Path to the saved training_history.json.
        save_dir: Directory where the generated plots will be saved.
    """
    if not os.path.exists(history_json_path):
        logger.error("History file %s not found. Cannot plot.", history_json_path)
        return

    # Load the training history
    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Plot Training & Validation Loss
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="red", marker="s")
    
    plt.title("Training and Validation Loss", fontsize=16, pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # 2. Plot Fake News Recall (Primary Metric)
    # ---------------------------------------------------------
    # Assuming trainer.py saves 'val_recall_fake' in the history dict
    if "val_recall_fake" in history:
        plt.figure(figsize=(8, 6))
        # If train_recall_fake exists, plot it; otherwise just plot validation
        if "train_recall_fake" in history:
            plt.plot(epochs, history["train_recall_fake"], label="Train Recall (Fake)", color="green", marker="o")
        
        plt.plot(epochs, history["val_recall_fake"], label="Validation Recall (Fake)", color="purple", marker="s")
        
        plt.title("Fake News Recall Progression", fontsize=16, pad=15)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Recall Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        recall_plot_path = os.path.join(save_dir, "recall_curve.png")
        plt.savefig(recall_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("✅ Training curves successfully saved to %s", save_dir)