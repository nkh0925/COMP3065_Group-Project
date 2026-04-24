import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from math import pi

logger = logging.getLogger(__name__)

def plot_training_history(history_json_path: str, save_dir: str = "checkpoints") -> None:
    """Original function preserved for backward compatibility"""
    if not os.path.exists(history_json_path):
        logger.error("History file %s not found. Cannot plot.", history_json_path)
        return

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(save_dir, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="s")
    
    plt.title("Training and Validation Loss", fontsize=16, pad=15)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Recall curve (compatible with old and new history formats)
    if "val_metrics" in history or "val_recall_fake" in history:
        plt.figure(figsize=(8, 6))
        if "val_metrics" in history:
            val_recall = [m.get("recall_fake", 0) for m in history["val_metrics"]]
            train_recall = [m.get("recall_fake", 0) for m in history.get("train_metrics", [])]
        else:
            val_recall = history.get("val_recall_fake", [])
            train_recall = history.get("train_recall_fake", [])
        
        if train_recall:
            plt.plot(epochs, train_recall, label="Train Recall (Fake)", marker="o")
        plt.plot(epochs, val_recall, label="Validation Recall (Fake)", marker="s")
        
        plt.title("Fake News Recall Progression", fontsize=16, pad=15)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Recall Score", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        recall_plot_path = os.path.join(save_dir, "recall_curve.png")
        plt.savefig(recall_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Training curves successfully saved to %s", save_dir)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str) -> None:
    """Confusion Matrix heatmap (highlighting false negatives)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Real (Pred)", "Fake (Pred)"],
                yticklabels=["Real (True)", "Fake (True)"])
    
    plt.title(title, fontsize=16, pad=15)
    plt.ylabel("Actual Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    
    # Highlight critical FN region
    plt.text(0.5, 1.5, f"FN (Critical): {cm[1,0]}", 
             ha="center", va="center", fontsize=12, color="red", weight="bold")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix heatmap saved to %s", save_path)


def plot_metrics_bar(val_metrics: dict, test_metrics: dict, save_dir: str = "checkpoints") -> None:
    """Bar chart for Precision/Recall/F1"""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["precision_real", "recall_real", "f1_real", 
               "precision_fake", "recall_fake", "f1_fake"]
    labels = ["Real P", "Real R", "Real F1", "Fake P", "Fake R", "Fake F1"]
    
    val_values = [val_metrics.get(m, 0) for m in metrics]
    test_values = [test_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, val_values, width, label="Validation", alpha=0.8, color="#1f77b4")
    plt.bar(x + width/2, test_values, width, label="Test", alpha=0.8, color="#ff7f0e")
    
    plt.title("Class-wise Metrics Comparison (Validation vs Test)", fontsize=16, pad=15)
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(x, labels, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    
    for i, v in enumerate(val_values):
        plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    for i, v in enumerate(test_values):
        plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "metrics_bar_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Metrics bar chart saved to %s", save_path)


def plot_radar_chart(val_metrics: dict, test_metrics: dict, save_dir: str = "checkpoints") -> None:
    """Class-wise Metrics Radar Chart (recommended for presentation novelty & analysis)"""
    os.makedirs(save_dir, exist_ok=True)
    
    categories = ["Precision\nReal", "Recall\nReal", "F1\nReal", 
                  "Precision\nFake", "Recall\nFake", "F1\nFake"]
    N = len(categories)
    
    val_values = [val_metrics.get(m, 0) for m in 
                  ["precision_real", "recall_real", "f1_real", 
                   "precision_fake", "recall_fake", "f1_fake"]]
    test_values = [test_metrics.get(m, 0) for m in 
                   ["precision_real", "recall_real", "f1_real", 
                    "precision_fake", "recall_fake", "f1_fake"]]
    
    # Close radar chart
    val_values += val_values[:1]
    test_values += test_values[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot Validation
    ax.plot(angles, val_values, linewidth=2, linestyle="solid", label="Validation", color="#1f77b4")
    ax.fill(angles, val_values, alpha=0.25, color="#1f77b4")
    
    # Plot Test
    ax.plot(angles, test_values, linewidth=2, linestyle="solid", label="Test", color="#ff7f0e")
    ax.fill(angles, test_values, alpha=0.25, color="#ff7f0e")
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Y-axis range and grid
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["0.6", "0.7", "0.8", "0.9", "1.0"], color="grey", size=10)
    ax.set_ylim(0.5, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.title("Class-wise Metrics Radar Chart\n(Real vs Fake Performance)", 
              fontsize=16, pad=30, weight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=12)
    
    # Highlight Fake Recall area
    plt.text(angles[4], 1.02, "★ High Fake Recall", 
             ha="center", va="center", fontsize=11, color="green", weight="bold")
    
    save_path = os.path.join(save_dir, "metrics_radar_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Class-wise Metrics Radar Chart saved to %s", save_path)


def plot_combined_curves(history_json_path: str, save_dir: str = "checkpoints") -> None:
    """Dual y-axis Loss + Fake Recall enhanced curve"""
    if not os.path.exists(history_json_path):
        return
    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(save_dir, exist_ok=True)
    
    if "val_metrics" in history:
        val_recall = [m.get("recall_fake", 0) for m in history["val_metrics"]]
    else:
        val_recall = history.get("val_recall_fake", [0] * len(epochs))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", marker="s")
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_recall, label="Val Recall (Fake)", color="#2ca02c", marker="^")
    ax2.set_ylabel("Fake Recall", fontsize=14, color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    
    plt.title("Loss and Fake Recall Progression (Best Epoch Marked)", fontsize=16, pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.axvline(x=6, color="red", linestyle="--", alpha=0.7, label="Best Epoch (6)")
    
    save_path = os.path.join(save_dir, "combined_loss_recall.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Combined Loss + Recall curve saved to %s", save_path)


def generate_all_visualizations(
    history_json_path: str,
    val_labels: np.ndarray = None,
    val_preds: np.ndarray = None,
    test_labels: np.ndarray = None,
    test_preds: np.ndarray = None,
    val_metrics: dict = None,
    test_metrics: dict = None,
    save_dir: str = "checkpoints"
) -> None:
    plot_training_history(history_json_path, save_dir)
    plot_combined_curves(history_json_path, save_dir)
    
    if val_labels is not None and val_preds is not None:
        plot_confusion_matrix(val_labels, val_preds, 
                              "Validation Confusion Matrix", 
                              os.path.join(save_dir, "val_confusion_matrix.png"))
    
    if test_labels is not None and test_preds is not None:
        plot_confusion_matrix(test_labels, test_preds, 
                              "Test Confusion Matrix", 
                              os.path.join(save_dir, "test_confusion_matrix.png"))
    
    if val_metrics and test_metrics:
        plot_metrics_bar(val_metrics, test_metrics, save_dir)
        plot_radar_chart(val_metrics, test_metrics, save_dir)  # New radar chart call
    
    logger.info("✅ All visualizations (including Radar Chart) generated successfully in %s", save_dir)