import os
import sys
import argparse
import logging

import torch
import numpy as np
from torch import nn

from src.utils import load_config, set_seed
from src.dataset import create_dataloaders
from src.model import build_model
from src.trainer import train, validate
from src.visualize import generate_all_visualizations

from src.matrics import (
    compute_metrics,
    get_classification_report,
    compute_confusion_matrix,
)


# --------------------------------------------------------------------------- 
# Logging Setup
# --------------------------------------------------------------------------- 
def setup_global_logger(log_file: str = "training.log") -> logging.Logger:
    """
    Configure root logger to output to both console and file.
    This ensures logging works even if the logger was previously initialized.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


logger = setup_global_logger("training.log")


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Log key metrics in a consistent format.
    """
    logger.info("%sAccuracy      : %.4f", prefix, metrics.get("accuracy", 0))
    logger.info("%sRecall_Fake   : %.4f", prefix, metrics.get("recall_fake", 0))
    logger.info("%sPrecision_Fake: %.4f", prefix, metrics.get("precision_fake", 0))
    logger.info("%sF1_Fake       : %.4f", prefix, metrics.get("f1_fake", 0))


def main():
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(
        description="ISOT Fake News Detection — BiLSTM Training Pipeline"
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

    # --- 2. Fix random seeds ---
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
    logger.info("Building BiLSTM model …")
    model = build_model(config, vocab_size, word2idx)

    # --- 6. Train model ---
    logger.info("Starting training …")
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # --- 7. Evaluate on Test Set ---
    logger.info("=" * 70)
    logger.info("  Starting Test Set Evaluation (using best model)")
    logger.info("=" * 70)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    logger.info("Running inference on test set …")
    test_loss, test_labels, test_preds = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Compute evaluation metrics
    test_metrics = compute_metrics(test_labels, test_preds)
    test_report = get_classification_report(test_labels, test_preds)
    test_cm = compute_confusion_matrix(test_labels, test_preds)

    logger.info("Test Classification Report:\n%s", test_report)
    logger.info("Test Confusion Matrix (rows=actual, cols=predicted):\n%s", test_cm)

    logger.info("-" * 70)
    logger.info("Test Set Final Metrics:")
    log_metrics(test_metrics, prefix="  ")

    # Save test predictions & labels
    save_dir = config.get("training", {}).get("save_dir", "checkpoints")
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    np.save(os.path.join(save_dir, "test_preds.npy"), test_preds)
    logger.info("Test labels and predictions saved to checkpoints/")

    # --- Final Summary ---
    best = results["best_metrics"]
    logger.info("=" * 70)
    logger.info("  TRAINING & EVALUATION COMPLETE")
    logger.info("  Best epoch              : %d", results["best_epoch"])
    logger.info("  Best Val Recall_Fake    : %.4f  (★ primary metric)", best.get("recall_fake", 0))
    logger.info("  Best Val F1_Fake        : %.4f", best.get("f1_fake", 0))
    logger.info("  Best Val Precision_Fake : %.4f", best.get("precision_fake", 0))
    logger.info("  Best Val Accuracy       : %.4f", best.get("accuracy", 0))
    logger.info("  Best Val Loss           : %.4f", best.get("val_loss", 0))
    logger.info("-" * 70)
    logger.info("  Test Recall_Fake        : %.4f", test_metrics.get("recall_fake", 0))
    logger.info("  Test F1_Fake            : %.4f", test_metrics.get("f1_fake", 0))
    logger.info("  Test Precision_Fake     : %.4f", test_metrics.get("precision_fake", 0))
    logger.info("  Test Accuracy           : %.4f", test_metrics.get("accuracy", 0))
    logger.info("=" * 70)

    logger.info("Model checkpoint      : checkpoints/best_model.pt")
    logger.info("Training history      : checkpoints/training_history.json")
    logger.info("Training log          : training.log")

    # ====================== Visualizations ======================
    history_path = os.path.join(save_dir, "training_history.json")
    logger.info("Generating all visualizations (including Radar Chart)...")

    try:
        # Validate on validation set to get labels & predictions
        val_loss, val_labels, val_preds = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        val_metrics = compute_metrics(val_labels, val_preds)

        # Generate all visualizations (Loss, Combined, Confusion, Bar, Radar)
        generate_all_visualizations(
            history_json_path=history_path,
            val_labels=val_labels,
            val_preds=val_preds,
            test_labels=test_labels,
            test_preds=test_preds,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            save_dir=save_dir
        )
        logger.info("✅ All visualizations generated successfully in '%s'.", save_dir)

    except Exception as e:
        logger.error("Failed to generate visualizations: %s", str(e))
        try:
            from src.visualize import plot_training_history
            plot_training_history(history_json_path=history_path, save_dir=save_dir)
        except Exception as fallback_e:
            logger.error("Fallback visualization also failed: %s", str(fallback_e))

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()