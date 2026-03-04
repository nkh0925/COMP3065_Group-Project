"""
matrics.py — Evaluation metrics for the ISOT Fake News Detection project.

The project requirement emphasises that **misclassifying fake news as real
has severe consequences**. Therefore the primary optimisation target is:

    ★ Fake-news Recall (Sensitivity for class 1)
      = TP_fake / (TP_fake + FN_fake)

Secondary metrics tracked:
    - Precision (Fake)  — how many flagged-as-fake are truly fake
    - F1-Score (Fake)   — harmonic mean of Precision & Recall for fake class
    - Overall Accuracy  — fraction of all correct predictions
    - Confusion Matrix  — full TP/FP/TN/FN breakdown

Label convention:
    0 = Real (truthful) news
    1 = Fake news  ← the *positive* class for recall optimisation

Author : COMP3065 Group
"""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics for binary fake-news classification.

    Args:
        y_true: Ground-truth labels, shape ``(N,)``. 0=Real, 1=Fake.
        y_pred: Predicted labels, shape ``(N,)``. 0=Real, 1=Fake.

    Returns:
        Dictionary with keys:
            ``accuracy``, ``precision_fake``, ``recall_fake``, ``f1_fake``,
            ``precision_real``, ``recall_real``, ``f1_real``.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # --- Fake news (positive class = 1) ---
        "precision_fake": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fake":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fake":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        # --- Real news (class = 0) ---
        "precision_real": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_real":    recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_real":        f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute the 2×2 confusion matrix.

    Layout::

                        Predicted
                     Real    Fake
        Actual Real [ TN      FP ]
              Fake  [ FN      TP ]

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        np.ndarray of shape ``(2, 2)``.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Generate a human-readable classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Formatted string with per-class and overall metrics.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        digits=4,
        zero_division=0,
    )


def log_metrics(
    metrics: Dict[str, float],
    prefix: str = "Val",
) -> None:
    """Log metrics in a readable format.

    Args:
        metrics: Dictionary returned by ``compute_metrics``.
        prefix: Label prefix (e.g. "Train", "Val", "Test").
    """
    logger.info(
        "%s Metrics — Acc: %.4f | "
        "Fake[P: %.4f  R: %.4f  F1: %.4f] | "
        "Real[P: %.4f  R: %.4f  F1: %.4f]",
        prefix,
        metrics["accuracy"],
        metrics["precision_fake"],
        metrics["recall_fake"],
        metrics["f1_fake"],
        metrics["precision_real"],
        metrics["recall_real"],
        metrics["f1_real"],
    )