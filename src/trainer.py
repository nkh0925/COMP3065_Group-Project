"""
trainer.py — Training & validation engine for the ISOT Fake News Detector.

Key design decisions
--------------------
* **Primary optimisation target**: Fake-news Recall (class 1) on the
  validation set.  The model checkpoint with the **highest fake recall**
  is saved as ``best_model.pt``.  When recall is tied, the checkpoint
  with the higher F1-Fake is preferred.
* **Early stopping** monitors validation loss by default, but can be
  switched to monitor ``recall_fake`` via config.
* Validation is performed every epoch; the **test set is never touched**
  during training or tuning.
* All random seeds are fixed at the very beginning for full reproducibility.

Author : COMP3065 Group
"""

import os
import time
import json
import copy
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.matrics import compute_metrics, log_metrics, get_classification_report, compute_confusion_matrix

logger = logging.getLogger(__name__)


# ======================================================================
# Early Stopping
# ======================================================================

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change in the monitored metric to qualify
                   as an improvement.
        mode: ``'min'`` for loss, ``'max'`` for recall/F1.
        verbose: Whether to log early-stopping events.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Update the stopper with a new score.

        Args:
            score: The current epoch's metric value.

        Returns:
            ``True`` if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: no improvement for %d / %d epochs",
                    self.counter, self.patience,
                )
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ======================================================================
# Single-epoch routines
# ======================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one full training epoch.

    Args:
        model: The LSTM model.
        loader: Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimiser instance.
        device: ``torch.device`` to use.

    Returns:
        ``(avg_loss, all_labels, all_preds)``
    """
    model.train()
    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for batch_idx, (seqs, labels, lengths) in enumerate(loader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(seqs, lengths)       # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to stabilise LSTM training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds)

    avg_loss = total_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return avg_loss, all_labels, all_preds


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run validation (no gradient computation).

    Args:
        model: The LSTM model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: ``torch.device`` to use.

    Returns:
        ``(avg_loss, all_labels, all_preds)``
    """
    model.eval()
    total_loss = 0.0
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    for seqs, labels, lengths in loader:
        seqs = seqs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(seqs, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * seqs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds)

    avg_loss = total_loss / len(loader.dataset)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return avg_loss, all_labels, all_preds


# ======================================================================
# Full Training Loop
# ======================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> Dict:
    """Full training loop with validation, early stopping, and model saving.

    Model selection rule (on validation set):
        1. **Primary**: highest fake-news recall (``recall_fake``)
        2. **Tie-break**: highest F1-fake

    Args:
        model: Initialised ``FakeNewsLSTM``.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Project configuration dictionary.
        device: ``torch.device``.

    Returns:
        Dictionary with training history and best metrics.
    """
    train_cfg = config.get("training", {})

    # --- Hyperparameters from config ---
    num_epochs     = train_cfg.get("num_epochs", 20)
    learning_rate  = train_cfg.get("learning_rate", 1e-3)
    weight_decay   = train_cfg.get("weight_decay", 1e-5)
    optimizer_name = train_cfg.get("optimizer", "adam")
    patience       = train_cfg.get("early_stopping_patience", 5)
    save_dir       = train_cfg.get("save_dir", "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # --- Loss function ---
    # Since misclassifying fake as real is costlier, we can optionally
    # weight the classes. By default, equal weight.
    class_weight = train_cfg.get("class_weight", None)
    if class_weight is not None:
        weight_tensor = torch.tensor(class_weight, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info("Using weighted CrossEntropyLoss: %s", class_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Optimiser ---
    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    logger.info(
        "Optimizer: %s | LR: %s | Weight decay: %s",
        optimizer_name, learning_rate, weight_decay,
    )

    # --- Learning rate scheduler ---
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_factor   = scheduler_cfg.get("factor", 0.5)
    scheduler_patience = scheduler_cfg.get("patience", 3)
    scheduler_min_lr   = scheduler_cfg.get("min_lr", 1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )

    # --- Early stopping (monitors fake recall) ---
    early_stopper = EarlyStopping(patience=patience, mode="max")

    # --- Training history ---
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "lr": [],
    }

    best_recall_fake = 0.0
    best_f1_fake = 0.0
    best_epoch = 0
    best_model_state = None
    best_metrics = {}

    logger.info("=" * 70)
    logger.info("Starting training for %d epochs on %s", num_epochs, device)
    logger.info("=" * 70)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        train_loss, train_labels, train_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics = compute_metrics(train_labels, train_preds)

        # --- Validate ---
        val_loss, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(val_labels, val_preds)

        # --- LR Scheduler step ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        elapsed = time.time() - epoch_start

        # --- Log ---
        logger.info("-" * 70)
        logger.info(
            "Epoch %d/%d (%.1fs) | LR: %.2e | Train Loss: %.4f | Val Loss: %.4f",
            epoch, num_epochs, elapsed, current_lr, train_loss, val_loss,
        )
        log_metrics(train_metrics, prefix="  Train")
        log_metrics(val_metrics, prefix="  Val  ")

        # --- Record history ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)
        history["lr"].append(current_lr)

        # --- Model selection: prioritise recall_fake, then f1_fake ---
        recall_fake = val_metrics["recall_fake"]
        f1_fake = val_metrics["f1_fake"]

        is_better = False
        if recall_fake > best_recall_fake:
            is_better = True
        elif recall_fake == best_recall_fake and f1_fake > best_f1_fake:
            is_better = True

        if is_better:
            best_recall_fake = recall_fake
            best_f1_fake = f1_fake
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            best_metrics["val_loss"] = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            # Save checkpoint
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": best_metrics,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info(
                "  ★ New best model saved (epoch %d) — "
                "Recall_Fake: %.4f | F1_Fake: %.4f | Acc: %.4f",
                epoch, recall_fake, f1_fake, val_metrics["accuracy"],
            )

        # --- Early stopping ---
        if early_stopper(recall_fake):
            logger.info(
                "Early stopping triggered at epoch %d. "
                "Best epoch: %d",
                epoch, best_epoch,
            )
            break

    # --- Restore best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model weights from epoch %d", best_epoch)

    # --- Final summary ---
    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(
        "Best epoch: %d | Recall_Fake: %.4f | F1_Fake: %.4f | Acc: %.4f",
        best_epoch,
        best_metrics.get("recall_fake", 0),
        best_metrics.get("f1_fake", 0),
        best_metrics.get("accuracy", 0),
    )
    logger.info("=" * 70)

    # --- Print full classification report for best validation ---
    val_loss_final, val_labels_final, val_preds_final = validate(
        model, val_loader, criterion, device
    )
    report = get_classification_report(val_labels_final, val_preds_final)
    cm = compute_confusion_matrix(val_labels_final, val_preds_final)
    logger.info("Validation Classification Report (best model):\n%s", report)
    logger.info("Confusion Matrix (rows=actual, cols=predicted):\n%s", cm)

    # --- Save training history ---
    history_path = os.path.join(save_dir, "training_history.json")
    # Convert numpy values to native Python types for JSON serialisation
    serialisable_history = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "lr": history["lr"],
        "train_metrics": [
            {k: float(v) for k, v in m.items()} for m in history["train_metrics"]
        ],
        "val_metrics": [
            {k: float(v) for k, v in m.items()} for m in history["val_metrics"]
        ],
        "best_epoch": best_epoch,
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serialisable_history, f, indent=2)
    logger.info("Training history saved to '%s'", history_path)

    return {
        "model": model,
        "history": serialisable_history,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }
